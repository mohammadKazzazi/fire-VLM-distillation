#!/usr/bin/env python3
"""
LoRA fine-tune Qwen2.5-VL on Detectium LLaVA-format JSON (fire QA).

Default data: detectium_llavamod_real_full_train.json (13,077 rows).
Val split is never used here — evaluate with eval_qwen25_vl_zeroshot.py after training.

Launched via torchrun for multi-GPU DDP (see run_finetune.sh).

Example:
  torchrun --nproc_per_node=8 scripts/finetune_qwen25_vl_detectium.py \\
    --train-json ../LLaVA_MoD/MoD/detectioumfire/detectium_llavamod_real_full_train.json \\
    --image-folder ../LLaVA-Instruct-150K \\
    --output-dir models/qwen25_vl_3b/detectium_lora
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT: Qwen2.5-VL on Detectium")
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--train-json", type=Path, required=True)
    p.add_argument("--image-folder", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-samples", type=int, default=0, help="Debug cap (0 = all).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def init_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def log_main(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def strip_image_prefix(text: str) -> str:
    s = (text or "").strip()
    for prefix in ("<image>\n", "<image>\r\n", "<image>"):
        if s.startswith(prefix):
            return s[len(prefix) :].strip()
    return s


def human_gpt(row: dict) -> tuple[str, str]:
    human = gpt = ""
    for turn in row.get("conversations", []):
        if turn.get("from") == "human":
            human = strip_image_prefix(turn.get("value") or "")
        elif turn.get("from") == "gpt":
            gpt = (turn.get("value") or "").strip()
    return human, gpt


class DetectiumQwenVLDataset(Dataset):
    def __init__(
        self,
        rows: List[dict],
        image_folder: Path,
        processor: AutoProcessor,
    ) -> None:
        self.rows = rows
        self.image_folder = image_folder
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        image_rel = row["image"]
        image_path = self.image_folder / image_rel
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        human, gpt = human_gpt(row)
        if not human or not gpt:
            raise ValueError(f"Row {row.get('id')} missing human/gpt text")

        messages_full = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": human},
                ],
            },
            {"role": "assistant", "content": gpt},
        ]
        messages_prompt = [messages_full[0]]

        text_full = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        text_prompt = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages_full)
        full_inputs = self.processor(
            text=[text_full],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=False,
        )
        prompt_inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=False,
        )

        input_ids = full_inputs.input_ids[0]
        labels = input_ids.clone()
        prompt_len = int(prompt_inputs.input_ids.shape[1])
        labels[:prompt_len] = -100

        item: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": full_inputs.attention_mask[0],
        }
        for key in ("pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos"):
            if key in full_inputs and full_inputs[key] is not None:
                item[key] = full_inputs[key]
        return item


VISION_KEYS = frozenset(
    {"pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos"}
)


def collate_batch(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    if len(batch) == 1:
        out: Dict[str, torch.Tensor] = {}
        for k, v in batch[0].items():
            if isinstance(v, torch.Tensor):
                if k in VISION_KEYS:
                    out[k] = v
                elif v.dim() == 1:
                    out[k] = v.unsqueeze(0)
                else:
                    out[k] = v
            else:
                out[k] = v
        return out

    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids, labels, attention_mask = [], [], []
    for x in batch:
        pad = max_len - x["input_ids"].shape[0]
        input_ids.append(
            torch.cat([x["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)])
        )

    out = {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }
    for key in ("pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos"):
        if key in batch[0]:
            out[key] = torch.cat([x[key] for x in batch], dim=0)
    return out


def load_train_rows(path: Path, max_samples: int) -> List[dict]:
    rows = json.loads(path.read_text())
    valid: List[dict] = []
    for row in rows:
        if not row.get("image"):
            continue
        human, gpt = human_gpt(row)
        if human and gpt:
            valid.append(row)
    if max_samples > 0:
        valid = valid[:max_samples]
    return valid


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)

    if not args.train_json.is_file():
        if is_main:
            print(f"train JSON not found: {args.train_json}", file=sys.stderr)
        sys.exit(1)
    if not args.image_folder.is_dir():
        if is_main:
            print(f"image folder not found: {args.image_folder}", file=sys.stderr)
        sys.exit(1)

    rows = load_train_rows(args.train_json, args.max_samples)
    if not rows:
        if is_main:
            print("No training rows found.", file=sys.stderr)
        sys.exit(1)

    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    effective_batch = args.batch_size * args.grad_accum * world_size

    log_main(rank, f"model:       {args.model_id}")
    log_main(rank, f"distributed: {world_size} GPU(s), rank {rank}, local_rank {local_rank}")
    log_main(rank, f"train rows:  {len(rows)}")
    log_main(rank, f"image_root:  {args.image_folder}")
    log_main(rank, f"output:      {args.output_dir}")
    log_main(rank, f"epochs:      {args.num_epochs}")
    log_main(
        rank,
        f"batch/accum: {args.batch_size} x {args.grad_accum} x {world_size} GPUs "
        f"(effective {effective_batch})",
    )
    log_main(rank, f"lr:          {args.learning_rate}")
    log_main(rank, f"dtype:       {dtype}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()
    model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    dataset = DetectiumQwenVLDataset(rows, args.image_folder, processor)
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=lambda b: collate_batch(b, pad_id),
        num_workers=0,
        pin_memory=True,
    )

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    log_every = min(args.logging_steps, max(1, total_steps))

    log_main(rank, f"optimizer steps/epoch: {steps_per_epoch} (total {total_steps})")
    log_main(rank, f"logging every {log_every} optimizer step(s)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    model.train()
    t0 = time.time()
    running_loss = 0.0
    micro_batches_since_log = 0

    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            is_accum_step = (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader)
            sync_ctx = model.no_sync() if world_size > 1 and not is_accum_step else nullcontext()
            with sync_ctx:
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum
                loss.backward()
            running_loss += float(outputs.loss.detach())
            micro_batches_since_log += 1

            if is_main and (step + 1) % 25 == 0:
                log_main(
                    rank,
                    f"micro-batch {step + 1}/{len(loader)} (epoch {epoch + 1})",
                )

            if is_accum_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0 or global_step == total_steps:
                    elapsed = time.time() - t0
                    avg = running_loss / max(1, micro_batches_since_log)
                    log_main(
                        rank,
                        f"epoch {epoch + 1}/{args.num_epochs} "
                        f"step {global_step}/{total_steps} "
                        f"loss {avg:.4f} "
                        f"lr {scheduler.get_last_lr()[0]:.2e} "
                        f"elapsed {elapsed:.0f}s",
                    )
                    running_loss = 0.0
                    micro_batches_since_log = 0

                if is_main and global_step % args.save_steps == 0:
                    ckpt = args.output_dir / f"checkpoint-{global_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    unwrap_model(model).save_pretrained(ckpt)
                    processor.save_pretrained(ckpt)
                    log_main(rank, f"saved {ckpt}")

    if is_main:
        unwrap_model(model).save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)

        meta = {
            "model_id": args.model_id,
            "train_json": str(args.train_json),
            "image_folder": str(args.image_folder),
            "num_rows": len(rows),
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "world_size": world_size,
            "effective_batch": effective_batch,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "total_steps": global_step,
        }
        (args.output_dir / "train_config.json").write_text(json.dumps(meta, indent=2) + "\n")
        elapsed = time.time() - t0
        log_main(
            rank,
            f"Training done. {global_step} optimizer steps in {elapsed:.1f}s. "
            f"Adapter saved to {args.output_dir}",
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
