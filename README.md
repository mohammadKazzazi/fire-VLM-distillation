# Fire VLM Distillation

Code for **distilling vision-language models for on-device fire understanding**: teacher fine-tuning, student knowledge distillation (KL + CE), DetectiumFire evaluation, and CPU / edge deployment helpers.

This repository contains **source code and documentation only**. It does **not** include model weights, checkpoints, Hugging Face caches, images, or evaluation run outputs.

## Layout

```text
LLava_MoD/           Training / distillation stack (LLaVA-MoD based)
detectioumfire/      DetectiumFire eval scripts + annotation JSON metadata
deploy/              On-device / llama.cpp export & sensor scripts (no weights)
evaluation/          Extra eval helpers
docs/                Notes, handoff guides, paper drafting notes
```

## What is excluded (on purpose)

- Checkpoints / LoRA adapters / GGUF / safetensors
- `cache_dir/`, HF hub caches, conda / miniconda envs
- Detectium image folders (`real_images/`, `synthetic_images/`)
- Eval outputs (`eval/runs/**/val_answers.jsonl`, metrics dumps)
- Secrets (`.pem`, `.env`)

## Requirements

See `LLava_MoD/requirements.txt` and optional:

- `LLava_MoD/requirements-vision-encoders.txt` (MobileCLIP / OpenCLIP)
- `LLava_MoD/requirements-backbones-llama3-qwen3.txt`

Typical training stack: Python 3.10, PyTorch, transformers, peft, DeepSpeed.

## Quick pointers

| Stage | Entry |
|-------|--------|
| Teacher projector pretrain | `LLava_MoD/shells/train/*/pretrain*.sh` |
| Teacher LoRA SFT | `LLava_MoD/shells/train/*/finetune_lora.sh` |
| Student distillation | `LLava_MoD/shells/train/*/dense2dense_distillation*.sh` |
| Detectium val VQA | `detectioumfire/eval/launchers/*/eval_detectium_val_parallel.sh` |
| Table-style fire reasoning | `detectioumfire/eval/lib/score_fire_reasoning_table4.py` |

Data images and teacher/student JSON mixes are expected under a workspace that also contains `LLaVA-Instruct-150K/` (not shipped here). Use the DetectiumFire release for images and splits.

## License

Upstream LLaVA-MoD license applies to `LLava_MoD/` (see `LLava_MoD/LICENSE`). Project-specific scripts and docs follow the same terms unless noted otherwise.

## Citation

If you use this code, please cite the accompanying paper on distilling VLMs for on-device fire understanding (Detectium / EPFL).
