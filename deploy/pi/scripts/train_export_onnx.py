#!/usr/bin/env python3
"""
Train a small MobileNetV3-Small fire/non-fire classifier and export ONNX.

Runs on the GPU/server WITH PyTorch. The Raspberry Pi only needs the .onnx file
and onnxruntime (see Deploy/pi/requirements-pi.txt).

Example:
  python scripts/train_export_onnx.py \\
    --fire-dir .../real_fire/images \\
    --nonfire-dir .../real_non_fire \\
    --epochs 3 \\
    --output models/fire_mobilenet_v3_small.onnx
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


class FireFolderDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _collect(fire_dir: Path, nonfire_dir: Path, max_per_class: int) -> list[tuple[Path, int]]:
    fire = sorted(fire_dir.rglob("*.jpg")) + sorted(fire_dir.rglob("*.jpeg")) + sorted(fire_dir.rglob("*.png"))
    nonfire = (
        sorted(nonfire_dir.rglob("*.jpg"))
        + sorted(nonfire_dir.rglob("*.jpeg"))
        + sorted(nonfire_dir.rglob("*.png"))
    )
    fire = fire[:max_per_class]
    nonfire = nonfire[:max_per_class]
    samples = [(p, 1) for p in fire] + [(p, 0) for p in nonfire]
    random.shuffle(samples)
    print(f"Collected fire={len(fire)} non_fire={len(nonfire)} total={len(samples)}")
    if len(fire) < 50 or len(nonfire) < 50:
        raise SystemExit("Need at least ~50 images per class.")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fire-dir", type=Path, required=True)
    parser.add_argument("--nonfire-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/fire_mobilenet_v3_small.onnx"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-per-class", type=int, default=2000)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = _collect(args.fire_dir, args.nonfire_dir, args.max_per_class)
    n_val = max(1, int(len(samples) * args.val_frac))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_loader = DataLoader(
        FireFolderDataset(train_samples, train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        FireFolderDataset(val_samples, val_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += xb.size(0)
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        model.eval()
        v_total, v_correct = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                v_correct += int((pred == yb).sum().item())
                v_total += xb.size(0)
        val_acc = v_correct / max(1, v_total)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.cpu()
    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    meta = {
        "architecture": "mobilenet_v3_small",
        "labels": ["non_fire", "fire"],
        "input_size": args.input_size,
        "preprocess": "imagenet",
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "epochs": args.epochs,
        "seed": args.seed,
        "final_val_acc": val_acc,
        "runtime_on_pi": "onnxruntime (no pytorch)",
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    # Quick ONNX Runtime sanity check on server
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"images": dummy.numpy()})[0]
        print(f"ONNX sanity OK — logits shape {out.shape}")
    except Exception as exc:
        print(f"WARNING: ONNX runtime check skipped/failed: {exc}")

    print(f"Wrote {args.output}")
    print(f"Wrote {meta_path}")
    print("Copy these two files to the Raspberry Pi under Deploy/pi/models/")


if __name__ == "__main__":
    main()
