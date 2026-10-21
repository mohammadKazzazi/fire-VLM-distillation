#!/usr/bin/env python3
"""Fire-only longer-gen pass for fair Table 4 (MobileCLIP student)."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.update(
    HF_HUB_OFFLINE="0",
    TRANSFORMERS_OFFLINE="0",
    CPU_OPT="1",
    CPU_USE_CACHE="1",
    CPU_DTYPE="bfloat16",
)

DEPLOY = Path("/home/ubuntu/llavamod/llava-deploy-h1-h4-20260716/workspace/Deploy")
sys.path.insert(0, str(DEPLOY))
from fire_detect.inference import FireDetector  # noqa: E402

VAL = Path("/home/ubuntu/llavamod/detectium-val-only")
OUT = DEPLOY / "results" / "gguf_val"
ANS = OUT / "val_answers_fire_qwen3_0.6b_mobileclip_pytorch_table4.jsonl"
LORA = (
    "/home/ubuntu/llavamod/qwen3-0.6b-mobileclip-students/checkpoints/"
    "plan4-d2d-multiquery-v4-kd-lm-qwen3-0.6b-mobileclip-s2-seed42"
)
LLAVA = "/home/ubuntu/llavamod/llava-deploy-h1-h4-20260716/workspace/LLaVA_MoD/MoD/LLava_MoD"


def lead(t: str):
    for tok in re.split(r"[^a-zA-Z]+", (t or "").strip().lower()):
        if tok in ("yes", "no"):
            return tok
    return None


def main() -> None:
    val = json.loads((VAL / "json" / "detectium_llavamod_real_full_val.json").read_text())
    # Fire rows: gpt answer starts with Yes
    fire_ids = []
    id_to_img = {}
    id_to_q = {}
    for row in val:
        gpt = ""
        for turn in row.get("conversations", []):
            if turn.get("from") == "gpt":
                gpt = (turn.get("value") or "").strip()
                break
        if not gpt.lower().startswith("yes"):
            continue
        qid = str(row["id"])
        fire_ids.append(qid)
        # image path in row
        img = row.get("image") or ""
        if isinstance(img, list):
            img = img[0]
        id_to_img[qid] = img
        # human question
        human = ""
        for turn in row.get("conversations", []):
            if turn.get("from") == "human":
                human = (turn.get("value") or "").replace("<image>", "").strip()
                break
        id_to_q[qid] = human

    # Prefer val_questions prompts when available
    qmap = {}
    for ln in (VAL / "json" / "val_questions.jsonl").read_text().splitlines():
        if not ln.strip():
            continue
        q = json.loads(ln)
        qmap[str(q["question_id"])] = q

    done = set()
    if ANS.is_file():
        for ln in ANS.read_text().splitlines():
            if ln.strip():
                done.add(str(json.loads(ln)["question_id"]))
    print(f"fire rows={len(fire_ids)} resume={len(done)}", flush=True)

    det = FireDetector(
        model_path=LORA,
        model_base="Qwen/Qwen3-0.6B",
        conv_mode="qwen3",
        device="cpu",
        max_new_tokens=64,
        temperature=0.0,
        llava_mod_root=LLAVA,
    )
    mode = "a" if done else "w"
    t0 = time.time()
    with ANS.open(mode) as fout:
        for i, qid in enumerate(fire_ids):
            if qid in done:
                continue
            q = qmap.get(qid, {})
            img_name = q.get("image") or id_to_img[qid]
            prompt = (q.get("text") or id_to_q[qid] or "").rstrip()
            if "Answer Yes or No" not in prompt:
                prompt = prompt + " Answer Yes or No only."
            img = VAL / "images" / img_name
            t1 = time.time()
            r = det.detect(str(img), prompt=prompt, max_new_tokens=64)
            row = {
                "question_id": qid,
                "image": img_name,
                "prompt": prompt,
                "text": r.raw_text,
                "pred": lead(r.raw_text),
                "latency_ms": round((time.time() - t1) * 1000, 1),
                "backend": "pytorch_mobileclip_bf16_t64",
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0 or i < 3:
                print(
                    f"[{i+1}/{len(fire_ids)}] {qid} {row['latency_ms']:.0f}ms :: {r.raw_text[:90]!r}",
                    flush=True,
                )
    print(f"done answers in {(time.time()-t0)/60:.1f} min", flush=True)

    import subprocess

    t4 = OUT / "table4_fire_qwen3_0.6b_mobileclip_pytorch_t64.json"
    subprocess.check_call(
        [
            sys.executable,
            str(VAL / "eval/score_fire_reasoning_table4.py"),
            "--val-json",
            str(VAL / "json/detectium_llavamod_real_full_val.json"),
            "--model",
            "qwen3-0.6b-mobileclip-pytorch-t64",
            str(ANS),
            "--json-out",
            str(t4),
        ]
    )
    print(t4.read_text(), flush=True)


if __name__ == "__main__":
    main()
