# DetectiumFire data + evaluation

## Layout

| Path | Purpose |
|------|---------|
| `detectium_llavamod_real_full_{train,val}.json` | Train/val splits (do not train on val) |
| `val_questions.jsonl` | Val prompts for VQA inference |
| `real_images/`, `synthetic_images/` | Image roots (symlinked under `LLaVA-Instruct-150K`) |
| `build_*.py`, `export_val_questions_jsonl.py` | Dataset builders |
| **`eval/`** | All validation tooling and results |

## eval/

```
eval/
  lib/              # score_detectium_val.py, score_fire_reasoning_table4.py, workers
  launchers/        # per-backbone inference scripts (qwen2.5, gemma2, llama3, qwen3)
  runs/
    teacher/        # e.g. qwen2.5-7b-lora/val_answers.jsonl
    students/       # e.g. qwen2.5-0.5b/plan4-v4-kd-lm-seed42/
    baselines/      # symlinks to Evaluation/models/…
  comparisons/      # table4_all_plans.json, table4_all_plans_with_yesno.json
```

**Cross-family results (2026-07-09):** Qwen2.5 + Gemma2 + Llama3 + Qwen3 students in `comparisons/table4_all_plans_with_yesno.json`. Regenerate:

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/detectioumfire/eval/launchers/qwen2.5
./run_table4_student_validation.sh
```

**Qwen3-0.6B edge train + eval:**

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
bash "shells/train/qwen3/student 0.6B/verify_tokenizer.sh"
bash "shells/train/qwen3/student 0.6B/pretrain_student_0.6b.sh"
bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4.sh"
NUM_GPUS=8 bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4_val.sh"
```

Each evaluated checkpoint has its own folder under `runs/` with `val_answers.jsonl` (and optional `metrics.json`, `table4.json`).

## External baselines

Native VL baselines (e.g. Qwen2.5-VL) live under workspace `Evaluation/models/<model_slug>/`.
