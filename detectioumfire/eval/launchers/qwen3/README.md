# Qwen3 validation (Detectium val)

Eval scripts for Qwen3-8B teacher and Qwen3-4B / 0.6B Plan 4 v4 student checkpoints.

**Conv mode:** `qwen3` (default when run from this directory)

## Completed eval runs

| Answers file | Model |
|--------------|-------|
| `runs/teacher/qwen3-8b-lora/val_answers.jsonl` | Teacher LoRA (99.86% Yes/No, 753/754) |
| `runs/students/qwen3-4b/plan4-v4-kd-lm-seed42/val_answers.jsonl` | Student 4B kd_lm (99.31% Yes/No, 752/754, FN=2, FP=8) |

## Full pipeline — Qwen3-0.6B edge (train + eval)

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
bash "shells/train/qwen3/student 0.6B/verify_tokenizer.sh"
bash "shells/train/qwen3/student 0.6B/pretrain_student_0.6b.sh"
tail -f logs/pretrain_qwen3_student_0.6b_latest.log
bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4.sh"
tail -f logs/plan4_d2d_multiquery_v4_kd_lm_qwen3_0.6b_latest.log
NUM_GPUS=8 bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4_val.sh"
cd ../detectioumfire/eval/launchers/qwen2.5 && ./run_table4_student_validation.sh
```

## Eval only

```bash
cd detectioumfire/eval/launchers/qwen3
MODEL_PATH=/path/to/checkpoint MODEL_BASE=Qwen/Qwen3-4B \
  ./eval_detectium_val_parallel.sh
```

Or from the training repo:

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
# 4B
NUM_GPUS=8 bash shells/train/qwen3/plan4_d2d_multiquery_v4_val.sh
# 0.6B edge
NUM_GPUS=8 bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4_val.sh"
```

Default answer paths:
- 4B: `eval/runs/students/qwen3-4b/plan4-v4-kd-lm-seed42/val_answers.jsonl`
- 0.6B: `eval/runs/students/qwen3-0.6b/plan4-v4-kd-lm-seed42/val_answers.jsonl`

`eval_detectium_val_parallel.sh` is a symlink to `../qwen2.5/eval_detectium_val_parallel.sh`; backbone defaults are selected from the launcher directory name.

Student eval uses `~/.cache/huggingface` for Qwen3 student bases (4B, 0.6B); teacher uses repo `cache_dir/`.

## Eval inference note (2026-07-04)

Llama 3 eval exposed a shared eval bug: `model_vqa_loader.py` used the **wrong end-of-turn stop** for new chat templates, so generation ran to `max_new_tokens` and answers looped (~1,200 chars) while the leading `Yes.` / `No.` stayed correct.

**Qwen3** uses `SeparatorStyle.QWEN` with stop token `` (`conv_qwen3.sep2`). The eval path must pass that stop and `eos_token_id` from the tokenizer — not Vicuna `</s>` or `conv_phi` defaults.

Fixes in `llavamod/eval/model_vqa_loader.py`: `get_conv_stop_str()`, `get_generate_token_kwargs()`, `trim_repetitive_answer()`, `MAX_NEW_TOKENS=64`. See [`shells/train/qwen3/README.md`](../../../../LLava_MoD/shells/train/qwen3/README.md) and [`shells/train/llama3/README.md`](../../../../LLava_MoD/shells/train/llama3/README.md) § Eval inference note.

Re-run eval after any template change; expect ~100–180 chars per answer, not 1,000+.
