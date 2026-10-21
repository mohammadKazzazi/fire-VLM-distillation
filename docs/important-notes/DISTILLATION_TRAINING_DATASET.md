# D2D distillation — Plan 4 v4 (active recipe)

**Last updated:** 2026-07-09  
**Active JSON:** `LLaVA-Instruct-150K/dense2dense_fire_nonbbox_v4.json` (46,026 rows)  
**Loss:** `kd_lm` only (required for deployable `Yes.` / `No.` prefixes)

---

## Historical note (development ladder — artifacts removed)

During development we ran a multi-plan ablation ladder on Qwen2.5-3B before settling on Plan 4 v4:

| Plan | Purpose | Outcome |
|------|---------|---------|
| 1 | D2D smoke test (2k rows) | Infra OK |
| 2 | Detectium-only baseline (`only_kd`, 13k) | KD works; no Yes/No prefixes |
| 3 | Non-bbox general mix (`kd_lm`, 28k) | Strong env; superseded by v4 |
| 4 v2 | Multi-query + mix665k | Table 4 OK; Yes/No broken |
| 4 v3 | Multi-query synthetic prompts | Table 4 OK; 0/754 prefix fire→Yes |
| 4 **v4** | **Val-aligned multitask** | **Winner — only recipe kept** |
| 5 | Champion LR tune (`2e-4`) | Optional; never beat v4 defaults |
| 6 | `kd_lm` on Plan 2 data | Proved `only_kd` breaks prefixes |

Checkpoints, shells, eval runs, and training JSONs for Plans 1–3, 4 v2/v3, 5, 6, and Gemma2 `only_kd` were **deleted** from the repo. This document records what we tried and why v4 won.

---

## Plan 4 v4 spec

**Question:** Does val-aligned multi-query fire training teach deployable Yes/No + Table 4 reasoning?

| Item | Value |
|------|--------|
| **JSON** | `dense2dense_fire_nonbbox_v4.json` |
| **Rows** | 46,026 |
| **Source** | Detectium train only — seven val prompt templates (4 fire + 3 non-fire) |
| **Loss** | `kd_lm` |
| **LR** | `2e-5` |
| **Epochs** | 1 |
| **Builder** | `LLaVA-Instruct-150K/build_d2d_plan_datasets.py` — `build_plan4_v4()` |

Rebuild v4 only:

```bash
cd /home/ubuntu/llava-test/LLaVA-Instruct-150K
python3 build_d2d_plan_datasets.py --only plan4_v4
```

---

## Training defaults

| Setting | Value |
|---------|--------|
| Script | `shells/train/*/dense2dense_distillation.sh` or `plan4_d2d_multiquery_v4.sh` |
| DeepSpeed | `zero2.json` (torch 2.4) or `zero2_offload.json` (legacy) |
| LoRA | `r=128`, `alpha=256`, `dropout=0.05` |
| `DISTILL_ALL_TOKENS` | `False` |
| `IMAGE_FOLDER` | `/home/ubuntu/llava-test/LLaVA-Instruct-150K` |

Per-backbone wrappers set `OUTPUT_DIR` and student projector paths — see each `shells/train/<family>/README.md`.

---

## Checkpoints (Plan 4 v4 kd_lm)

| Backbone | Checkpoint |
|----------|------------|
| Qwen2.5 0.5B | `plan4-d2d-multiquery-v4-kd-lm-0.5b-seed42/` |
| Qwen2.5 1.5B | `plan4-d2d-multiquery-v4-kd-lm-1.5b-seed42/` |
| Qwen2.5 3B | `plan4-d2d-multiquery-v4-kd-lm-seed42/` |
| Gemma2 2B | `plan4-d2d-multiquery-v4-kd-lm-gemma2-2b-seed42/` |
| Llama3 3B / 1B | `plan4-d2d-multiquery-v4-kd-lm-llama3-{3b,1b}-seed42/` |
| Qwen3 4B / 0.6B | `plan4-d2d-multiquery-v4-kd-lm-qwen3-{4b,0.6b}-seed42/` |

---

## Evaluation

```bash
cd MoD/LLava_MoD
NUM_GPUS=8 bash shells/train/<family>/plan4_d2d_multiquery_v4_val.sh

cd detectioumfire/eval/launchers/qwen2.5
./run_table4_student_validation.sh
```

| Metric | Notes |
|--------|--------|
| Yes/No prefix accuracy | Primary (1,452 val rows) |
| FN / FP | Prioritize low FN for fire detection |
| Table 4 | Burning object / environment / severity (fire-only subset) |

Results: [`D2D_RESULTS.md`](D2D_RESULTS.md) · [`CURRENT_STATUS.md`](CURRENT_STATUS.md)

---

## Key lessons (why v4 won)

1. **`only_kd` does not teach `Yes.` / `No.` prefixes** — use `kd_lm` for all deploy students.
2. **Training prompts must match val templates** — v3 had strong Table 4 but 0/754 prefix fire→Yes.
3. **mix665k caption data hurt Yes/No** — v4 uses Detectium-only multitask rows.
4. **Every prompt must explicitly ask for Yes/No first** — v2 severity prompt caused mixed supervision.

---

## Multi-seed ablation (random seed — NOT train/val splits)

**Manifest:** `LLaVA-Instruct-150K/plan4_v4_seed_ablation.json`

| Term | What it means |
|------|----------------|
| **Random seed** (`42`, `123`, `456`) | Controls LoRA init, dropout, and shuffle order of the **same** 46k training rows (`--seed` in D2D). |
| **Train/val split** | **Already fixed** — `detectium_llavamod_real_full_train.json` → training JSON; `detectium_llavamod_real_full_val.json` (1,452 rows) held out. Not split 3 ways. |
| **3-fold CV** | Different experiment (3 train + 3 val JSONs). Not implemented. |

Existing checkpoints used implicit seed 42 and are renamed `...-seed42/`.

**Fixed:** teacher LoRA, student projector, train JSON, val JSON.  
**Re-run per seed:** D2D only.

```bash
cd MoD/LLava_MoD
# For seeds 123/456 only (42 already done):
SEEDS="123 456" \
BASE_OUTPUT_DIR=./checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-0.6b \
PLAN4_SCRIPT="shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4.sh" \
  bash shells/train/run_plan4_v4_seed_sweep.sh

# All models × remaining seeds (sequential, blocking):
bash shells/train/run_all_plan4_remaining_seeds.sh
# Detached:
nohup bash shells/train/run_all_plan4_remaining_seeds.sh >> logs/plan4_all_seeds_sweep.log 2>&1 &
```
