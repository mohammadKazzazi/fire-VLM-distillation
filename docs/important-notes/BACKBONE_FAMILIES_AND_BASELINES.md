# Backbone families & external VLM baselines

**Last updated:** 2026-07-01  
**Purpose:** Roadmap for (1) **LLaVA-MoD teacher→student** runs across backbone families, and (2) **native VLM baselines** to compare against distilled students after KD.

**Resume:** [`CURRENT_STATUS.md`](CURRENT_STATUS.md) · **Qwen pipeline:** [`README_Qwen2_LLAVA_MoD_Pipeline.md`](README_Qwen2_LLAVA_MoD_Pipeline.md)

---

## Two experimental tracks

| Track | What | When |
|-------|------|------|
| **A — LLaVA-MoD KD** | CLIP ViT-L/336 + `mm_projector` + LM; mimic KD (`kd_lm`) from fire teacher to small student | **Now** — primary method |
| **B — Native VLM baselines** | Off-the-shelf multimodal models (built-in vision encoder); fine-tune or zero-shot on Detectium | **After** best KD students are ready |

Track A answers: *“Does fire specialization transfer via our distillation recipe?”*  
Track B answers: *“How do our distilled students compare to standard open VLMs on the same fire val?”*

Use the **same held-out val** for both tracks: `detectium_llavamod_real_full_val.json` (1,452 rows), Yes/No prefix scoring, Table 4 reasoning — see [`D2D_RESULTS.md`](D2D_RESULTS.md).

---

## Critical rule: tokenizer match (mimic KD)

Mimic distillation aligns **per-token logits**. Teacher and student must share the **same tokenizer and token↔id mapping** within one family.

| Pair | KD-safe? | Lesson |
|------|----------|--------|
| Qwen2.5-7B → Qwen2.5-0.5B / 1.5B / 3B | **Yes** | Primary track |
| Qwen2-7B → Qwen2.5-3B | **No** | Different vocab / IDs — legacy ablation only |
| Qwen2.5-7B → Qwen3-8B | **No** | Different generation — needs **new wrapper** + id-by-id tokenizer proof |
| Gemma2-9B → Gemma2-2B | **Yes** | Same 256,128 vocab |
| Llama-3.1-8B → Llama-3.2-3B | **Yes** | Meta used 3.1 logits as token-level targets when training 3.2 (requires identical tokenizer) |
| Qwen2.5 ↔ Gemma2 ↔ Llama 3 | **No** | Cross-family — separate full pipelines, compare only on val metrics |

**Before any new family or generation:** run the tokenizer gate in [§Tokenizer verification](#tokenizer-verification) on your fire prompt strings.

---

## Track A — Approved backbone families (teacher → student)

Only these four families are in scope for cross-backbone comparison. Each needs its **own** Phase 1–3 pipeline (projectors, teacher LoRA, Plan 4 v4 D2D). Reuse **fire JSONs** and **Plan 4 v4** recipe; do **not** reuse another family’s `mm_projector.bin` or LoRA.

### Summary table

| Priority | Family | Teacher (HF) | Student (HF) | `vocab_size` | Conv / `--version` | Code in `LLava_MoD` | Status |
|----------|--------|--------------|--------------|-------------:|--------------------|---------------------|--------|
| 1 | **Qwen2.5** | `Qwen/Qwen2.5-7B-Instruct` | `Qwen2.5-0.5B` / `1.5B` / `3B-Instruct` | 151,643 | `qwen` | `LlavaQwen2*`, `shells/train/qwen/` | **Primary — Plan 4 v4 done (0.5B, 1.5B, 3B)** |
| 2 | **Gemma 2** | `google/gemma-2-9b-it` | `google/gemma-2-2b-it` | 256,128 | `gemma_2` | `LlavaGemma2*`, `shells/train/gemma2/` | **Done** — Plan 4 v4 kd_lm (2026-07-01) |
| 3 | **Llama 3** | `meta-llama/Llama-3.1-8B-Instruct` | `meta-llama/Llama-3.2-3B-Instruct` · `Llama-3.2-1B-Instruct` | 128,256 | `llama_3` | `LlavaLlama3*`, `shells/train/llama3/` | **Done** — Plan 4 v4 kd_lm 3B + 1B edge (2026-07-05) |
| 4 | **Qwen3** | `Qwen/Qwen3-8B` | `Qwen/Qwen3-4B` · `Qwen/Qwen3-0.6B` | 151,669 | `qwen3` | `LlavaQwen3*`, `shells/train/qwen3/` | **4B done** — 0.6B edge scaffold (2026-07-09) |

**Vision encoder (Track A default):** `openai/clip-vit-large-patch14-336` — used for all completed backbone comparisons.

**Vision encoder ablation (deploy study):** Qwen3-0.6B with CLIP-L / **SigLIP2-B/384** / MobileCLIP-S2 — [`VISION_ENCODER_ROADMAP.md`](VISION_ENCODER_ROADMAP.md).

**Distillation loss:** `LOSS_TYPE=kd_lm` (required for deployable `Yes.` / `No.` prefixes).

**Student data (all families):** `dense2dense_fire_nonbbox_v4.json` (46,026 rows) — [`DISTILLATION_TRAINING_DATASET.md`](DISTILLATION_TRAINING_DATASET.md).

---

### 1. Qwen2.5 (primary)

| Role | Checkpoint / path |
|------|-------------------|
| Teacher LoRA | `checkpoints/llavaqwen2.5-7b-finetune-lora/` → `REF_MLLM` |
| Edge student | `plan4-d2d-multiquery-v4-kd-lm-0.5b-seed42/` |
| 1.5B student | `plan4-d2d-multiquery-v4-kd-lm-1.5b-seed42/` |
| 3B student | `plan4-d2d-multiquery-v4-kd-lm-seed42/` |

Shells: `shells/train/qwen/` · Eval: `detectioumfire/eval/launchers/qwen2.5/` with `--conv-mode qwen`.

---

### 2. Gemma 2 (complete)

| Role | Checkpoint / path |
|------|-------------------|
| Teacher projector | `checkpoints/llavagemma2-9b-pretrain/mm_projector.bin` |
| Teacher LoRA | `checkpoints/llavagemma2-9b-finetune-lora/` — 99.93% Yes/No, 754/754 fire→Yes |
| Student projector | `checkpoints/llavagemma2-2b-student-pretrain/mm_projector.bin` |
| **Student kd_lm** | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-gemma2-2b-seed42/` — 752/754, FN=2, FP=8 |

> **Historical:** Gemma2 `only_kd` ablation was run during development and removed.

Shells: `shells/train/gemma2/` · Eval: `detectioumfire/eval/launchers/gemma2/` · Answers: `eval/runs/students/gemma2-2b/plan4-v4-kd-lm-seed42/`

---

### 3. Llama 3 (complete)

| Role | Checkpoint / path |
|------|-------------------|
| Teacher projector | `checkpoints/llavallama3.1-8b-pretrain/mm_projector.bin` |
| Teacher LoRA | `checkpoints/llavallama3.1-8b-finetune-lora/` — 99.86% Yes/No, 753/754 fire→Yes |
| Student projector (3B) | `checkpoints/llavallama3.2-3b-student-pretrain/mm_projector.bin` |
| **Student kd_lm (3B)** | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-llama3-3b-seed42/` — 752/754, FN=2, FP=2 |
| Student projector (1B) | `checkpoints/llavallama3.2-1b-student-pretrain/mm_projector.bin` |
| **Student kd_lm (1B edge)** | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-llama3-1b-seed42/` — 752/754, FN=2, FP=1 |

Shells: `shells/train/llama3/` (+ `student 1B/` for edge) · Eval: `detectioumfire/eval/launchers/llama3/` · Answers: `eval/runs/students/llama3-{3b,1b}/plan4-v4-kd-lm-seed42/`

| Item | Value |
|------|--------|
| Model code | `llavamod/model/language_model/llava_llama3.py` (transformers `LlamaForCausalLM`) |
| Deps | `pip install -r requirements-backbones-llama3.txt` (torch 2.4 + transformers 4.57) |
| Gated HF | License accepted; `hf auth login` |

---

### 4. Qwen3 (4B done — 0.6B edge scaffold)

| Role | Checkpoint / path |
|------|-------------------|
| Teacher projector | `checkpoints/llavaqwen3-8b-pretrain/mm_projector.bin` |
| Teacher LoRA | `checkpoints/llavaqwen3-8b-finetune-lora/` — 99.86% Yes/No, 753/754 fire→Yes |
| Student projector (4B) | `checkpoints/llavaqwen3-4b-student-pretrain/mm_projector.bin` |
| **Student kd_lm (4B)** | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-4b-seed42/` — 752/754, FN=2, FP=8 |
| Student projector (0.6B) | `checkpoints/llavaqwen3-0.6b-student-pretrain/` — **next** |
| Student kd_lm (0.6B) | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-0.6b-seed42/` — pending |
| Table 4 4B (obj / env / sev) | 70.29% / 79.74% / 89.54% |

Shells: `shells/train/qwen3/` (+ `student 0.6B/` for edge) · Eval: `detectioumfire/eval/launchers/qwen3/` · Answers: `eval/runs/students/qwen3-{4b,0.6b}/plan4-v4-kd-lm-seed42/`

> HF repo IDs: `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-0.6B` (no `-Instruct` suffix). Requires torch ≥ 2.4, transformers ≥ 4.51.

**0.6B edge — run commands:**

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

**4B — already done:**

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
bash shells/train/qwen3/pretrain/pretrain_student.sh
bash shells/train/qwen3/plan4_d2d_multiquery_v4.sh
NUM_GPUS=8 bash shells/train/qwen3/plan4_d2d_multiquery_v4_val.sh
```


### Families explicitly out of scope

| Family | Reason |
|--------|--------|
| Qwen2-7B | Superseded; tokenizer incompatible with Qwen2.5 / Qwen3 students |
| Phi-3 / Phi-4 | Not in comparison set; Phi-4 changed vocab vs Phi-3 |
| Vicuna / Mistral / Falcon / Hunyuan | Not in comparison set |
| Cross-family KD (e.g. Gemma teacher → Qwen student) | Logit alignment invalid |

---

## Track B — Native VLM baselines (compare after KD students)

These models are **not** plugged into `dense2dense_distillation.sh`. They ship with their own vision tower and fusion. Compare them on **Detectium val** against your best KD students (e.g. Plan 4 v4 0.5B / 3B).

### Recommended comparison set

| Model | HF ID (instruct) | ~Params | Vision | Edge? | Notes |
|-------|------------------|--------:|--------|-------|--------|
| **Qwen2.5-VL** | `Qwen/Qwen2.5-VL-3B-Instruct` | 3B | Native | ✅ | Close size to Qwen2.5-3B student; strong open VLM |
| **Qwen2.5-VL** | `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | Native | — | Teacher-scale native baseline |
| **InternVL2.5** | `OpenGVLab/InternVL2_5-4B` | 4B | InternViT | ✅ | Common ACL baseline |
| **InternVL2.5** | `OpenGVLab/InternVL2_5-8B` | 8B | InternViT | — | Larger native baseline |
| **MiniCPM-V** | `openbmb/MiniCPM-V-2_6` | 8B | SigLIP + Qwen2-7B | ✅ (int4) | Edge-focused; LLaVA-MoD paper compares to MiniCPM-V |
| **Gemma 3** | `google/gemma-3-4b-it` | 4B | SigLIP (native) | ✅ | Multimodal out of the box; different tokenizer than Gemma2 |
| **Phi-4-multimodal** | `microsoft/Phi-4-multimodal-instruct` | 5.6B | Native (MoE-LoRA) | ✅ | Compact multimodal; different stack than LLaVA-MoD |
| **LLaVA-OneVision** | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | 7B | Native | — | Modern LLaVA line |
| **PaliGemma 2** | `google/paligemma2-10b-mix-448` | 10B | SigLIP | — | Task-specific; good reference |

### How to run Track B fairly

1. **Same val:** `detectium_llavamod_real_full_val.json` — never train on it.
2. **Same metrics:** Yes/No accuracy, fire→Yes, FN, FP, Table 4 (obj / env / sev) — reuse `detectioumfire/eval/lib/` scorers where possible or adapt prompts per model chat template.
3. **Optional fine-tune:** Fire SFT on `detectium_llavamod_real_full_train.json` (13,077 rows) for each native VLM — matches domain exposure to your D2D student; report zero-shot vs fine-tuned separately.
4. **Report latency / VRAM** on target hardware alongside accuracy (deployment claim).
5. **Do not mix stacks:** Track B numbers are **external baselines**; Track A numbers are **our method**.

### Track B workflow (suggested order)

```
1. ~~Finish best Qwen2.5 KD students (0.5B / 1.5B / 3B Plan 4 v4)~~ **Done**
2. Run Track B zero-shot eval on val for 2–3 edge VLMs (Qwen2.5-VL-3B, InternVL2.5-4B, Gemma-3-4B-it)
3. Optional: fire fine-tune each baseline on Detectium train → re-eval val
4. Add cross-family KD students (~~Gemma2~~, ~~Qwen3~~, ~~Llama3~~) → repeat Track B comparison
5. Paper table: rows = {our students × families} + {native VLM baselines}
```

---

## Future task checklist

### Per new backbone family (Gemma2, Llama3, Qwen3)

**Gemma 2 — all done (2026-07-01):**

- [x] Tokenizer gate passed
- [x] Phase 1 projector pretrain (teacher + student)
- [x] Phase 2 teacher LoRA
- [x] Phase 3 Plan 4 v4 D2D `kd_lm`
- [x] Val eval → [`D2D_RESULTS.md`](D2D_RESULTS.md)

**Llama 3 — all done (2026-07-05):**

- [x] Meta HF license accepted on teacher + student model pages
- [x] Tokenizer gate passed (3B + 1B)
- [x] Model wrapper + shells + eval launchers
- [x] Phase 1a teacher projector pretrain
- [x] Phase 2 teacher LoRA (+ val 99.86%)
- [x] Phase 1b student projector pretrain (3B + 1B)
- [x] Phase 3 Plan 4 v4 D2D with `kd_lm` (3B + 1B edge)
- [x] Val eval → [`D2D_RESULTS.md`](D2D_RESULTS.md)

**Qwen3 — 4B done; 0.6B edge scaffold (2026-07-09):**

- [x] `LlavaQwen3ForCausalLM` + train wiring
- [x] Tokenizer gate passed (8B ↔ 4B)
- [x] Teacher + 4B student pipeline + Plan 4 v4 D2D `kd_lm`
- [x] 4B val eval → [`D2D_RESULTS.md`](D2D_RESULTS.md)
- [x] Tokenizer gate passed (8B ↔ 0.6B)
- [x] 0.6B edge shells in `student 0.6B/`
- [ ] 0.6B pretrain → D2D → eval

### Track B (native VLM baselines)

- [ ] Pick 2–4 models from table above (edge + one 7–8B)
- [ ] Document eval commands under `Evaluation/` or `eval/runs/baselines/`
- [ ] Zero-shot val run → JSON metrics
- [ ] (Optional) Detectium train fine-tune → val run
- [ ] Comparison table vs `plan4-d2d-multiquery-v4-kd-lm-0.5b-seed42/`

---

## Tokenizer verification

Run after `huggingface-cli login` (gated models need accepted licenses):

```bash
python3 << 'EOF'
from transformers import AutoTokenizer

pairs = [
    ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct"),
    ("google/gemma-2-9b-it", "google/gemma-2-2b-it"),
    # Add when integrating Qwen3:
    # ("Qwen/Qwen3-8B-Instruct", "Qwen/Qwen3-4B-Instruct"),
]

prompt = "Carefully analyze the image for signs of fire or smoke."
for t, s in pairs:
    tt = AutoTokenizer.from_pretrained(t)
    st = AutoTokenizer.from_pretrained(s)
    assert tt.get_vocab() == st.get_vocab(), f"VOCAB MISMATCH: {t} vs {s}"
    assert tt.encode(prompt) == st.encode(prompt), f"TOKEN ID MISMATCH: {t} vs {s}"
    print(f"PASS: {t} <-> {s}")
EOF
```

---

## Related documents

| Doc | Content |
|-----|---------|
| [`CURRENT_STATUS.md`](CURRENT_STATUS.md) | Checkpoints done / next command |
| [`README_Qwen2_LLAVA_MoD_Pipeline.md`](README_Qwen2_LLAVA_MoD_Pipeline.md) | Qwen2.5 operational phases 1–3 |
| [`DISTILLATION_TRAINING_DATASET.md`](DISTILLATION_TRAINING_DATASET.md) | Plan 4 v4 JSON spec |
| [`D2D_RESULTS.md`](D2D_RESULTS.md) | Student metrics |
| [`FIRE_MLLM_PROJECT_README.md`](FIRE_MLLM_PROJECT_README.md) | Project strategy & data |
