# Gemma 2, Llama 3 & Qwen3 — implementation scaffold

This document is the **complete record** of files created and code changed to add **Gemma 2**, **Llama 3**, and **Qwen3** backbone support to the fire LLaVA-MoD pipeline. It does not cover Qwen2.5 (primary track).

**Status:** **Gemma 2 complete** (2026-07-01). **Llama 3 complete** (2026-07-05) — teacher + 3B + 1B edge students. **Qwen3 complete** (2026-07-05).

**Related:** [`BACKBONE_FAMILIES_AND_BASELINES.md`](BACKBONE_FAMILIES_AND_BASELINES.md) · [`CURRENT_STATUS.md`](CURRENT_STATUS.md)

---

## 0. Critical: conversation template ↔ tokenizer pairing

Each backbone has its **own** chat format. Reusing another family’s template (e.g. Llama 2 `[INST]` on Llama 3, or Vicuna/Phi `USER:` on Qwen `im_start`) causes **`WARNING: tokenization mismatch`** and **`loss: 0.0`** because every label is masked to `IGNORE_INDEX`.

| `--version` | Conv template | Preprocess | Native format |
|-------------|---------------|------------|----------------|
| `gemma_2` | `conv_gemma_2` | `preprocess_gemma_2` | `<start_of_turn>user` / `<end_of_turn>` |
| `llama_3` | `conv_llama_3` | `preprocess_llama_3` | `<|start_header_id|>user` / `<|eot_id|>` |
| `qwen3` | `conv_qwen3` | `preprocess_qwen` | `<|im_start|>user` / `` |
| `qwen` (Qwen2.5 track) | `conv_phi` | `preprocess_phi` | Legacy Vicuna-style — **existing Qwen2.5 checkpoints only** |

### Pretrain sanity checks (run after any template change)

```bash
cd MoD/LLava_MoD && source shells/conda_env.sh
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 1) No tokenization mismatch spam in log
rg -c 'tokenization mismatch' logs/pretrain_*_latest.log   # expect 0

# 2) Loss must be non-zero (typically ~4–7 for caption pretrain)
rg "'loss':" logs/pretrain_*_latest.log | head -5

# 3) Offline label check (500 samples)
python3 - <<'PY'
import json, copy
from dataclasses import dataclass
from transformers import AutoTokenizer
from llavamod import conversation as cl
from llavamod.data.data_utils import preprocess, preprocess_multimodal

@dataclass
class DA:
    is_multimodal=True; mm_use_im_start_end=False; num_frames=8

VERSION, MODEL = "llama_3", "meta-llama/Llama-3.1-8B-Instruct"  # or qwen3 / Qwen/Qwen3-8B-Instruct
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
cl.default_conversation = cl.conv_templates[VERSION]
data = json.load(open("llavamod/data/PreTrain/blip_laion_cc_sbu_558k.json"))
zero = sum(1 for item in data[:500]
    if (preprocess(preprocess_multimodal(copy.deepcopy([item["conversations"]]), DA()),
                   tok, has_image=True)["labels"][0] != -100).sum().item() == 0)
print(f"{VERSION}: zero-label samples in 500 = {zero} (must be 0)")
PY
```

### Llama 3 fix (2026-07-01)

**Symptom:** `llama_3` initially aliased to `conv_llama_2` → `preprocess_llama_2` → thousands of `tokenization mismatch: N vs. N+1` → `loss: 0.0`.

**Fix:** `conv_llama_3` + `preprocess_llama_3` in `conversation.py` / `data_utils.py`. Teacher pretrain restarted; loss ~4.6–5.0, zero mismatch warnings.

**Also fixed:** `pad_token=None` on Llama 3 → set `tokenizer.pad_token = tokenizer.eos_token` in `train.py` / `align_train.py` / `dpo_train.py`.

### Qwen3 fix (2026-07-01, preventive)

**Risk:** Scaffold mapped `"qwen3": conv_phi` and `preprocess_phi` (Vicuna `USER:` / `ASSISTANT:`). Qwen3 expects `<|im_start|>role` / `` — same bug class as Llama 3.

**Fix:** `conv_qwen3` + `preprocess_qwen` (`SeparatorStyle.QWEN`). Verified offline: **500/500** pretrain samples get non-zero labels (tested with Qwen2.5 tokenizer as format proxy; run again with `Qwen/Qwen3-*` after HF install).

**Qwen2.5 unchanged:** `"qwen"` still uses `conv_phi` for backward compatibility with finished Qwen2.5 checkpoints.

**Qwen3 thinking mode:** Use **Instruct** (non-thinking) checkpoints and disable thinking in eval so fire answers stay `Yes.` / `No.` deployable.


## 1. What was built

Three parallel **Track A** pipelines (same phases as Qwen2.5):

| Family | Teacher (HF) | Student (HF) | `--version` | Wrapper class | New model code? |
|--------|--------------|--------------|-------------|---------------|-----------------|
| **Gemma 2** | `google/gemma-2-9b-it` | `google/gemma-2-2b-it` | `gemma_2` | `LlavaGemma2ForCausalLM` | No — pre-existing |
| **Llama 3** | `meta-llama/Llama-3.1-8B-Instruct` | `meta-llama/Llama-3.2-3B-Instruct` | `llama_3` | `LlavaLlama3ForCausalLM` | Yes |
| **Qwen3** | `Qwen/Qwen3-8B` | `Qwen/Qwen3-4B` · `Qwen/Qwen3-0.6B` | `qwen3` | `LlavaQwen3ForCausalLM` | Yes |

All use:
- Vision: `openai/clip-vit-large-patch14-336`
- Distillation: `LOSS_TYPE=kd_lm`, Plan 4 v4 JSON (`dense2dense_fire_nonbbox_v4.json`)
- Separate `mm_projector.bin` per teacher and student (do **not** reuse Qwen2.5 artifacts)

**Attention:** Gemma2 shells use `attn_implementation=eager` (required by existing `train.py` / `align_train.py`). Llama3 and Qwen3 shells use `sdpa`.

---

## 2. Design decisions

### Gemma 2 — shells only; model code already existed

`LlavaGemma2ForCausalLM` (`llava_gemma2.py`) was already in the repo and wired in `train.py` / `align_train.py` / `builder.py`. This scaffold adds **fire training shells** only.

Gemma2-9B and Gemma2-2B share tokenizer vocab (256,128). Training must use `--attn_implementation eager` — the train entry points hard-code `eager` for `"gemma-2"` paths.

### Llama 3 — new wrapper, transformers-native Llama

The existing `LlavaLlamaForCausalLM` uses **vendored LLaMA-1/2** code under `llavamod/model/language_model/llama/`, which does not support Llama 3.1/3.2 (rope scaling, vocab 128,256, etc.).

`llava_llama3.py` follows the same pattern as `llava_mistral.py`: it subclasses **transformers** `LlamaConfig`, `LlamaModel`, and `LlamaForCausalLM` and mixes in `LlavaMetaModel` / `LlavaMetaForCausalLM` for the CLIP projector.

### Qwen3 — new wrapper, not a Qwen2.5 fork

Qwen3 is **not** a drop-in replacement for Qwen2.5:
- Different `model_type` (`qwen3` vs `qwen2`)
- Different vocab (151,669 vs 151,643)
- Architecture deltas (QK-Norm, etc.)

`llava_qwen3.py` is a **separate** file (mirrors `llava_qwen2.py` structure) and registers `model_type = "llava_qwen3"`. It requires **transformers ≥ 4.51**; the base env (`requirements.txt` pins `transformers==4.37.0`) cannot load Qwen3.

### Path detection order

In train entry points, **Qwen3 is checked before Qwen2** and **Llama3 before legacy Llama**, because paths like `Qwen/Qwen3-8B` contain the substring `qwen`.

Helpers live in `llavamod/model/backbone_utils.py`:
- `is_qwen3_path(path)` — `qwen3` or `qwen-3` in path
- `is_llama3_path(path)` — `llama-3` or `llama3` in path

### Conversation templates

Added to `llavamod/conversation.py` `conv_templates` (each must match the backbone tokenizer):

| Key | Template | Preprocess |
|-----|----------|------------|
| `"gemma_2"` | `conv_gemma_2` | `preprocess_gemma_2` |
| `"llama_3"` | `conv_llama_3` | `preprocess_llama_3` |
| `"qwen3"` | `conv_qwen3` | `preprocess_qwen` |
| `"qwen"` | `conv_phi` | `preprocess_phi` (Qwen2.5 legacy only) |

See **§0** for why cross-family templates break training.


## 3. New files created

### 3.1 Model & dependencies (`LLava_MoD/`)

| File | Purpose |
|------|---------|
| `llavamod/model/language_model/llava_llama3.py` | `LlavaLlama3Config`, `LlavaLlama3Model`, `LlavaLlama3ForCausalLM`; registers `llava_llama3` with `AutoConfig` / `AutoModelForCausalLM` |
| `llavamod/model/language_model/llava_qwen3.py` | Same pattern for Qwen3; raises clear `ImportError` if transformers &lt; 4.51 |
| `llavamod/model/backbone_utils.py` | `is_qwen3_path`, `is_qwen2_path`, `is_llama3_path` |
| `requirements-backbones-llama3-qwen3.txt` | Optional upgrade: `transformers>=4.51`, `tokenizers>=0.21`, `accelerate>=0.27` |

### 3.2 Shared training utility

| File | Purpose |
|------|---------|
| `shells/train/verify_tokenizer_gate.py` | Python script: compares teacher vs student `tokenizer.encode()` on fire prompts (`Yes.`, `No.`, etc.) for KD safety |

### 3.3 Gemma 2 shells (`shells/train/gemma2/`)

| File | Purpose |
|------|---------|
| `common_gemma2.env.sh` | Shared HF IDs, checkpoint paths, `CONV_VERSION=gemma_2`, `ATTN_IMPLEMENTATION=eager` |
| `pretrain/pretrain_teacher.sh` | Phase 1a — frozen `Gemma2-9B` + CLIP projector → `llavagemma2-9b-pretrain/` |
| `pretrain/pretrain_student.sh` | Phase 1b — frozen `Gemma2-2B` + CLIP projector → `llavagemma2-2b-student-pretrain/` |
| `finetune_lora.sh` | Phase 2 — teacher LoRA SFT → `llavagemma2-9b-finetune-lora/` |
| `dense2dense_distillation.sh` | Phase 3 — mimic KD (`align_train.py`, `kd_lm` default) |
| `plan4_d2d_multiquery_v4.sh` | Wrapper: validates prerequisites, runs D2D with Plan 4 v4 JSON |
| `plan4_d2d_multiquery_v4_val.sh` | Eval wrapper → `detectioumfire/eval/launchers/gemma2/` |
| `verify_tokenizer.sh` | Runs `verify_tokenizer_gate.py` for 9B ↔ 2B |
| `README.md` | Short quick-start |

**Checkpoint naming (Gemma 2):**

| Artifact | Path |
|----------|------|
| Teacher projector | `checkpoints/llavagemma2-9b-pretrain/mm_projector.bin` |
| Student projector | `checkpoints/llavagemma2-2b-student-pretrain/mm_projector.bin` |
| Teacher LoRA | `checkpoints/llavagemma2-9b-finetune-lora/` |
| Plan 4 student | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-gemma2-2b-seed42/` |

### 3.4 Llama 3 shells (`shells/train/llama3/`)

| File | Purpose |
|------|---------|
| `common_llama3.env.sh` | Shared HF IDs, checkpoint paths, `CONV_VERSION=llama_3`, eval dir |
| `pretrain/pretrain_teacher.sh` | Phase 1a — frozen `Llama-3.1-8B` + CLIP projector → `llavallama3.1-8b-pretrain/` |
| `pretrain/pretrain_student.sh` | Phase 1b — frozen `Llama-3.2-3B` + CLIP projector → `llavallama3.2-3b-student-pretrain/` |
| `finetune_lora.sh` | Phase 2 — teacher LoRA SFT on `teacher_training_data.json` → `llavallama3.1-8b-finetune-lora/` |
| `dense2dense_distillation.sh` | Phase 3 — mimic KD (`align_train.py`, `kd_lm` default) |
| `plan4_d2d_multiquery_v4.sh` | Wrapper: validates prerequisites, runs D2D with Plan 4 v4 JSON |
| `plan4_d2d_multiquery_v4_val.sh` | Eval wrapper → `detectioumfire/eval/launchers/llama3/` |
| `verify_tokenizer.sh` | Runs `verify_tokenizer_gate.py` for 3.1 ↔ 3.2 |
| `README.md` | Short quick-start for this family |

**Checkpoint naming (Llama 3):**

| Artifact | Path |
|----------|------|
| Teacher projector | `checkpoints/llavallama3.1-8b-pretrain/mm_projector.bin` |
| Student projector | `checkpoints/llavallama3.2-3b-student-pretrain/mm_projector.bin` |
| Teacher LoRA | `checkpoints/llavallama3.1-8b-finetune-lora/` |
| Plan 4 student | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-llama3-3b-seed42/` |

### 3.5 Qwen3 shells (`shells/train/qwen3/`)

Same layout as Llama 3; paths use `common_qwen3.env.sh` and Qwen3 HF IDs.

| File | Purpose |
|------|---------|
| `common_qwen3.env.sh` | Shared paths, `CONV_VERSION=qwen3` |
| `pretrain/pretrain_teacher.sh` | → `llavaqwen3-8b-pretrain/` |
| `pretrain/pretrain_student.sh` | → `llavaqwen3-4b-student-pretrain/` |
| `finetune_lora.sh` | → `llavaqwen3-8b-finetune-lora/` |
| `dense2dense_distillation.sh` | D2D with `REF_MLLM` / `POLICY_LLM` for Qwen3 |
| `plan4_d2d_multiquery_v4.sh` | Plan 4 v4 launcher |
| `plan4_d2d_multiquery_v4_val.sh` | Eval launcher |
| `verify_tokenizer.sh` | Tokenizer gate for 8B ↔ 4B |
| `README.md` | Short quick-start |

**Checkpoint naming (Qwen3):**

| Artifact | Path |
|----------|------|
| Teacher projector | `checkpoints/llavaqwen3-8b-pretrain/mm_projector.bin` |
| Student projector | `checkpoints/llavaqwen3-4b-student-pretrain/mm_projector.bin` |
| Teacher LoRA | `checkpoints/llavaqwen3-8b-finetune-lora/` |
| Plan 4 student | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-4b-seed42/` |

### 3.6 Eval launchers (`detectioumfire/eval/launchers/`)

| Directory | Conv mode | Notes |
|-----------|-----------|-------|
| `gemma2/` | `gemma_2` | Symlink to shared `eval_detectium_val_parallel.sh` |
| `llama3/` | `llama_3` | Same shared script; defaults from directory name |
| `qwen3/` | `qwen3` | Same shared script; defaults from directory name |

Qwen2.5 reference launcher: `eval/launchers/qwen2.5/` (canonical script source).

---

## 4. Existing files modified

### 4.1 `llavamod/model/__init__.py`

Added try/except imports (so Qwen3 failure on old transformers does not break the whole package):

```python
from .language_model.llava_llama3 import LlavaLlama3ForCausalLM, LlavaLlama3Config  # try
from .language_model.llava_qwen3 import LlavaQwen3ForCausalLM, LlavaQwen3Config    # try
```

### 4.2 `llavamod/conversation.py` and `llavamod/data/data_utils.py`

`conv_templates` (in `conversation.py`):

```python
"qwen": conv_phi,       # Qwen2.5 legacy — do not use for qwen3
"qwen3": conv_qwen3,    # native im_start / im_end
"llama_3": conv_llama_3,
"gemma_2": conv_gemma_2,
```

New `SeparatorStyle` values: `LLAMA_3`, `QWEN`.

Preprocess routing (in `data_utils.py` `preprocess()`):

- `LLAMA_3` → `preprocess_llama_3`
- `QWEN` → `preprocess_qwen`
- `GEMMA_2` → `preprocess_gemma_2`
- `phi` version only → `preprocess_phi` (removed `version.startswith("qwen")` so `qwen3` cannot fall through to phi)

### 4.3 `llavamod/train/train.py`

- Import: `from llavamod.model.backbone_utils import is_llama3_path, is_qwen3_path`
- **Dense model load:** `elif is_qwen3_path(...)` → `LlavaQwen3ForCausalLM`; `elif is_llama3_path(...)` → `LlavaLlama3ForCausalLM` (before generic `qwen` / `llama` branches)
- **Tokenizer:** dedicated `is_qwen3_path` branch → `AutoTokenizer`
- **Conversation:** `elif model_args.version in ("qwen3", "llama_3")` sets template + pad token

### 4.4 `llavamod/train/align_train.py`

Same changes as `train.py` for model loading, tokenizer, and conversation version handling (used by D2D / Plan 4).

### 4.5 `llavamod/train/dpo_train.py`

Same model-loading and tokenizer branches as above (DPO path parity).

### 4.6 `llavamod/model/builder.py`

Inference / LoRA merge support for eval and deployment:

- Imports: `LlavaLlama3ForCausalLM`, `LlavaQwen3ForCausalLM` (try/except), `backbone_utils`
- **LoRA + model_base:** Qwen3 / Llama3 branches before generic Qwen / Llama
- **mm_projector-only load:** Qwen3 / Llama3 when `model_type` or path matches
- **Full checkpoint load:** `is_qwen3_path` / `is_llama3_path` in direct-load path

### 4.7 `Important Notes/BACKBONE_FAMILIES_AND_BASELINES.md`

Updated summary table and per-family sections:
- Status: **Scaffold ready** for Llama 3 and Qwen3
- Shell paths, model file names, `requirements-backbones-llama3-qwen3.txt`
- Checklist: model wrapper + shells marked done; tokenizer gate and first training runs still open

---

## 5. What was **not** created or changed

| Item | Notes |
|------|--------|
| MoE variants | No `llava_llama3_moe.py` or `llava_qwen3_moe.py` — fire track uses dense only |
| Vendored `qwen3/` under `language_model/` | Relies on upgraded transformers instead |
| Changes to `llava_llama.py` (legacy) | Llama 3 uses separate `llava_llama3.py` |
| Changes to `llava_qwen2.py` | Qwen3 is fully separate |
| `eval/launchers/{gemma2,llama3,qwen3}/` parallel eval | Done (symlink + backbone auto-defaults) |
| Base `requirements.txt` | Still pins `transformers==4.37.0`; Llama3/Qwen3 upgrade is optional separate file |
| Gemma2 model code changes | Wrapper pre-existed; only shells added |

---

## 6. Prerequisites before first run

**Workspace root for all commands:**

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
source shells/conda_env.sh
export PYTHONPATH="${PWD}:${PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
```

### Gemma 2 (complete — 2026-07-01)

Pipeline finished: teacher pretrain + LoRA + student pretrain + Plan 4 v4 `kd_lm`. Val metrics in [`D2D_RESULTS.md`](D2D_RESULTS.md).

> **Historical:** Gemma2 `only_kd` ablation was run during development and removed (no deployable Yes/No prefixes).

| Artifact | Path |
|----------|------|
| Teacher LoRA | `checkpoints/llavagemma2-9b-finetune-lora/` |
| Student kd_lm | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-gemma2-2b-seed42/` |

**Student kd_lm val:** 99.31% Yes/No, 752/754 fire→Yes, FN=2, FP=8.

### Llama 3 (complete — 2026-07-05)

**Val results (Plan 4 v4 kd_lm):**

| Student | Yes/No | Fire→Yes | FN | FP |
|---------|-------:|---------:|---:|---:|
| 3B | 99.72% | 752/754 | 2 | 2 |
| **1B edge** | **99.79%** | **752/754** | **2** | **1** |

See [`shells/train/llama3/README.md`](../LLava_MoD/shells/train/llama3/README.md) and [`student 1B/README.md`](../LLava_MoD/shells/train/llama3/student%201B/README.md).

### Qwen3 (4B done — 0.6B edge next)

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
pip install -r requirements-backbones-llama3-qwen3.txt
hf auth login
bash shells/train/qwen3/verify_tokenizer.sh                    # 8B ↔ 4B
bash "shells/train/qwen3/student 0.6B/verify_tokenizer.sh"    # 8B ↔ 0.6B
```

**0.6B edge pipeline:**

```bash
bash "shells/train/qwen3/student 0.6B/pretrain_student_0.6b.sh"
tail -f logs/pretrain_qwen3_student_0.6b_latest.log
bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4.sh"
tail -f logs/plan4_d2d_multiquery_v4_kd_lm_qwen3_0.6b_latest.log
NUM_GPUS=8 bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4_val.sh"
```

**Verified on server (2026-06-30):**
- `LlavaLlama3ForCausalLM` imports on `transformers==4.46.3` (via `requirements-backbones-llama3.txt`)
- `transformers>=4.51` breaks torch 2.1.2 — use `requirements-backbones-llama3.txt` for Llama only
- `LlavaQwen3ForCausalLM` **fails** until full `requirements-backbones-llama3-qwen3.txt` + compatible torch

---

## 7. End-to-end pipeline commands

### Gemma 2 (active — training started 2026-06-30)

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
source shells/conda_env.sh
export PYTHONPATH="${PWD}:${PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES

# 0) Tokenizer gate
bash shells/train/gemma2/verify_tokenizer.sh

# 1a) Teacher projector pretrain → llavagemma2-9b-pretrain/mm_projector.bin
LOG_DIR=logs; STAMP=$(date +%Y%m%d_%H%M%S)
nohup env NUM_GPUS=8 bash shells/train/gemma2/pretrain/pretrain_teacher.sh \
  >> "${LOG_DIR}/pretrain_gemma2_teacher_${STAMP}.log" 2>&1 &
tail -f "${LOG_DIR}/pretrain_gemma2_teacher_${STAMP}.log"

# 1b) Student projector pretrain → llavagemma2-2b-student-pretrain/mm_projector.bin
nohup env NUM_GPUS=8 bash shells/train/gemma2/pretrain/pretrain_student.sh \
  >> "${LOG_DIR}/pretrain_gemma2_student_${STAMP}.log" 2>&1 &

# 2) Teacher LoRA SFT → llavagemma2-9b-finetune-lora/
bash shells/train/gemma2/finetune_lora.sh

# 3) Plan 4 v4 D2D (detached by default in plan script)
bash shells/train/gemma2/plan4_d2d_multiquery_v4.sh
tail -f logs/plan4_d2d_multiquery_v4_kd_lm_gemma2_2b_latest.log

# 4) Eval
NUM_GPUS=8 bash shells/train/gemma2/plan4_d2d_multiquery_v4_val.sh
```

**Checkpoints:**

| Phase | Path |
|-------|------|
| Teacher projector | `checkpoints/llavagemma2-9b-pretrain/mm_projector.bin` |
| Student projector | `checkpoints/llavagemma2-2b-student-pretrain/mm_projector.bin` |
| Teacher LoRA | `checkpoints/llavagemma2-9b-finetune-lora/` |
| Plan 4 student | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-gemma2-2b-seed42/` |

### Llama 3 (complete — 2026-07-05)

Full pipeline done for teacher + 3B + 1B edge students. See [`shells/train/llama3/README.md`](../LLava_MoD/shells/train/llama3/README.md).

**Checkpoints:**

| Phase | Path |
|-------|------|
| Teacher projector | `checkpoints/llavallama3.1-8b-pretrain/mm_projector.bin` |
| Student projector (3B) | `checkpoints/llavallama3.2-3b-student-pretrain/mm_projector.bin` |
| Student projector (1B) | `checkpoints/llavallama3.2-1b-student-pretrain/mm_projector.bin` |
| Teacher LoRA | `checkpoints/llavallama3.1-8b-finetune-lora/` |
| Plan 4 student (3B) | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-llama3-3b-seed42/` |
| Plan 4 student (1B) | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-llama3-1b-seed42/` |

### Qwen3 (4B done — 0.6B edge next)

**4B (done):**

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
source shells/conda_env.sh
export PYTHONPATH="${PWD}:${PYTHONPATH}"
pip install -r requirements-backbones-llama3-qwen3.txt
bash shells/train/qwen3/verify_tokenizer.sh
bash shells/train/qwen3/pretrain/pretrain_student.sh
bash shells/train/qwen3/plan4_d2d_multiquery_v4.sh
NUM_GPUS=8 bash shells/train/qwen3/plan4_d2d_multiquery_v4_val.sh
```

**0.6B edge (next):**

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

**Checkpoints:**

| Phase | Path |
|-------|------|
| Teacher projector | `checkpoints/llavaqwen3-8b-pretrain/mm_projector.bin` |
| Student projector (4B) | `checkpoints/llavaqwen3-4b-student-pretrain/mm_projector.bin` |
| Student projector (0.6B) | `checkpoints/llavaqwen3-0.6b-student-pretrain/mm_projector.bin` |
| Teacher LoRA | `checkpoints/llavaqwen3-8b-finetune-lora/` |
| Plan 4 student (4B) | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-4b-seed42/` |
| Plan 4 student (0.6B) | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-0.6b-seed42/` |

All Llama3/Qwen3 scripts use `--version llama_3` or `--version qwen3` and `attn_implementation sdpa`. Gemma2 scripts use `--version gemma_2` and `attn_implementation eager`.

---

## 8. File tree (new files only)

```
LLava_MoD/
├── requirements-backbones-llama3-qwen3.txt
├── llavamod/
│   └── model/
│       ├── backbone_utils.py
│       └── language_model/
│           ├── llava_llama3.py
│           └── llava_qwen3.py
└── shells/train/
    ├── verify_tokenizer_gate.py
    ├── gemma2/
    │   ├── README.md
    │   ├── common_gemma2.env.sh
    │   ├── verify_tokenizer.sh
    │   ├── finetune_lora.sh
    │   ├── dense2dense_distillation.sh
    │   ├── plan4_d2d_multiquery_v4.sh
    │   ├── plan4_d2d_multiquery_v4_val.sh
    │   └── pretrain/
    │       ├── pretrain_teacher.sh
    │       └── pretrain_student.sh
    ├── llama3/
    │   ├── README.md
    │   ├── common_llama3.env.sh
    │   ├── verify_tokenizer.sh
    │   ├── finetune_lora.sh
    │   ├── dense2dense_distillation.sh
    │   ├── plan4_d2d_multiquery_v4.sh
    │   ├── plan4_d2d_multiquery_v4_val.sh
    │   └── pretrain/
    │       ├── pretrain_teacher.sh
    │       └── pretrain_student.sh
    └── qwen3/
        ├── README.md
        ├── common_qwen3.env.sh
        ├── verify_tokenizer.sh
        ├── finetune_lora.sh
        ├── dense2dense_distillation.sh
        ├── plan4_d2d_multiquery_v4.sh
        ├── plan4_d2d_multiquery_v4_val.sh
        └── pretrain/
            ├── pretrain_teacher.sh
            └── pretrain_student.sh

detectioumfire/eval/
├── launchers/
│   ├── qwen2.5/          # canonical eval_detectium_val_parallel.sh
│   ├── gemma2/           # symlink → qwen2.5 (gemma_2 defaults)
│   ├── llama3/           # symlink → qwen2.5 (llama_3 defaults)
│   └── qwen3/            # symlink → qwen2.5 (qwen3 defaults)
├── runs/                 # teacher/, students/, baselines/
└── comparisons/          # table4_all_plans.json, etc.
```

**Gemma2 new files (9 shells + 2 README):** no Python model changes.

**Llama3/Qwen3 new files:** model wrappers + train wiring (see §3.1, §4).

**Modified:** `llavamod/model/__init__.py`, `llavamod/conversation.py`, `llavamod/data/data_utils.py`, `llavamod/train/train.py`, `llavamod/train/align_train.py`, `llavamod/train/dpo_train.py`, `llavamod/model/builder.py`, `Important Notes/BACKBONE_FAMILIES_AND_BASELINES.md`

---

## 9. Remaining work

1. **Track B native VLM baselines** — zero-shot + optional fire fine-tune (see [`BACKBONE_FAMILIES_AND_BASELINES.md`](BACKBONE_FAMILIES_AND_BASELINES.md) §Track B)
2. Regenerate batch Table 4 after any new eval: `detectioumfire/eval/launchers/qwen2.5/run_table4_student_validation.sh`
3. **Archive tokenizer gate** results per teacher↔student pair after each family starts
4. **Qwen3 thinking mode** — use Instruct weights; disable thinking in eval for `Yes.` / `No.` prefixes

---

## 10. Quick reference — env vars (`common_*.env.sh`)

| Variable | Gemma 2 | Llama 3 | Qwen3 |
|----------|---------|---------|-------|
| `TEACHER_LLM` | `google/gemma-2-9b-it` | `meta-llama/Llama-3.1-8B-Instruct` | `Qwen/Qwen3-8B-Instruct` |
| `STUDENT_LLM` | `google/gemma-2-2b-it` | `meta-llama/Llama-3.2-3B-Instruct` | `Qwen/Qwen3-4B` (edge: `Qwen/Qwen3-0.6B`) |
| `CONV_VERSION` | `gemma_2` | `llama_3` | `qwen3` |
| `ATTN_IMPLEMENTATION` | `eager` | `sdpa` (in shells) | `sdpa` (in shells) |
| `EVAL_DIR` | `eval/launchers/gemma2` | `eval/launchers/llama3` | `eval/launchers/qwen3` |

---

## 11. Audit log

### 2026-07-01 — conversation / preprocess (Llama 3 + Qwen3)

| Issue | Severity | Fix |
|-------|----------|-----|
| `llama_3` used `conv_llama_2` / `preprocess_llama_2` | **Critical** | `conv_llama_3` + `preprocess_llama_3`; loss was `0.0`, all samples masked |
| Llama 3 `pad_token=None` crashed collator | **Critical** | `tokenizer.pad_token = tokenizer.eos_token` in train entry points |
| `qwen3` used `conv_phi` / `preprocess_phi` (Vicuna format) | **Critical** | `conv_qwen3` + `preprocess_qwen`; verified 500/500 pretrain labels offline |
| `version.startswith("qwen")` would route `qwen3` → phi | **High** | Route by `SeparatorStyle.QWEN` first; phi branch is `version.startswith("phi")` only |

### 2026-06-30 — scaffold shell / builder issues

| Issue | Severity | Fix |
|-------|----------|-----|
| `REPO_ROOT` in `common_*.env.sh` used `../..` (resolved to `shells/` not `LLava_MoD/`) | **Critical** | Changed to `../../..` in `gemma2`, `llama3`, `qwen3` common env files |
| `builder.py` LoRA merge: Gemma2 teacher/student checkpoints fell through to legacy `LlavaLlamaForCausalLM` | **Critical** | Added `is_gemma2_path` / `llava_gemma2` branch in LoRA load path |
| `builder.py` mm_projector load: Qwen3/Llama3 branches used `cfg_pretrained` / `tokenizer` before assignment | **Critical** | Rewrote branches to load tokenizer + config first; added Gemma2 mm_projector branch |
| `dpo_train.py` missing Gemma2 model + `gemma_2` conversation handling | Medium | Added `LlavaGemma2ForCausalLM` branch (`eager` attn), tokenizer, and `gemma_2` version block |
| `backbone_utils.py` | Low | Added `is_gemma2_path()` helper; used in `builder.py` |
| Llama3/Qwen3 `conda_env.sh` wrong path in pretrain/finetune/D2D | **Critical** | Fixed 2026-06-30 — use `${REPO_ROOT}/shells/conda_env.sh` after `common_*.env.sh` |
| Gemma2 same `conda_env.sh` path bug | **Critical** | Fixed 2026-06-30 (same pattern as Llama3) |

### Verified correct

- **Shell syntax** — all `*.sh` pass `bash -n`
- **Path detection** — `Qwen3` before `Qwen2`; `Llama3` before legacy Llama
- **Gemma2** — `gemma_2` → `preprocess_gemma_2` (no mismatch in pretrain logs)
- **Qwen2.5** — `qwen` → `conv_phi` unchanged (existing checkpoints); new families must **not** reuse this
- **D2D shells** — `REF_MLLM_BASE` / `POLICY_LLM` set correctly; `LOSS_TYPE=kd_lm` default

### Technical debt (optional later)

- Qwen2.5 `conv_phi` does not match native Qwen `im_start` format; finished Qwen2.5 pipeline still works with legacy prompts — migrate only if retraining Qwen2.5 from scratch
- Qwen3 needs `requirements-backbones-llama3-qwen3.txt` (transformers ≥ 4.51) before first run
