# Preference distillation dataset (Phase 3b — after D2D)

**Phase:** LLaVA-MoD **preference distillation** (`dpo_train.py` / `preference_distillation.sh`)  
**Goal:** Reduce **false fire alarms**, **missed fires**, and **hallucinated scene details** in the deployable **dense** student by training on **chosen vs rejected text answers** on the **same** Detectium image.

**Last updated:** 2026-06-17

**Prerequisite:** A working **dense-to-dense (D2D)** student checkpoint from the five-plan ladder — see [`DISTILLATION_TRAINING_DATASET.md`](DISTILLATION_TRAINING_DATASET.md). Preference distillation is a **short polish pass**, not a substitute for mimic KD.

**You do not need new fire images.** All preference rows use **Detectium real train photos** already under `LLaVA-Instruct-150K/` (`real_fire/`, `real_non_fire/`).

---

## 1. Where this fits in the pipeline

```text
Phase 1  Student projector pretrain     (done)
Phase 2  Teacher SFT                    (done)
Phase 3a D2D mimic KD (Plans 1→5)     → best dense student
Phase 3b Preference distillation        → THIS DOC (optional, recommended)
Deploy   Best dense checkpoint
```

Optional later (not required for v1 deployment):

- Dense → sparse (MoE) — [`dense2sparse_distillation.sh`](../../LLava_MoD/shells/train/qwen/dense2sparse_distillation.sh)

**When to run preference distillation**

| Run PD? | Condition |
|---------|-----------|
| **Yes** | D2D student works but val shows **FP** (false fire on non-fire), weak explanations, or invented scene details |
| **Skip** | D2D already meets FP/FN targets; save GPU time |
| **Not yet** | D2D Plans 1–2 fail — fix mimic KD first |

---

## 2. What this is **not** (Detectium `preference.json`)

The Kaggle / Detectium bundle includes `preference_dataset/preference.json` (~11k–12k rows). That file is **for a different task**.

| | Detectium `preference.json` | This doc (fire QA preference) |
|--|------------------------------|-------------------------------|
| **Paper purpose** | Diffusion-DPO / RLHF for **Stable Diffusion** | LLaVA-MoD **text** preference (DPO/KTO) |
| **Compares** | Two **synthetic** images (`img2/` vs `img3/`) | Two **text answers** on one **real** image |
| **Fields** | `prompt`, `image1`, `image2`, `preference`, `reason` | `image`, `chosen`, `rejected` (LLaVA conversations) |
| **Use in this project** | **Do not** feed to `preference_distillation.sh` | **Build** `detectium_fire_qa_preference_train.json` |

See DetectiumFire paper §3.2 (arXiv:2511.02495): human pairwise prefs align **image generation**, not the LLaMA/VLM fire-QA model.

Upstream LLaVA-MoD uses **RLAIF-V**-style data (~80k rows). We build a **Detectium-only, fire-QA** analogue from real train images.

---

## 3. What the trainer expects

Loader: `LazyDPODataset` in [`llavamod/data/dataset.py`](../../LLava_MoD/llavamod/data/dataset.py)  
Entry point: [`dpo_train.py`](../../LLava_MoD/llavamod/train/dpo_train.py)

Each record:

```json
{
  "id": "pref_non_fire_01300_fp",
  "image": "real_non_fire/images (37).jpg",
  "chosen": [
    {
      "from": "human",
      "value": "<image>\nIs there any evidence of active fire in this image? Explain briefly."
    },
    {
      "from": "gpt",
      "value": "No. There are no visible flames or clear smoke indicating an active fire in this image."
    }
  ],
  "rejected": [
    {
      "from": "human",
      "value": "<image>\nIs there any evidence of active fire in this image? Explain briefly."
    },
    {
      "from": "gpt",
      "value": "Yes. Visible flames and smoke are present in the scene, indicating an active fire."
    }
  ]
}
```

**Rules**

- **`chosen` and `rejected` must use the same human prompt** (same question, same `<image>` token).
- Only the **`gpt` answer** differs (good vs bad / hallucinated).
- **`image`** path is **relative to `IMAGE_FOLDER`** (same layout as teacher / D2D JSON).
- Same schema as LLaVA SFT — see [`TEACHER_TRAINING_DATASET.md`](TEACHER_TRAINING_DATASET.md) §5 for path conventions.

Shell variables (when launching [`preference_distillation.sh`](../../LLava_MoD/shells/train/qwen/preference_distillation.sh)):

```bash
JSON_FILE='.../detectium_fire_qa_preference_train.json'
IMAGE_FOLDER='/home/ubuntu/llava-test/LLaVA-Instruct-150K'
POLICY_MLLM='./checkpoints/plan4-d2d-multiquery-v1/'   # best D2D student
REF_MLLM='./checkpoints/llavaqwen2.5-7b-finetune-lora'
REF_MLLM_BASE='Qwen/Qwen2.5-7B-Instruct'
OUTPUT_DIR='./checkpoints/plan4-preference-v1/'
```

---

## 4. Data sources (Detectium only)

| Asset | Path | Role |
|-------|------|------|
| **Train QA (images + labels + prompts)** | [`detectium_llavamod_real_full_train.json`](../detectioumfire/detectium_llavamod_real_full_train.json) | 13,077 rows — **primary index** |
| **Val (held out)** | [`detectium_llavamod_real_full_val.json`](../detectioumfire/detectium_llavamod_real_full_val.json) | **Never** use for building preference JSON |
| **Images** | `LLaVA-Instruct-150K/real_fire/`, `real_non_fire/` | Symlinks via [`link_detectium_images.sh`](../../../LLaVA-Instruct-150K/link_detectium_images.sh) |
| **Teacher (chosen)** | `llavaqwen2.5-7b-finetune-lora` + `Qwen/Qwen2.5-7B-Instruct` | Generate or validate good answers |
| **Student (rejected, v2)** | Best D2D checkpoint | Wrong student outputs after eval on train/hard subset |

**Train split stats:** ~6,795 fire / ~6,282 non-fire rows; **7** distinct human prompt templates in the train JSON.

---

## 5. Four preference pair types (priority)

Build a **mix** of these. For factory deployment, weight **Type A** highest (false alarms).

### Type A — Non-fire false positive (highest priority)

**Goal:** Student (and policy) learns **not** to say fire when there is none.

| Field | Content |
|-------|---------|
| **Image** | `real_non_fire/...` from train |
| **chosen** | Starts with **`No.`** — safe, generic explanation (**no invented room/object**) |
| **rejected** | Starts with **`Yes.`** — plausible but **wrong** fire claim |

Example rejected (do not copy scene details you cannot see):

```text
Yes. Visible flames and smoke are present, indicating an active fire in the scene.
```

### Type B — Fire false negative

**Goal:** Do not miss real fire.

| Field | Content |
|-------|---------|
| **Image** | `real_fire/...` from train |
| **chosen** | Teacher or filtered train GT; must start with **`Yes.`** |
| **rejected** | Starts with **`No.`** — generic non-fire denial |

### Type C — Same Yes/No, hallucinated detail

**Goal:** Reduce invented objects, places, or severity.

| Field | Content |
|-------|---------|
| **chosen** | Teacher answer (grounded, short) |
| **rejected** | **Same prefix** as GT (`Yes.` or `No.`) but adds **unsupported** detail (e.g. “kitchen”, “warehouse”, “severe” when not justified) |

### Type D — Student mistake pairs (build after D2D)

**Goal:** Most targeted correction for **your** student.

1. Run D2D student on train rows (or hard subset).  
2. Where student ≠ teacher/GT on Yes/No or clearly hallucinates:  
   - **chosen** = teacher (or GT-aligned) answer  
   - **rejected** = student answer  

Recommended for **preference v2** after first PD pass.

---

## 6. How to fill **chosen** answers

### Option 1 — Teacher generation (recommended for fire rows)

Run **Qwen2.5-7B LoRA teacher** on `(image, human_prompt)`:

```text
temperature: 0.0
max_new_tokens: 64–128
do_sample: False
```

Use the same validation scripts as [`eval/launchers/qwen2.5/`](../detectioumfire/eval/launchers/qwen2.5/) with train JSON paths.

### Option 2 — Reuse train `gpt` field

For rows where train answer already matches teacher behavior (Yes/No + short explanation), you may use the existing **`gpt`** value as **chosen** after prefix/filter checks.

### Option 3 — Fixed non-fire templates (no teacher call)

When label is non-fire (`gpt` starts with `No.`), **chosen** can be a **task-specific template** (see §8) instead of teacher generation — faster and avoids teacher over-alarm on hard negatives.

### Chosen filters (drop row if violated)

- Fire image → answer must start with **`Yes.`**
- Non-fire image → answer must start with **`No.`**
- No bounding boxes, coordinates, or `[x1, y1, x2, y2]`
- Not only `Yes.` / `No.` with zero explanation
- No bbox/localization language

---

## 7. How to fill **rejected** answers (no new images)

| Method | Type | How |
|--------|------|-----|
| **Prefix flip** | A, B | Non-fire: force **`Yes.`** + generic false fire text; Fire: force **`No.`** + generic denial |
| **Detail hallucination** | C | Keep GT prefix; append invented object/environment/severity |
| **Student output** | D | Logged wrong answer from D2D student |
| **Severity perturbation** | C (soft) | Same structure; swap moderate ↔ severe without evidence |

**Do not** invent specific scenes (“kitchen”, “office”) in **chosen** unless the image/metadata supports it. **Rejected** may include such hallucinations **intentionally** for Type C.

---

## 8. Template library (non-fire **chosen** and **rejected**)

Use the **same human prompt** as the train row (or one of the deployment prompts in §9).

### Non-fire — chosen (detection-style prompt)

```text
No. There are no visible flames or clear smoke indicating an active fire in this image.
```

Variant (risk prompt):

```text
No. This appears to be a non-fire scene. The fire risk appears low because there are no visible flames, clear smoke, or burning materials.
```

Variant (alarm prompt):

```text
No. This should not trigger a fire alarm because there is no clear visual evidence of active fire or dangerous smoke. Normal monitoring is sufficient.
```

### Non-fire — rejected (Type A false positive)

```text
Yes. Visible flames and smoke are present in the scene, indicating an active fire.
```

### Fire — rejected (Type B false negative)

```text
No. This is a non-fire scene. No visible flames or clear smoke indicating an active fire are present.
```

### Fire — rejected (Type C detail hallucination)

Keep **`Yes.`** from chosen, then add unsupported detail, e.g.:

```text
Yes. A severe fire is burning in the kitchen with heavy smoke filling the room.
```

(Only valid as **rejected** when kitchen is not known.)

---

## 9. Prompt set

### Core (from train JSON — reuse as-is)

1. `Carefully analyze the image for signs of fire or smoke.`
2. `Check this scene for signs of fire, flame, or smoke.`
3. `Is there any evidence of active fire in this image? Explain briefly.`
4. `Assess whether this is a fire incident or a non-fire scene.`
5. `What is burning, where is it happening, and how severe is the fire?`
6. `Describe the fire scene and estimate risk level.`
7. `Provide a concise fire-focused description of this image.`

**v1 recommendation:** one preference row per train image using **that row’s existing human prompt**.

### Optional deployment prompts (v2 / multi-query preference)

Add extra rows on the **same images** (not required for v1):

```text
<image>
Is there visible fire or smoke in this image? Answer Yes or No first, then explain briefly.

<image>
What is happening in the scene, and how severe is the fire risk?

<image>
Should this image trigger a fire alarm, monitoring only, or no action? Explain briefly.
```

For fire images: **chosen** = teacher-generated; non-fire = templates (§8).

---

## 10. Recommended dataset sizes

| Tier | Rows | Purpose |
|------|-----:|---------|
| **Pilot** | 1,000–2,000 | Verify `dpo_train.py` + dense shell |
| **v1** | 5,000–8,000 | Balanced Types A–C |
| **Full** | 10,000–13,000 | ~1 pair per train image |

**Suggested mix (v1)**

| Type | Share | Focus |
|------|------:|-------|
| A — non-fire FP | 40–50% | Reduce false alarms |
| B — fire FN | 30–35% | Reduce missed fire |
| C — detail hallucination | 15–25% | Grounded explanations |
| D — student mistakes | Add in v2 | After D2D eval |

---

## 11. Build procedure (step-by-step)

### Step 0 — Prerequisites

- [ ] D2D student checkpoint exists (best of Plans 2–5)  
- [ ] `IMAGE_FOLDER` images resolve (`link_detectium_images.sh` done)  
- [ ] Val JSON **not** included  

### Step 1 — Index train rows

Load [`detectium_llavamod_real_full_train.json`](../detectioumfire/detectium_llavamod_real_full_train.json). For each row:

- `id`, `image`, human prompt, `gpt` answer  
- Label: fire if `gpt` starts with `Yes.`, else non-fire  

### Step 2 — Build Type A rows (non-fire)

For each non-fire train row (or stratified sample):

1. Copy human prompt into `chosen` and `rejected`.  
2. Set **chosen** `gpt` = §8 non-fire template (matched to prompt style) or filtered train `gpt`.  
3. Set **rejected** `gpt` = §8 false-positive **`Yes.`** template.  
4. Assign new id: `pref_{original_id}_fp`.  

### Step 3 — Build Type B rows (fire)

For each fire train row:

1. **chosen** = teacher generation **or** filtered train `gpt`.  
2. **rejected** = §8 **`No.`** false-negative template.  
3. Id: `pref_{original_id}_fn`.  

### Step 4 — Build Type C rows (subset)

Sample ~15–25% of train (stratified fire/non-fire):

1. **chosen** = teacher or GT-aligned answer.  
2. **rejected** = same Yes/No prefix + hallucinated detail (§7).  
3. Id: `pref_{original_id}_hall`.  

### Step 5 — (Optional v2) Type D from student

After D2D eval on train or error log:

1. **chosen** = teacher answer.  
2. **rejected** = student wrong answer.  

### Step 6 — Validate and export

- Count rows; check Yes/No consistency with image label for **chosen**.  
- Spot-check 50 random rows manually.  
- Write:

```text
LLaVA-Instruct-150K/detectium_fire_qa_preference_train.json
```

**Builder script (TODO):** `MoD/detectioumfire/build_detectium_fire_qa_preference.py` — not in repo yet; until then, implement Steps 1–6 in a script or notebook.

### Step 7 — Preference distillation training

Adapt [`preference_distillation.sh`](../../LLava_MoD/shells/train/qwen/preference_distillation.sh) for **dense** deployable student:

```bash
POLICY_MODEL_TYPE='dense'
MOE_ENABLE=False
MOE_LOSS_ENABLE=False
POLICY_MLLM='./checkpoints/<best-d2d-checkpoint>/'
LOSS_TYPE='kto_pair'          # or sigmoid
learning_rate=2e-6              # low — polish pass only
num_train_epochs=1
```

Teacher stays dense LoRA (`REF_MLLM` + `REF_MLLM_BASE`).

### Step 8 — Evaluation

Same as D2D — [`eval/launchers/qwen2.5/`](../detectioumfire/eval/launchers/qwen2.5/) with **student** paths:

| Metric | Why |
|--------|-----|
| Yes/No accuracy | Overall correctness |
| **FP** on `real_non_fire` | Primary PD target |
| **FN** on `real_fire` | Must not regress |
| Unseen prompts | Generalization beyond 7 train templates |
| Table 4 reasoning (fire subset) | Explanation quality |

Compare **D2D-only** vs **D2D + preference** on the same val.

---

## 12. Training defaults vs D2D

| Setting | D2D (mimic) | Preference (this phase) |
|---------|-------------|-------------------------|
| Script | `align_train.py` | `dpo_train.py` |
| Loss | `only_kd` | `kto_pair` or `sigmoid` |
| LR | `2e-5` (LoRA) | **`2e-6`** (lower) |
| Epochs | 1 | 1 (do not over-train) |
| Student | Base + projector + new LoRA | **Continues from D2D checkpoint** |
| Reference | Teacher (KD) | Teacher (DPO reference) |
| Data | Single conversation | **chosen + rejected** pairs |

---

## 13. Checklist before launching preference distillation

- [ ] `detectium_fire_qa_preference_train.json` exists  
- [ ] Every `image` resolves under `IMAGE_FOLDER`  
- [ ] Val rows **not** in preference JSON  
- [ ] `POLICY_MLLM` points to **best D2D** checkpoint (not base 3B alone)  
- [ ] `REF_MLLM` + `REF_MLLM_BASE` = Qwen2.5-7B LoRA + base  
- [ ] Shell configured for **`dense`** student (`MOE_ENABLE=False`)  
- [ ] `OUTPUT_DIR` unique (e.g. `./checkpoints/plan4-preference-v1/`)  
- [ ] Baseline D2D val metrics saved for before/after comparison  

---

## 14. Expected outcomes and limits

**What preference distillation can improve**

- False **Yes** on non-fire images (Type A)  
- Over-confident or invented explanations (Type C)  
- Some student-specific errors (Type D)  

**What it cannot fix**

- Broken D2D / tokenizer mismatch / bad eval setup  
- Lack of visual diversity (same 13k images as D2D)  
- Bbox/localization (out of scope — do not train bbox prefs)  
- Teacher errors amplified into **chosen** labels  

**Honest limits (for papers / reports)**

- No new images — multi-query prefs reuse the same photos  
- Template non-fire answers are task-aware but not scene-specific  
- PD is a small alignment step; gains are often **modest** vs good D2D  

---

## 15. Version plan

| Version | Contents | When |
|---------|----------|------|
| **PD-v1** | Types A + B + subset C; templates + teacher; 5k–8k rows | After D2D Plan 2+ passes gates |
| **PD-v2** | Add Type D (student mistakes) + optional deployment prompts | After first PD eval |
| **PD-v3** | Full ~13k coverage + multi-query prefs | If v2 shows clear FP/FN gains |

---

## 16. Related docs

| Doc | Purpose |
|-----|---------|
| [`DISTILLATION_TRAINING_DATASET.md`](DISTILLATION_TRAINING_DATASET.md) | D2D Plans 1→5 (run first) |
| [`TEACHER_TRAINING_DATASET.md`](TEACHER_TRAINING_DATASET.md) | JSON schema, image paths |
| [`CURRENT_STATUS.md`](CURRENT_STATUS.md) | Resume / checkpoint paths |
| [`FIRE_MLLM_PROJECT_README.md`](FIRE_MLLM_PROJECT_README.md) | Project strategy |
| [TRAIN_EVAL.md](../../LLava_MoD/docs/TRAIN_EVAL.md) | Upstream preference stage (RLAIF-V) |
| DetectiumFire [arXiv:2511.02495](https://arxiv.org/abs/2511.02495) | **Synthetic** preference data (diffusion only) |

---

*Update this file when `build_detectium_fire_qa_preference.py` is added or PD checkpoint paths are fixed.*
