# Red Parts: Full Answers and Paper-Ready Replacements

This note explains every red-marked gap in `Explain.md` / `fire_distillation.pdf`, with the concrete facts from the training code and data. The last section recommends the best wording to put in the paper.

Source of truth in the repo:
- `LLaVA-Instruct-150K/build_teacher_training_data.py`
- `LLaVA_MoD/MoD/detectioumfire/build_detectium_bbox_synthetic2k_nonfire3k.py`
- `LLaVA-Instruct-150K/build_d2d_plan_datasets.py` (`FIRE_PROMPTS_V4` / `NON_FIRE_PROMPTS_V4`)
- `LLaVA_MoD/MoD/LLava_MoD/shells/train/qwen/pretrain/pretrain_teacher.sh`
- `LLaVA_MoD/MoD/LLava_MoD/shells/train/qwen/finetune_lora.sh`
- `LLaVA_MoD/MoD/LLava_MoD/shells/train/qwen/dense2dense_distillation.sh`
- `LLaVA_MoD/MoD/LLava_MoD/llavamod/conversation.py`
- `LLaVA_MoD/MoD/LLava_MoD/shells/train/qwen3/vision_encoders/common_ve.env.sh`

---

## 1. MobileCLIP visual token count (`???`)

### What the draft says
MobileCLIP-S2 takes a 256×256 image and produces `???` visual tokens.

### Fact
**64 visual tokens.**

| Encoder | Input size | Visual tokens |
|---------|------------|---------------|
| CLIP ViT-L/336 | 336×336 | 576 |
| MobileCLIP-S2 (FastViT) | 256×256 | **64** |

This is why MobileCLIP reduces TTFT so strongly (far fewer visual tokens in the LLM prefill).

---

## 2. Table 5 subsets: Short description / Bbox / Simple VQA

### What the red note asks
These names do not appear in the original LLaVA Mix665K release. How were they built, and what Q/A do we use?

### Fact
They are **our filtered subsets** sampled from `llava_v1_5_mix665k.json`, not official Mix665K category names. Filters are in `build_teacher_training_data.py`.

Common filters for all three:
- Image under `coco/train2017/` or `vg/VG_100K/` (bbox subset: **COCO only**)
- Single-turn (`len(conversations) == 2`)
- No CJK characters in the conversation text

| Paper label (current) | Rows | Selection rule | Example human prompt | Example answer |
|-----------------------|-----:|----------------|----------------------|----------------|
| Short description | 12,000 | Human text matches describe/detail-style (`describe`, `detail`, `what do you see`, `explain the visual`, …); not bbox | `Write a detailed description of the given image.` | Free-form scene caption |
| Bbox | 2,409 | Human text asks for bounding-box coordinates; COCO only (pool is small; target was 7,500) | `Please provide the bounding box coordinate of the region this sentence describes: top keyboard.` | `[0.16, 0.3, 0.85, 0.5]` |
| Simple VQA | 3,000 | Single-turn COCO/VG that is neither description nor bbox, and not “why / step-by-step” | Short factual / MCQ-style VQA from Mix665K | Short answer (option letter or short phrase) |

**Important for the paper:** do not imply Mix665K ships named “Short description / Bbox / Simple VQA” splits. Say they are filtered samples from Mix665K.

---

## 3. Synthetic fire + non-fire bbox block: instruction and caption

### What the red note asks
For synthetic data and non-fire images, what text instruction and caption are used? Same as DetectiumFire VQA?

### Fact
**No — different from DetectiumFire VQA.** This block is localization prior + hard negatives for teacher fine-tuning.

**Shared human prompt (synthetic fire and real non-fire):**
```text
Please provide all bounding box coordinates for regions with visible fire or flame.
```

**Synthetic fire (2,000)** — YOLO labels converted to normalized xyxy (Mix665K style):
```text
[0.21, 0.17, 0.75, 0.67]
```
(Multiple boxes = one `[...]` per line.)

**Real non-fire (3,000)** — explicit negative:
```text
No fire or flame is visible in this image. There are no bounding boxes to report.
```

---

## 4. Concrete questions used for training / evaluation

### What the red note asks
“Seven question types (four for fire, three for non-fire)” is unclear — list them.

### Fact — student distillation prompts (also used at evaluation)

**Fire scenes (4):**
1. Carefully analyze the image for signs of fire or smoke.
2. What is burning, where is it happening, and how severe is the fire?
3. Describe the fire scene and estimate risk level.
4. Provide a concise fire-focused description of this image.

**Non-fire scenes (3):**
5. Carefully analyze the image for signs of fire or smoke.
6. Check this scene for signs of fire, flame, or smoke.
7. Is there any evidence of active fire in this image? Explain briefly.

### Val-only generalization prompt
In addition, the DetectiumFire **validation** set includes:

8. **Assess whether this is a fire incident or a non-fire scene.**

Facts about this prompt:
- Appears on **190 val rows**, all **non-fire**
- Present in DetectiumFire / teacher train (1,583 rows)
- **Absent** from student distillation JSON `dense2dense_fire_nonbbox_v4.json` (0 rows)
- Used to probe generalization to an unseen question wording for distilled students

Quality check (kd_lm, seed-averaged): most students ≥98% on this prompt; teachers 100%. Main weak case is Qwen2.5-3B CLIP (~91.6% Assess accuracy vs ~99.7% on other non-fire prompts), which tends to over-trigger `Yes.` on lights/candles under this wording.

Do **not** put answer templates (`Yes.` / `No.` strings) in this section of the paper unless you intentionally discuss supervision format elsewhere.

---

## 5. Table 6 — Teacher training hyperparameters

From `pretrain_teacher.sh` (initialization) and `finetune_lora.sh` (fine-tuning), 8× NVIDIA A100 80GB, DeepSpeed ZeRO-2:

| Configuration | Initialization | Fine-Tuning |
|---------------|----------------|-------------|
| LLM | ✗ (frozen) | ✓ (LoRA) |
| VL Adaptor | ✓ | ✓ |
| ViT | ✗ | ✗ |
| LLM sequence length | 2048 | 1536 |
| Optimizer | AdamW (DeepSpeed) | AdamW (DeepSpeed) |
| Optimizer hyperparameters | β₁=0.9, β₂=0.999 (DS default) | same |
| Learning rate | 1×10⁻³ (adaptor) | 2×10⁻⁴ (LoRA); 2×10⁻⁵ (adaptor) |
| LR schedule | cosine | cosine |
| Weight decay | 0 | 0 |
| Training epochs | 1 | 1 |
| Warm-up ratio | 0.03 | 0.03 |
| Global batch size | 128 (8 GPUs × 8 × 2) | 64 (8 × 2 × 4) |
| Numerical precision | bf16 | bf16 |
| Model parallelism | ZeRO-2, 8 GPUs | ZeRO-2, 8 GPUs |
| LoRA | — | r=128, α=256, dropout=0.05 |

---

## 6. Table 7 — Student training hyperparameters

From student projector pretrain + `dense2dense_distillation.sh`:

| Configuration | Initialization | Distillation |
|---------------|----------------|--------------|
| LLM | ✗ (frozen) | ✓ (LoRA) |
| VL Adaptor | ✓ | ✓ |
| ViT | ✗ | ✗ |
| LLM sequence length | 2048 | 2048 |
| Optimizer | AdamW (DeepSpeed) | AdamW (DeepSpeed) |
| Learning rate | 1×10⁻³ (adaptor) | 2×10⁻⁵ (LoRA + adaptor) |
| LR schedule | cosine | cosine |
| Weight decay | 0 | 0 |
| Training epochs | 1 | 1 |
| Warm-up ratio | 0.03 | 0.03 |
| Global batch size | 128 | 64 (8 × 1 × 8) |
| Numerical precision | bf16 | bf16 |
| Model parallelism | ZeRO-2, 8 GPUs | ZeRO-2 / ZeRO-2-offload, 8 GPUs |
| LoRA | — | r=128, α=256, dropout=0.05 |
| Distillation objective | — | \(\mathcal{L}=\mathcal{L}_{\mathrm{KL}}+\mathcal{L}_{\mathrm{CE}}\) |
| Seeds | — | 42, 123, 456 |

---

## 7. Table 10 — Prompt / chat templates per family

Match training `CONV_MODE` and `llavamod/conversation.py`. These templates are embedded at GGUF export so inference matches training.

| Family | Prompt format |
|--------|----------------|
| **Qwen2.5** | `USER: <image>\n{prompt} ASSISTANT:` |
| **Qwen3** | `<\|im_start\|>system\n{system}<\|im_end\|>\n<\|im_start\|>user\n<image>\n{prompt}<\|im_end\|>\n<\|im_start\|>assistant\n` |
| **Llama-3.2** | `<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\n\n{system}<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>\n\n<image>\n{prompt}<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n\n` |
| **Gemma-2** | `<start_of_turn>user\n<image>\n{prompt}<end_of_turn>\n<start_of_turn>model\n` |

System string for Qwen3 / Llama-3.2:
> You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.

Gemma-2 uses an empty system prompt.

---

# Best paper-ready replacements (use these instead of the red parts)

Below is what to put in the manuscript. Prefer this over dumping all implementation minutiae.

---

### A. MobileCLIP tokens (replace `???`)

> The second is MobileCLIP-S2 (MCi2) (Vasu et al. 2024), a lightweight vision encoder based on FastViT (Vasu et al. 2023), which takes a 256×256 input image and produces **64** visual tokens. We load the `apple/MobileCLIP-S2-OpenCLIP` checkpoint through OpenCLIP (Ilharco et al. 2021).

---

### B. Table 5 row names + one clarifying sentence

**Rename rows** (recommended):

| Subset | # | Source |
|--------|--:|--------|
| Mix665K description-style | 12,000 | Filtered LLaVA-1.5 Mix665K (COCO / Visual Genome) |
| Mix665K localization | 2,409 | Filtered LLaVA-1.5 Mix665K (COCO only) |
| Mix665K simple VQA | 3,000 | Filtered LLaVA-1.5 Mix665K (COCO / Visual Genome) |
| Fire VQA | 13,077 | DetectiumFire train |
| Bbox + non-fire | 5,000 | DetectiumFire (2k synthetic fire boxes + 3k real non-fire) |

**Add after the table (short):**

> The three Mix665K blocks are not official splits of LLaVA-1.5 Mix665K. We sample single-turn English rows whose images lie under COCO train2017 or Visual Genome, and partition them by prompt type into description-style captions, bounding-box localization queries, and simple VQA (excluding description, localization, and multi-step reasoning prompts).

Do **not** invent public Mix665K category names. Do **not** expand every example Q/A in the main text; one sentence + the renamed table is enough. Details can stay in this README / appendix if reviewers ask.

---

### C. Synthetic / non-fire bbox supervision (replace the red question block)

> We additionally include 2,000 synthetic fire images with fire/flame bounding boxes and 3,000 real non-fire images as localization negatives. Both use the same instruction asking for bounding-box coordinates of visible fire or flame. Synthetic examples are supervised with normalized box coordinates derived from the annotations; non-fire examples are supervised with an explicit response that no fire region is present. This block is distinct from the DetectiumFire VQA annotations and is used only in teacher fine-tuning.

---

### D. Prompt section (replace the “seven question types” red note)

**Best main-text paragraph:**

> For distillation, each DetectiumFire training image is paired with multiple question types so that the student observes the same scene under several phrasings. Fire images use four questions and non-fire images use three:
>
> **Fire scenes.** (1) Carefully analyze the image for signs of fire or smoke. (2) What is burning, where is it happening, and how severe is the fire? (3) Describe the fire scene and estimate risk level. (4) Provide a concise fire-focused description of this image.
>
> **Non-fire scenes.** (5) Carefully analyze the image for signs of fire or smoke. (6) Check this scene for signs of fire, flame, or smoke. (7) Is there any evidence of active fire in this image? Explain briefly.
>
> These seven prompts are also used at evaluation. In addition, the DetectiumFire validation set includes one further question that does not appear in the student distillation data: “Assess whether this is a fire incident or a non-fire scene.” We retain this validation-only prompt to assess generalization to an unseen question wording.

**What to omit here:** answer templates (`Yes.` / `No.` strings). Mention answer format only if you discuss supervision format in another subsection.

---

### E. Table 6 / Table 7 (fill empty hyperparameter tables)

Use the filled tables in §§5–6 above. For the paper, keep one compact table each (Initialization vs Fine-Tuning / Distillation). Optional appendix: LoRA rank/alpha, ZeRO stage, exact micro-batch × accumulation.

**One-line summary suitable for the caption:**

> Table 6/7: Training hyperparameters for teacher and student stages on 8×A100 GPUs with bf16 and DeepSpeed ZeRO-2. Teacher fine-tuning and student distillation use LoRA (r=128, α=256). Student results are averaged over seeds 42, 123, and 456.

---

### F. Table 10 prompt templates (replace `[fill]`)

Use the four family rows in §7. In the paper, a compact table is better than long token dumps if space is tight; put full special-token strings in the appendix if needed.

**Short main-text version:**

> Table 10: Prompt template used for each student family at inference, matching the format used during training (Qwen2.5 `USER`/`ASSISTANT`; Qwen3 ChatML; Llama-3.2 header tokens; Gemma-2 turn markers).

---

## Priority checklist for editing the PDF / Explain.md

1. Replace MobileCLIP `???` with **64**.
2. Rename Table 5 Mix665K rows + add the one clarifying sentence.
3. Paste the synthetic/non-fire bbox paragraph (C).
4. Paste the seven prompts + val-only Assess paragraph (D); **no answer templates**.
5. Fill Tables 6 and 7 from §§5–6.
6. Fill Table 10 from §7.

That is the minimum set that removes every red gap without overloading the paper.
