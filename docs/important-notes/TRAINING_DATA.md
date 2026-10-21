# Training data

## Projector pretrain (teacher & student)
**LLaVA LCS-558K** (BLIP captions on LAION-CC-SBU) — general image–caption pairs for aligning the vision projector only.

## Teacher Phase 2 (fire SFT)
**35,486** rows total. The Detectium val split is held out.

| Block | Rows | Source |
|-------|-----:|--------|
| Short description | 12,000 | LLaVA mix665k (filtered; COCO / Visual Genome) |
| Bbox-format prior | 2,409 | LLaVA mix665k (filtered; COCO only) |
| Simple VQA | 3,000 | LLaVA mix665k (filtered; COCO / Visual Genome) |
| Detectium real VQA | 13,077 | DetectiumFire train |
| Detectium bbox + non-fire | 5,000 | DetectiumFire (2k synthetic fire boxes + 3k real non-fire “no boxes”) |

Rough mix: **~51% DetectiumFire**, **~43% general vision**, **~7% bbox prior**.

General-vision rows come from **LLaVA v1.5 mix665k**, sampled from **COCO train2017** and **Visual Genome (VG_100K)** images only (single-turn, no CJK). DetectiumFire supplies the fire / non-fire VQA and bbox blocks.

## Student distillation
**DetectiumFire train images only** — no general-vision mix.

Detectium uses **7 prompt types** (the same styles as validation: **4 fire** + **3 non-fire**). Distillation needs more supervision than one question per image, so each train image is expanded with those 7 question types.