# Distilling Vision-Language Models for On-Device Fire Understanding

Official implementation of [Distilling Vision-Language Models for On-Device Fire Understanding](https://arxiv.org/abs/2609.05782).

**Mohammad Kazzazi**<sup>1\*</sup>, **Zixuan Liu**<sup>2\*†</sup>, **Siavash Khajavi**<sup>2,3†</sup>

<sup>1</sup>EPFL, Lausanne, Switzerland  
<sup>2</sup>Detectium, Espoo, Finland  
<sup>3</sup>Aalto University, Espoo, Finland

<sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding authors.  
[seyedmohammad.ojaghkazazi@epfl.ch](mailto:seyedmohammad.ojaghkazazi@epfl.ch), [zixuan@detectium.io](mailto:zixuan@detectium.io), [siavash.khajavi@aalto.fi](mailto:siavash.khajavi@aalto.fi)

Large vision-language models can tell a hazardous fire from a controlled flame, but they are too large for an embedded fire sensor. This repository fine-tunes a fire-specialized teacher, then distills it into a small student from the same model family. Students keep most of the teacher’s fire-understanding behavior and can be deployed without sending images off the device.

The training code is built on [LLaVA-MoD](https://github.com/shufangxun/LLaVA-MoD) (Shu et al., ICLR 2025). We use dense students and a fire-domain objective, rather than the original sparse mixture-of-experts student.

## Method

A general-purpose VLM is not a reliable fire teacher, so each family is specialized before distillation.

**Teacher.** The language model and vision encoder stay frozen. Only the vision–language projector is trained, on general image–caption pairs (LLaVA LCS-558K), with next-token prediction. The teacher is then LoRA-finetuned on DetectiumFire together with general visual instruction data, so it learns fire context without losing the ability to describe ordinary scenes.

**Student.** The student projector is initialized the same way. Distillation then matches the teacher’s next-token distribution on fire-understanding examples with a KL term, and adds a cross-entropy term on the ground-truth answer:

\[
\mathcal{L} = \mathcal{L}_{\mathrm{KL}}(\pi_S; \pi_T) + \mathcal{L}_{\mathrm{CE}}(\pi_S)
\]

KL alone pushes several students toward predicting fire on almost every image. The cross-entropy term is what keeps non-fire scenes from becoming false alarms. Students see only the DetectiumFire training split.

**Models.** Each student is distilled from a teacher in the same family.

| Teacher | Students |
|---|---|
| Qwen2.5-7B-Instruct | Qwen2.5-0.5B, 1.5B, 3B |
| Qwen3-8B | Qwen3-0.6B, 4B |
| Llama-3.1-8B-Instruct | Llama-3.2-1B, 3B |
| Gemma-2-9B-IT | Gemma-2-2B |

The default vision encoder is CLIP ViT-L/14 at 336px. MobileCLIP-S2 is the lighter alternative used for the on-device setting. Student numbers are averaged over seeds 42, 123, and 456.

**Evaluation.** Models are scored on the DetectiumFire validation set (1,452 images: 754 fire, 698 non-fire). Fire and non-fire detection uses the full set. Burning object, indoor/outdoor environment, and fire severity are scored on fire images only.

## Repository

```text
LLava_MoD/         training and distillation
detectioumfire/    DetectiumFire conversations and validation
deploy/            on-device export and inference
evaluation/        native VLM baseline scripts
```

Training entry points live under `LLava_MoD/shells/train/`, one directory per family (`qwen`, `qwen3`, `llama3`, `gemma2`). MobileCLIP runs are under `vision_encoders/` inside those directories. Validation launchers are in `detectioumfire/eval/launchers/`.

## Setup

```bash
cd LLava_MoD
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Llama-3 and Qwen3 backbones also need `requirements-backbones-llama3-qwen3.txt`. MobileCLIP and SigLIP2 need `requirements-vision-encoders.txt`.

Base language models are loaded from Hugging Face (`Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-2-9b-it`, and the matching student checkpoints). Gemma and Llama weights require accepting the model license on Hugging Face.

## Dataset

### DetectiumFire

Fire and non-fire images come from [DetectiumFire on Kaggle](https://www.kaggle.com/datasets/yimengfuyao/detectiumfire) (Liu, Khajavi, and Jiang). Create a Kaggle API token, save it as `~/.kaggle/kaggle.json`, then:

```bash
pip install "kagglehub[pandas-datasets]"
python detectioumfire/Download_Dataset.py
```

`kagglehub` prints the local download path. The archive contains real fire images, real non-fire images, and synthetic fire images.

Conversation files used by the scripts are already in `detectioumfire/`:

| File | Role |
|---|---|
| `detectium_llavamod_real_full_train.json` | Fire VQA training split (13,077) |
| `detectium_llavamod_real_full_val.json` | Held-out validation (1,452). Do not train on this file. |
| `val_questions.jsonl` | Validation prompts |
| `detectium_bbox_synthetic2k_nonfire3k.json` | Teacher localization data: 2,000 synthetic fire boxes and 3,000 real non-fire negatives |
| `fire_prompts.json` | Question wordings |

Image fields in those JSON files are relative (`real_fire/...`, `real_non_fire/...`). Set `--image_folder` to the directory that contains those folders after the Kaggle download.

### Projector pretraining

Both teacher and student projectors are aligned on [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) (LCS-558K, `blip_laion_cc_sbu_558k`). Place the annotation JSON and images under:

```text
LLava_MoD/llavamod/data/PreTrain/
```

The pretrain scripts expect `blip_laion_cc_sbu_558k.json` at that location. Follow the [LLaVA data instructions](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) for the image archive.

### Teacher instruction mix

Teacher LoRA fine-tuning mixes DetectiumFire training conversations with [LLaVA-1.5 Mix665K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) (`llava_v1_5_mix665k.json`). Mix665K images (COCO, GQA, OCR-VQA, TextVQA, Visual Genome) are the standard LLaVA-1.5 image bundles, documented in the same LLaVA data guide. Students are distilled on DetectiumFire training data only.

Override `JSON_FILE` and `IMAGE_FOLDER` in the shell scripts if your data root differs from the defaults.

## Training

Run scripts from `LLava_MoD/`. The Qwen2.5 + CLIP track is the reference pipeline. The other families use the same three stages.

```bash
cd LLava_MoD
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 1. Teacher projector, then LoRA fine-tune
bash shells/train/qwen/pretrain/pretrain_teacher.sh
bash shells/train/qwen/finetune_lora.sh

# 2. Student projector
bash shells/train/qwen/pretrain/pretrain_student.sh          # 3B
bash shells/train/qwen/student\ 0.5B/pretrain_student_0.5b.sh
bash shells/train/qwen/student\ 1.5B/pretrain_student_1.5b.sh

# 3. Distillation (KL + CE). SEED defaults to 42.
bash shells/train/qwen/dense2dense_distillation.sh
bash shells/train/qwen/student\ 0.5B/dense2dense_distillation_0.5b.sh
bash shells/train/qwen/student\ 1.5B/dense2dense_distillation_1.5b.sh
```

Llama-3, Gemma-2, and Qwen3 use `shells/train/llama3/`, `shells/train/gemma2/`, and `shells/train/qwen3/`. Set `NUM_GPUS` for the machine you are on. The default student loss is `kd_lm` (KL + CE). `LOSS_TYPE=only_kd` drops the cross-entropy term and is the ablation that produces the fire-prediction bias.

## Evaluation

From the launcher for the family you trained:

```bash
cd detectioumfire/eval/launchers/qwen2.5
MODEL_PATH=/path/to/student_or_teacher \
MODEL_BASE=Qwen/Qwen2.5-0.5B-Instruct \
NUM_GPUS=8 \
./eval_detectium_val_parallel.sh
```

`MODEL_BASE` is the Hugging Face language model when `MODEL_PATH` is a LoRA checkpoint. Matching launchers exist for `gemma2`, `llama3`, and `qwen3`. Answers are written to `detectioumfire/eval/runs/`. Score fire reasoning with:

```bash
python detectioumfire/eval/lib/score_fire_reasoning_table4.py \
  --val-json detectioumfire/detectium_llavamod_real_full_val.json \
  --answers-lora detectioumfire/eval/runs/.../val_answers.jsonl
```

## On-device inference

CLIP students are not run as PyTorch or ONNX on the sensor. Each LoRA adapter is merged, converted to a Q8_0 GGUF language model, and paired with its own CLIP mmproj (F16). MobileCLIP-S2 has no llama.cpp image encoder, so that student stays PyTorch on CPU.

```bash
# Merge one student LoRA and write llm/ + llava.projector
python deploy/scripts/export_student_generic.py

# Q8_0 language GGUF + CLIP mmproj, with the family chat template baked in
bash deploy/scripts/convert_one_kd_lm_gguf.sh /path/to/registry.json <student-slug>
```

Chat templates are in `deploy/bundles/gguf/` (`llava_qwen_phi.jinja`, `llava_llama3.jinja`, `llava_gemma2.jinja`, `llava_qwen3.jinja`). A wrong template changes the false-positive rate, so each family keeps its own file.

On the device, llama.cpp `llama-mtmd-cli` loads the language GGUF and the mmproj. `deploy/pi/run_one_image.sh` is the one-image runner used for the Qwen2.5-0.5B student: 4 threads, temperature 0, context 4096. Build llama.cpp on the target CPU with `deploy/pi/build_llamacpp.sh`. Validation of the GGUF models is `deploy/fire_detect/eval_gguf_detectium_val.py`.

## Citation

```bibtex
@article{kazzazi2026firevlm,
  title         = {Distilling Vision-Language Models for On-Device Fire Understanding},
  author        = {Kazzazi, Mohammad and Liu, Zixuan and Khajavi, Siavash},
  journal       = {arXiv preprint arXiv:2609.05782},
  year          = {2026},
  eprint        = {2609.05782},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.05782}
}
```

DetectiumFire:

```bibtex
@article{liu2026detectiumfire,
  title   = {DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding},
  author  = {Liu, Zixuan and Khajavi, Siavash and Jiang, Fan},
  journal = {arXiv preprint arXiv:2511.02495},
  year    = {2026}
}
```

## Acknowledgements

Training and distillation follow [LLaVA-MoD](https://github.com/shufangxun/LLaVA-MoD) and [LLaVA](https://github.com/haotian-liu/LLaVA). Fire images and labels are from [DetectiumFire](https://www.kaggle.com/datasets/yimengfuyao/detectiumfire).
