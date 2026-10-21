# LLaVA-MoD Code Understanding Notes

This file captures our ongoing discussion while learning the codebase.  
We will keep updating it as we understand more parts of the project.

---

## 1) Project learning context

- Main goal: understand the LLaVA-MoD code deeply for paper-level work.
- Current focus area: data pipeline and how shell/args choices affect runtime behavior.

---

## 2) Architectures in this codebase

There are two levels:

- Base LM architectures often seen in `config.json` under `architectures` (e.g. `Qwen2ForCausalLM`).
- LLaVA/LLaVA-MoD wrapper classes used by training/inference code (e.g. `LlavaQwen2ForCausalLM`, MoE variants).

Important observation from checkpoint config:

- `model_type` can be multimodal (`llava_qwen2`) while `architectures` still shows base LM (`Qwen2ForCausalLM`).

---

## 3) Shells + arguments + docs (high-level)

Reviewed:

- `README.md`
- `docs/TRAIN_EVAL.md`
- `docs/INFERENCE.md`
- training shells in `shells/train/qwen/`
- eval shells in `shells/eval/`
- `llavamod/config/args.py`

Key takeaway:

- Shell scripts mostly pass flags and paths.
- Core behavior is implemented in Python (`train.py`, `align_train.py`, `dpo_train.py`, dataset/data utils, model code).

Important args that affect data pipeline:

- `--data_path`
- `--image_folder` / `--video_folder`
- `--image_aspect_ratio`
- `--lazy_preprocess`
- multimodal setup from model-side init (`image_tower`, `video_tower`, processors attached into `data_args`).

---

## 4) What `dataset.py` does

File: `LLaVA-MoD/llavamod/data/dataset.py`

Main responsibilities:

- Load training samples from JSON files.
- Build token tensors (`input_ids`, `labels`) and visual tensors.
- Collate variable-length data into padded batches.

Two dataset families:

- `LazySupervisedDataset` for supervised/mimic training.
- `LazyDPODataset` for preference training (`chosen` vs `rejected`).

Two collators:

- `DataCollatorForSupervisedDataset`
- `DataCollatorForDPODataset`

---

## 5) Meaning of `input_ids`

- `input_ids` are token IDs produced by the tokenizer from conversation text.
- In multimodal mode, text includes image/video placeholders that map to special token handling.
- Model later uses these IDs to build embeddings and run transformer forward.

Related tensor:

- `labels` are usually based on `input_ids` but masked with `IGNORE_INDEX` where loss should not be computed.

---

## 6) `preprocess_multimodal` vs `preprocess`

File: `LLaVA-MoD/llavamod/data/data_utils.py`

### `preprocess_multimodal(...)`

- Rewrites conversation text for multimodal placeholders before tokenization.
- Handles `<image>` / `<video>` normalization.
- Converts video placeholder into repeated image placeholder pattern (based on `num_frames`).
- Applies optional token wrapping (`mm_use_im_start_end`).

### `preprocess(...)`

- Builds prompt text according to conversation template style.
- Tokenizes into `input_ids`.
- Creates `labels` and masks non-target regions (`IGNORE_INDEX`), typically user/prompt parts.
- Returns model-ready text tensors.

Short summary:

- `preprocess_multimodal`: text-level multimodal normalization.
- `preprocess`: tokenize + label masking for training loss.

---

## 7) Meaning of selected lines in `data_utils.py` (mmtag block)

Snippet discussed:

```python
if "mmtag" in conversation_lib.default_conversation.version:
    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                  '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
```

Meaning:

- If active conversation template is an mmtag-style format, plain `<image>` is wrapped as `<Image><image></Image>`.
- This is template-compatibility formatting.

And:

- `DEFAULT_IMAGE_TOKEN` is defined in `llavamod/constants.py` as `"<image>"`.

---

## 8) What `sources` and `conversation_lib` mean

### `sources`

- Raw conversation samples passed into preprocessing.
- Typical structure: list of conversation lists; each turn has `from` and `value`.

### `conversation_lib`

- Imported module (`llavamod.conversation`) containing chat templates and formatting rules.
- `preprocess` uses `conversation_lib.default_conversation` to decide prompt style, separators, and masking behavior.

---

## 9) Next updates we plan

As we continue, we will append:

- concrete sample walkthroughs (raw JSON -> `input_ids`/`labels`/`images`)
- `llava_arch.py` tensor-flow notes
- loss path notes (`align_trainer.py`, `dpo_trainer.py`)
- mapping between dataset schema and DetectiumFire converters

