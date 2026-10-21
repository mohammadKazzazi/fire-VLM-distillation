# Hardware, training GPUs, and CPU conversion

Paper-oriented notes for **what hardware we used** and **how GPU LoRA students were turned into CPU (GGUF) models**.  
This file is for the experimental-setup section. It is not a training tutorial.

---

## 1. Training GPUs (use this in the paper)

All LLaVA-MoD stages in this project (projector pretrain, teacher LoRA SFT, and a single Phase 2 distillation used for every student) ran on **NVIDIA A100 80 GB** GPUs.

| Item | Value |
|------|--------|
| GPU | **NVIDIA A100** (Ampere, compute capability 8.0) |
| Memory per GPU | **80 GB HBM2e** |
| GPUs used | **8** |
| Precision | **bfloat16** (`--bf16 True`) |
| Parallelism | **DeepSpeed ZeRO-2** (`llavamod/config/dpconfig/zero2.json`) |
| Attention | PyTorch SDPA (`--attn_implementation sdpa`) |
| Per-GPU micro-batch | 1 |
| Gradient accumulation | 8 |
| Effective global batch (8 GPUs) | \(1 \times 8 \times 8 = 64\) |

**Software stack on the training node**

| Package | Version used |
|---------|----------------|
| Python | 3.10.20 |
| PyTorch | 2.4.1+cu121 |
| CUDA toolkit (PyTorch wheel) | 12.1 |
| transformers | 4.57.6 |
| peft | 0.4.0 |
| DeepSpeed | 0.9.5 |
| tokenizers | 0.22.2 |
| accelerate | 1.14.0 |

Scripts take `NUM_GPUS` (default **8** on a full node). Training is launched with DeepSpeed, for example:

```text
deepspeed --num_gpus 8 --master_port <port> llavamod/train/align_train.py \
  --deepspeed ./llavamod/config/dpconfig/zero2.json \
  --bf16 True --seed {42|123|456} …
```

**What 8×A100-80GB is for**

- Teacher projectors and LoRA (7B / 8B / 9B class) plus CLIP ViT-L/14@336.
- Student projector pretrain (0.5B–4B).
- **Phase 2** (one recipe for all students): teacher LoRA and student LoRA in one process. That stage needs both models in GPU memory at once; **80 GB × 8** is the node that fits it without CPU offload.

**Follow-on note (do not mix with the 8-GPU recipe in the main text):** some later jobs used **4× A100 80 GB** with the same per-GPU batch. That **halves** the global batch unless `gradient_accumulation_steps` is doubled. The paper default is **8× A100 80 GB**, ZeRO-2, no ZeRO-Offload.

Official A100 product page: [https://www.nvidia.com/en-us/data-center/a100/](https://www.nvidia.com/en-us/data-center/a100/)

---

## 2. Converting each student from GPU LoRA → CPU (GGUF)

Training artifacts are **PEFT LoRA adapters**, not full merged weights:

```text
checkpoints/<student-phase2>-seedN/
  adapter_config.json
  adapter_model.bin  (or adapter_model.safetensors)
  non_lora_trainables.bin   ← MLP projector (mm_projector)
  config.json
```

CPU inference uses **[llama.cpp](https://github.com/ggml-org/llama.cpp)** and the **[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)** format.  
Project: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)  
(Older URL, same project: [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp).)

This is **not** ONNX for the VLM, and **not** Hugging Face `optimum`. The VLM CPU path is llama.cpp **mtmd** (multimodal): language GGUF + CLIP mmproj GGUF.

### 2.1 Toolchain (links)

| Step | Tool | Link |
|------|------|------|
| Merge LoRA into base LM | Hugging Face **PEFT** `PeftModel.merge_and_unload()` | [https://github.com/huggingface/peft](https://github.com/huggingface/peft) |
| Load / save HF weights | **transformers** `AutoModelForCausalLM` | [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers) |
| HF LM → GGUF F16 | llama.cpp `convert_hf_to_gguf.py` | [https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) |
| Quantize LM | llama.cpp `llama-quantize` | built from the same repo (`cmake --build … --target llama-quantize`) |
| CLIP + projector → mmproj | `tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py` | [https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py) |
| Bake chat template | `gguf-py` `gguf_new_metadata.py --chat-template-file` | llama.cpp `gguf-py` |
| CPU runtime | `llama-server` / `llama-mtmd-cli` | llama.cpp `tools/server`, `tools/mtmd` |
| BLAS on ARM CPU | **OpenBLAS** + OpenMP | [https://www.openblas.net/](https://www.openblas.net/) |

Local wrappers in this repo:

- `Deploy/scripts/export_student_lean.py` — merge LoRA, write `llm/` + `llava.projector` (no CLIP load; low RAM).
- `Deploy/scripts/merge_and_bundle.sh` — calls `LLava_MoD/scripts/merge_lora_weights.py` for a full HF bundle.
- `Deploy/scripts/convert_llama3_3b_qwen3_4b_gguf.sh` — HF → F16 → Q8_0 / Q4_K_M + mmproj (same recipe for other families).
- `Deploy/notes/GGUF.md` — fidelity checklist vs GPU PyTorch.

### 2.2 Pipeline (same for every family)

**Step A — Merge LoRA on CPU (fp16)**

`export_student_lean.py`:

1. Load base Instruct LM (`Llama-3.2-*`, `Qwen2.5-*`, `gemma-2-2b-it`, `Qwen3-*`) in float16, **without** the CLIP tower.
2. Attach LoRA with PEFT; `merge_and_unload()` into dense weights.
3. Save Hugging Face `llm/` (`safetensors` + tokenizer).
4. From `non_lora_trainables.bin`, extract the 2-layer MLP projector (`mlp2x_gelu`) into `llava.projector` with llama.cpp key names:
   - `model.mm_projector.0.{weight,bias}`
   - `model.mm_projector.2.{weight,bias}`
5. Record `mm_vision_select_layer = -2` and `image_aspect_ratio = pad` (must match training).

**Step B — Language GGUF**

```bash
python3 convert_hf_to_gguf.py <llm_dir> --outfile fire-<stem>-f16.gguf --outtype f16
```

Then quantize (binary from a native llama.cpp build):

```bash
./build/bin/llama-quantize fire-<stem>-f16.gguf fire-<stem>-Q8_0.gguf   Q8_0
./build/bin/llama-quantize fire-<stem>-f16.gguf fire-<stem>-Q4_K_M.gguf Q4_K_M
```

| Quant | Role |
|-------|------|
| **F16** | Conversion source; max fidelity, large |
| **Q8_0** | Default accuracy CPU build (~8-bit) |
| **Q4_K_M** | Smaller / faster (Pi-class RAM); re-check Detectium Yes/No |

**Step C — Vision mmproj GGUF (one per student)**

CLIP tower: `openai/clip-vit-large-patch14-336`.  
**Do not share mmproj across students** — each LoRA has its own projector.

Before convert, patch CLIP `config.json`:

```json
"mm_vision_select_layer": -2,
"image_aspect_ratio": "pad"
```

Then:

```bash
python3 tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py \
  -m <clip_snapshot> \
  --llava-projector llava.projector \
  --projector-type mlp \
  --output-dir <tmp>
```

Default mmproj is **F16**. llama.cpp `llama-quantize` does **not** support CLIP; optional mmproj Q8 is a separate script (`quantize_mmproj_q8.py`). Re-validate accuracy if you use it.

After convert, GGUF must contain `clip.vision.feature_layer` (CLIP-L/14-336: **23**, i.e. layer −2 of 24 blocks). Missing this caused a several-point Yes/No drop vs GPU.

**Step D — Chat template (per backbone)**

Bake a **native** Jinja template into the LM GGUF (`gguf_new_metadata.py --chat-template-file`). Wrong templates caused large false-positive spikes.

| Student | Template file | Do not use |
|---------|---------------|------------|
| Qwen2.5-0.5B | `llava_qwen_phi.jinja` (`USER:` / `ASSISTANT:`) | ChatML `<\|im_start\|>` |
| Llama 3.2 1B / 3B | `llava_llama3.jinja` **without** `<\|begin_of_text\|>` (runtime already adds BOS) | Double BOS |
| Gemma 2 2B | `llava_gemma2.jinja` | Wrong `<eos>` placement |
| Qwen3 0.6B / 4B | `llava_qwen3.jinja` | Qwen2.5 / ChatML templates |

**Step E — Build llama.cpp on the CPU host (native)**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build -j$(nproc) --target llama-quantize llama-mtmd-cli llama-server
```

Run with **8 OpenMP threads**, `ctx=4096`, `temperature=0`. Each request re-encodes the image (no cached CLIP state across images).

### 2.3 Students converted this way

| Phase 2 student (GPU LoRA) | CPU LM GGUF | CPU mmproj |
|----------------------------|-------------|------------|
| Qwen2.5-0.5B | `fire-qwen2.5-0.5b-{f16,Q8_0,Q4_K_M}.gguf` | `mmproj-fire-clip-vit-l-f16.gguf` |
| Llama-3.2-1B | `fire-llama3.2-1b-Q8_0.gguf` | `mmproj-fire-llama3-clip-vit-l-f16.gguf` |
| Gemma-2-2B | `fire-gemma2-2b-Q8_0.gguf` | `mmproj-fire-gemma2-clip-vit-l-f16.gguf` |
| Llama-3.2-3B | same script (`convert_llama3_3b_qwen3_4b_gguf.sh`) | own mmproj |
| Qwen3-4B | same script | own mmproj |

**Default CPU accuracy combo:** LM **Q8_0** + mmproj **F16**.  
F16 LM after a correct mmproj recovered almost no extra Detectium cases; quantization is secondary to mmproj layer and template.

Image resize: training/GPU uses Pillow bicubic + pad-to-square. llama.cpp stock LLaVA resize is slightly different. Full-val bake-off: Llama/Gemma preferred **stock** resize; Qwen2.5-0.5B preferred a **Pillow-parity** patch in `tools/mtmd/clip.cpp`. Pick preprocess **per student**.

---

## 3. CPU test hardware

There are two CPU stories. Do not mix them in one table row.

### 3.1 Measured VLM CPU numbers

The llama.cpp fire-student timings in `Deploy/notes/H4_hardware_metrics.md` were measured on **ARM64 CPU only** (no GPU):

| Item | Value |
|------|--------|
| CPU | AWS **Graviton3** (ARM Neoverse-V1 class) |
| vCPUs | **8** |
| RAM | **16 GiB** |
| GPU | none |
| OS | **Ubuntu 22.04 LTS ARM64** |
| Runtime | llama.cpp + OpenBLAS + OpenMP, **8 threads** |

This is a **server ARM** box used so CPU VLM latency is reproducible. It is **not** a Raspberry Pi.

### 3.2 Raspberry Pi OS (Pi-class path)

Project docs target **Raspberry Pi OS Bookworm** (Debian 12) on ARM64. Rebuild llama.cpp **on the Pi** (`-DGGML_NATIVE=ON`); do not copy binaries from the server ARM box (different CPU flags).

| Item | Intended stack |
|------|----------------|
| OS | **Raspberry Pi OS Bookworm** (64-bit), Debian 12 |
| Packages | `git`, `cmake`, `build-essential`, `libopenblas-dev` |
| VLM (if RAM allows) | llama.cpp mtmd, prefer **Q4_K_M** LM + mmproj Q8/F16 |
| Ultra-low power (no VLM) | **ONNX Runtime** MobileNetV3-Small binary fire classifier (`Deploy/pi/`) — [https://onnxruntime.ai/](https://onnxruntime.ai/) |

A full CLIP + LM is not a realistic default Pi workload. The paper split is: **VLM on GPU (or large ARM CPU)**; **ONNX classifier on Pi-class devices**.

**Do not** report QEMU / VirtualBox / Docker-emulated Pi OS latency as Raspberry Pi performance.

If you add a physical Pi row, fill these from `lscpu` / `cat /proc/cpuinfo` / `vcgencmd` on that board (model, cores, RAM) — they are not in the training logs.

Typical 64-bit Pi OS identifiers (for the methods text, not as measured results):

```text
PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"   # Raspberry Pi OS Bookworm
Architecture: aarch64
```

---

## 4. Short methods paragraph (draft)

> Training used **eight NVIDIA A100 GPUs with 80 GB of memory each**, mixed-precision **bfloat16**, and **DeepSpeed ZeRO-2**. Distillation used a per-device batch of 1 and 8 gradient-accumulation steps (global batch 64).  
> For CPU inference we merged each student’s LoRA into the base LM with **Hugging Face PEFT**, converted the LM and CLIP+MLP projector to **GGUF** with **llama.cpp** (`convert_hf_to_gguf.py`, `llama-quantize`, `convert_image_encoder_to_gguf.py`), and ran **llama.cpp** with OpenBLAS. The default CPU accuracy setting is **Q8_0** weights and an **F16** multimodal projector, with CLIP features taken from layer **−2** as in training.  
> CPU VLM latency was measured on an **8-core ARM64** host (16 GB RAM, Ubuntu 22.04). Raspberry Pi OS Bookworm is the intended on-device OS; on Pi-class hardware we either rebuild llama.cpp natively or use the **ONNX** binary classifier rather than claiming server-ARM numbers as Pi numbers.

---

## 5. Where the numbers live

| Topic | File |
|-------|------|
| GGUF conversion + fidelity bugs | `Deploy/notes/GGUF.md` |
| ARM CPU llama.cpp latency | `Deploy/notes/H4_hardware_metrics.md` |
| Pi ONNX classifier | `Deploy/pi/README.md` |
| Aligned GPU vs GGUF clocks | `Deploy/notes/TIMING_PROTOCOL.md` |
