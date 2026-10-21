# Fire students — llama.cpp / GGUF (H4 / Raspberry Pi path)

Converted on **H4 ARM64** (`c7g.2xlarge`) from LLaVA-MoD fire LoRA students
(Qwen2.5-0.5B, Llama-3.2-1B, Gemma-2-2B) + CLIP-L/14-336.

## Files

| File | Role |
|------|------|
| `fire-qwen2.5-0.5b-Q8_0.gguf` | Qwen LM Q8 (**accuracy default**) |
| `fire-llama3.2-1b-Q8_0.gguf` | Llama3 LM Q8 |
| `fire-gemma2-2b-Q8_0.gguf` | Gemma2 LM Q8 |
| `mmproj-fire-clip-vit-l-f16.gguf` | Qwen CLIP + projector F16 |
| `mmproj-fire-llama3-clip-vit-l-f16.gguf` | Llama3 CLIP + projector F16 |
| `mmproj-fire-gemma2-clip-vit-l-f16.gguf` | Gemma2 CLIP + projector F16 |
| `llava_qwen_phi.jinja` / `llava_llama3.jinja` / `llava_gemma2.jinja` | Native chat templates |
| `fire_prompt.txt` | Deployment prompt |

**Each student must use its own mmproj** (projector weights differ). Do not share one mmproj across LMs.

## Recommended combos

| Goal | LLM | mmproj |
|------|-----|--------|
| Accuracy (default) | Q8_0 | F16 |
| Pi / speed | Q4_K_M | Q8_0 (re-validate accuracy) |
| Max fidelity | F16 | F16 |

Use the **native** template for each student (phi / llama3 / gemma2). Never ChatML for Qwen.

## Run on this H4 box

```bash
cd /home/ubuntu/llavamod/llava-deploy-h1-h4-20260716/workspace/Deploy
export LD_LIBRARY_PATH="$(pwd)/bundles/gguf/bin-arm64:$LD_LIBRARY_PATH"

# Accuracy default
bash scripts/run_llamacpp_mtmd.sh --image samples/bench_images/<image>.jpg

# Speed / Pi
MODEL=bundles/gguf/fire-qwen2.5-0.5b-Q4_K_M.gguf \
MMPROJ=bundles/gguf/mmproj-fire-clip-vit-l-Q8_0.gguf \
  bash scripts/run_llamacpp_mtmd.sh --image path.jpg
```

## Raspberry Pi

1. Copy LLM + mmproj + `fire_prompt.txt` + `llava_qwen_phi.jinja` (prefer Q4 + mmproj Q8 for RAM).
2. On the Pi, **rebuild** `llama.cpp` (do not assume the H4 `bin-arm64` binary matches Pi CPU flags):

```bash
sudo apt install -y git cmake build-essential libopenblas-dev
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build -j$(nproc) --target llama-mtmd-cli
```

3. Run `scripts/run_llamacpp_mtmd.sh` with local `MODEL` / `MMPROJ`.

## Rebuild from LoRA

```bash
source ../.venv/bin/activate
bash scripts/build_llamacpp_gguf.sh
```

Produces F16 → Q8_0 + Q4_K_M LLM, mmproj F16 + Q8_0, with phi template baked into LLM GGUFs.

## Notes

- Export: `scripts/export_for_llamacpp.py` / `export_student_lean.py` (not stock Qwen2.5-VL).
- Training used `conv_mode=qwen` → `conv_phi` (`USER:` / `ASSISTANT:`). ChatML caused a large FP spike.
- Leading `\n` after the image marker matches PyTorch (`<image>\n{question}`).
- `ffn up/down are swapped` on mmproj load is expected (llama.cpp auto-swaps).
- mmproj Q8 via `scripts/quantize_mmproj_q8.py` (llama-quantize does not support clip arch) — **re-validate Detectium accuracy** before shipping.

---

## Fidelity pitfalls (GGUF vs GPU) — read before converting

These bugs caused large Detectium Yes/No gaps vs PyTorch GPU. Checklist for every rebuild:

### 1. mmproj must bake `mm_vision_select_layer=-2`

**Symptom:** Accuracy drops several points (e.g. Llama3 ~98% vs GPU ~99.8%). F16 LLM barely helps.

**Cause:** LLaVA-MoD training selects CLIP features from layer **-2**. If the temporary CLIP `config.json` used by `convert_image_encoder_to_gguf.py` omits `mm_vision_select_layer`, llama.cpp defaults to a different layer (off-by-one). The GGUF then lacks `clip.vision.feature_layer`.

**Fix:** `scripts/build_llamacpp_gguf.sh` (and student rebuild scripts) must inject into CLIP config before convert:

```json
"mm_vision_select_layer": -2,
"image_aspect_ratio": "pad"
```

**Verify after convert:**

```bash
PYTHONPATH=/path/to/llama.cpp/gguf-py python3 -c "
import gguf
r = gguf.GGUFReader('mmproj-….gguf')
assert 'clip.vision.feature_layer' in r.fields
assert 'clip.vision.image_aspect_ratio' in r.fields
print('ok', r.fields['clip.vision.feature_layer'])
"
```

Expect `feature_layer=[23]` for CLIP-L/14-336 (24 blocks → −2 → index 23).

### 2. Llama3 template must NOT emit BOS

**Symptom:** Extra FN/FP after an otherwise correct mmproj; prompts look “almost right”.

**Cause:** `llava_llama3.jinja` used to start with `<|begin_of_text|>`, but llama.cpp `mtmd_tokenize` already sets `add_special=true` and prepends BOS → **double BOS**.

**Fix:** Omit `<|begin_of_text|>` from the Jinja (see current `llava_llama3.jinja`). Let the runtime add BOS once.

### 3. Use the correct chat template per student

| Student | Template | Mistake to avoid |
|---------|----------|------------------|
| Qwen 0.5B | `llava_qwen_phi.jinja` (`USER:` / `ASSISTANT:`) | ChatML (`<\|im_start\|>`) → big FP spike |
| Llama3 1B | `llava_llama3.jinja` | Missing/extra BOS; wrong header tokens |
| Gemma2 2B | `llava_gemma2.jinja` | Wrong `<eos>` placement (after model turns only) |

### 4. Image preprocess: evaluate **both** stock and Pillow-parity (per model)

**Do not assume one resize path is best for every student.** After mmproj/template are correct, run full Detectium Yes/No **and** Table 4 under **both**:

| Setup | What it is | How |
|-------|------------|-----|
| **Stock / “fixed”** | llama.cpp default LLaVA fixed-square bicubic (non-Pillow) | Unpatched `llama-server` → metrics `*_gguf_q8_fixed` |
| **Pillow** | HF CLIP–style pad + `RESIZE_ALGO_BICUBIC_PILLOW` | Patch `tools/mtmd/clip.cpp`, rebuild → metrics `*_gguf_q8_pillow` |

**Why both:** Training/GPU use Pillow bicubic + pad-to-square (CLIP mean). Stock llama.cpp used a different bicubic → small pixel diffs that flip borderline fire/non-fire cases. Closer-to-GPU preprocess **helps some students and hurts others**.

**Measured on full val (1452) + Table 4 (754 fire) after mmproj/template fixes:**

| Model | Better preprocess | Evidence |
|-------|-------------------|----------|
| **Llama-3.2-1B** | **Stock (fixed)** | Pillow Yes/No 99.38% vs fixed 99.45%; burning-object worse on Pillow |
| **Gemma-2-2B** | **Stock (fixed)** | Pillow Yes/No 98.83% vs fixed 99.04% (more FPs); Table 4 mixed |
| **Qwen2.5-0.5B** | **Pillow** | Pillow Yes/No 99.45% vs fixed 99.38%; Table 4 all three axes higher (esp. environment) |

**Deploy rule of thumb:** pick preprocess **per student** from a full-val bake-off, not from a single 8-image smoke or from “closer to PyTorch must win.” Document the chosen path next to the GGUF preset.

**Pillow patch (local llama.cpp):** for `PROJECTOR_TYPE_MLP` without pinpoints in `tools/mtmd/clip.cpp`:

- `image_resize_algo = RESIZE_ALGO_BICUBIC_PILLOW`
- `image_resize_pad = PAD_NEAREST`
- pad color `{122, 116, 104}` (CLIP mean × 255)

Rebuild `llama-server` after changing `clip.cpp`. Re-val **all** students if you change this.

Numbers: `results/H4_hardware_metrics.md`, logs `pipeline_logs/full_val_mmproj_fix.log` / `full_val_pillow_patch.log`.

### 5. Quantization is secondary

On Q8 error sets, switching LLM to F16 recovered few cases once mmproj/template were fixed. Prefer **Q8 LLM + F16 mmproj** for H4; do not assume “F16 LLM will close the GPU gap.”

### 6. One projector / vision tower per student

Each LoRA has its own `non_lora_trainables.bin` projector. Export and convert **per preset**. Sharing Qwen’s mmproj with Llama/Gemma is wrong even if CLIP weights look identical.

### Quick rebuild checklist

1. Inject `mm_vision_select_layer=-2` + `image_aspect_ratio=pad` before mmproj convert; verify GGUF metadata.
2. Bake / pass the native Jinja; for Llama3, no explicit BOS.
3. **Bake-off preprocess per student:** full-val Yes/No + Table 4 on **stock** and **Pillow** `llama-server`; keep the winner per model (today: Llama/Gemma → stock, Qwen → Pillow).
4. Score Detectium Yes/No **and** Table 4 (`score_fire_reasoning_table4.py`).
5. Diff remaining errors vs PyTorch on the same images before blaming quantization.

See also: `results/H4_hardware_metrics.md`, `pipeline_logs/full_val_pillow_patch.log`.
