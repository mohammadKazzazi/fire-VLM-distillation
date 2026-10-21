# Run the Pi ONNX fire classifier on a Mac

**What this is:** Raspberry Pi–oriented fire detector using **ONNX Runtime** (no PyTorch needed for inference).  
**What this is not:** The LLaVA-MoD VLM (that stays on GPU servers).

**Package on the server:**
```text
/home/ubuntu/llava-test/Deploy/transfer/pi-onnx-mac-20260716.tar.gz
```
(~6 MB)

---

## 1. Download this one file from the server

```bash
# From your Mac (replace host/key as needed):
scp ubuntu@YOUR_SERVER:/home/ubuntu/llava-test/Deploy/transfer/pi-onnx-mac-20260716.tar.gz ~/Downloads/
```

Or download via Cursor / S3 / whatever you use — **only that tarball**.

### What’s inside

| Path | Purpose |
|------|---------|
| `pi/fire_classifier/` | Inference + CLI + bench (no Torch) |
| `pi/models/fire_mobilenet_v3_small.onnx` | Trained binary fire/non-fire model (~5.9 MB) |
| `pi/models/fire_mobilenet_v3_small.meta.json` | Labels + preprocess |
| `pi/requirements-pi.txt` | `onnxruntime`, `numpy`, `Pillow` |
| `pi/scripts/run_pi_infer.sh` | One-image detect |
| `pi/scripts/run_pi_bench.sh` | Latency / RAM bench |
| `pi/README.md` | Full Pi docs |
| `samples/demo.jpg` | Sample non-fire image |

**Do not download** for Mac Pi testing:
- Full `LLaVA_MoD/` / checkpoints / conda
- `Deploy/fire_detect/` VLM stack
- H1–H4 GPU transfer package

---

## 2. Unpack on Mac

```bash
cd ~/Downloads
tar -xzf pi-onnx-mac-20260716.tar.gz
cd pi
```

---

## 3. Create venv and install (Apple Silicon or Intel)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-pi.txt
```

Verify:

```bash
python - <<'PY'
import onnxruntime as ort, platform
print("arch:", platform.machine())
print("onnxruntime:", ort.__version__)
print("providers:", ort.get_available_providers())
PY
```

---

## 4. Run inference

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# bundled sample (non-fire)
python fire_classifier/daemon.py --image ../samples/demo.jpg

# or any local photo
python fire_classifier/daemon.py --image /path/to/your.jpg

# same via script
bash scripts/run_pi_infer.sh --image ../samples/demo.jpg
```

Expected JSON fields: `fire_detected`, `confidence`, `label`, `latency_ms`.

---

## 5. Benchmark on Mac (optional)

```bash
bash scripts/run_pi_bench.sh --image ../samples/demo.jpg --runs 50
# writes results/pi_bench_*.json
```

Note: Mac timing ≠ Raspberry Pi timing. Useful for code validation only.

---

## 6. Prompt for a new Cursor chat on your Mac

Paste this:

```text
I unpacked pi-onnx-mac-20260716.tar.gz on my Mac. Follow pi/README.md and
Deploy/docs/MAC_PI_ONNX_TRANSFER.md (if present) / the instructions below.

Goal:
1) Create a venv and install pi/requirements-pi.txt (onnxruntime only — no PyTorch).
2) Run fire_classifier/daemon.py on samples/demo.jpg and one local image.
3) Optionally run the ONNX bench script.
4) Report fire_detected, confidence, latency_ms, and confirm no torch import is required.

Do not set up the LLaVA VLM. Do not train. Inference-only ONNX path.
```

---

## Optional: copy files without the tarball

If scp of a folder is easier, download only:

```text
Deploy/pi/                  (entire folder, including models/*.onnx)
Deploy/samples/demo.jpg     (optional sample)
```

Skip `pi/scripts/train_export_onnx.py` dependencies unless you want to retrain (that needs Torch on the server, not the Mac).
