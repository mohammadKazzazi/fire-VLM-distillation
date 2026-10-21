# Fine-Tuning LLaVA for Detectium Fire Dataset

This document summarizes the steps that have been automatically completed to set up LLaVA for your custom `detectiumfire` dataset, and provides instructions on how you can start and monitor the training process.

## What Was Done

1. **Repository Setup:**
   - Cloned the official [LLaVA repository](https://github.com/haotian-liu/LLaVA) into the workspace.

2. **Environment Configuration:**
   - Set up the Python dependencies and the `llava` conda environment.
   - Fixed compatibility issues with the `flash-attn` package. Specifically, downgraded `setuptools` and pinned `flash-attn==2.4.2` to ensure perfect compatibility with PyTorch 2.1.2 and the LLaVA v1.5 architecture.

3. **Custom Fine-Tuning Script:**
   - Created a ready-to-use bash script at `LLaVA/finetune_detectium_lora.sh`.
   - **Hardware Optimization:** Adjusted the script specifically for your **4x A100 (40GB)** GPU setup. The parameters `--per_device_train_batch_size 16` and `--gradient_accumulation_steps 2` have been set. Multiplied by 4 GPUs, this perfectly hits the officially recommended **Global Batch Size of 128**, ensuring optimal learning dynamics and speed.
   - **Data Mapping:** Linked the script directly to your custom dataset:
     - **JSON Data (train):** `MoD/detectiumfire/detectium_llavamod_real_full_train.json` (~90/10 split from the full JSON; val: `detectium_llavamod_real_full_val.json`)
     - **Images:** `MoD/detectiumfire/real_images/real_images`

## What You Should Do Next

All configurations are complete. You just need to trigger the training process from the terminal.

### 1. Start the Training (recommended: GNU `screen` + log)

Training must run in the **`llava` conda environment**. Use `screen` so you can detach and close your terminal or SSH session while training continues.

From the **workspace root** (`mohammadkazzazi`), start a named session with scrollback logged to a file:

```bash
cd /home/mohammadkazzazi/mohammadkazzazi
screen -L -Logfile training_log.txt -S training_session
```

Inside the new `screen` shell, enable conda and start training:

```bash
eval "$(conda shell.bash hook)"
conda activate llava
cd /home/mohammadkazzazi/mohammadkazzazi/LLaVA
./finetune_detectium_lora.sh
```

Detach from `screen` (leave training running): **Ctrl+A**, then **D**.

- Reattach later: `screen -r training_session`
- Log file (stdout/stderr from the session): `training_log.txt` in the workspace root (same directory where you ran `screen`).

### 1b. Start the Training (foreground, no `screen`)

```bash
eval "$(conda shell.bash hook)"
conda activate llava
cd /home/mohammadkazzazi/mohammadkazzazi/LLaVA
./finetune_detectium_lora.sh
```

### 2. Monitor the Output
- DeepSpeed will initialize across all 4 GPUs.
- The model will automatically download the base LLaVA weights (`liuhaotian/llava-v1.5-7b`) and the CLIP vision tower if they aren't already cached.
- The training logs will output directly to the terminal, showing the learning rate, loss, and estimated time remaining.

### 3. Retrieve Your Fine-Tuned Model
Once the training completes, the LoRA adapter weights (which represent what the model learned about fire detection) will be saved in:
```text
LLaVA/checkpoints/llava-v1.5-7b-detectium-lora
```

### 4. Running Inference with Your New Model
You can chat with your fine-tuned model by pointing LLaVA's CLI to your LoRA checkpoint alongside the base model:

```bash
conda activate llava
cd LLaVA

python -m llava.serve.cli \
    --model-path ./checkpoints/llava-v1.5-7b-detectium-lora \
    --model-base liuhaotian/llava-v1.5-7b \
    --image-file /home/mohammadkazzazi/mohammadkazzazi/MoD/detectiumfire/real_images/real_images/real_fire/<test_image_name>.jpg
```