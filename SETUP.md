# DeepSight — Environment Setup & Training Walkthrough

This document walks you through setting up the environment, preparing data, and running the first training job on **Bench2Drive-mini** with a Qwen2.5-VL-3B + DINOv3 backbone.

> All commands assume you've `cd`'d into the project root (`DeepSight/`) after `git clone`.

---

## 0. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Linux + bash | — | Tested on Ubuntu 22.04 / 24.04 |
| NVIDIA driver | recent | `nvidia-smi` must work |
| CUDA toolkit | 12.4 / 12.6 / 12.8 | Pick the index URL that matches in §2 |
| Python | **3.11** | 3.10 also works if you need closed-loop CARLA later |
| [uv](https://docs.astral.sh/uv/) | latest | Faster than pip; `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| GPU memory | ≥ 24 GB per card (≥ 8 GB if QLoRA) | Full FT of Qwen2.5-VL-3B + DINOv3 needs ≥ 48 GB or DeepSpeed Zero-2 |
| Disk | ≥ 50 GB free | Models + data + checkpoints |
| Hugging Face account | — | DINOv3 is a **gated** repo — see §5 |

---

## 1. Clone and create the virtual environment

```bash
git clone https://github.com/<your-org>/DeepSight.git
cd DeepSight
uv venv --python 3.11
source .venv/bin/activate
```

> **Auto-activate** (optional): add this snippet to `~/.bashrc` so the venv activates whenever you `cd` into `DeepSight/`:
>
> ```bash
> _deepsight_auto_venv() {
>   local target="$HOME/path/to/DeepSight"
>   if [[ "$PWD" == "$target" || "$PWD" == "$target"/* ]]; then
>     [[ -z "$VIRTUAL_ENV" && -f "$target/.venv/bin/activate" ]] && source "$target/.venv/bin/activate"
>   else
>     [[ "$VIRTUAL_ENV" == "$target/.venv" ]] && deactivate
>   fi
> }
> PROMPT_COMMAND="_deepsight_auto_venv${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
> ```

---

## 2. Install PyTorch (CUDA-matched)

Pick **one** of the following based on your CUDA toolkit. We use `torch==2.7.1` (matches the reference environment in `example.txt`).

```bash
# CUDA 12.6 (recommended)
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126

# Or CUDA 12.4
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.7.1+cu126 True
```

---

## 3. Install Python dependencies

```bash
uv pip install -r requirements.txt
```

This installs everything except **flash-attn** (special build), **the bundled transformers fork** (next step), and **the project itself** (later).

> If you hit version conflicts on `vllm`, leave it commented out in `requirements.txt` (it's only needed for the `infer_with_vllm.py` script, not for training). Install it in a separate venv when you actually need it.

---

## 4. Install the bundled transformers (with DeepSight's DINOv3 modifications)

DeepSight customizes `Qwen2_5_VLForConditionalGeneration` to take BEV inputs (`pixel_values_bevs`) and run a frozen DINOv3 branch that supervises future-frame latent prediction. These changes live in `src/transformers/`, which is a complete transformers 4.56.2 source tree.

```bash
uv pip install -e src/transformers --no-deps
```

`--no-deps` is important — we already installed everything via `requirements.txt`.

Verify the fork is in use:
```bash
python -c "
import transformers, inspect
from transformers import Qwen2_5_VLForConditionalGeneration
print('version:', transformers.__version__)
print('path:', transformers.__file__)
sig = inspect.signature(Qwen2_5_VLForConditionalGeneration.forward)
print('has pixel_values_bevs?', 'pixel_values_bevs' in sig.parameters)
"
# expected:
# version: 4.56.2
# path: .../DeepSight/src/transformers/src/transformers/__init__.py
# has pixel_values_bevs? True
```

---

## 5. Get access to DINOv3 ViT-L/16 weights

DINOv3 is a [gated repo](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).

1. Visit the model page above and click **"Request access"**. Meta usually approves within hours.
2. Log in locally:
   ```bash
   huggingface-cli login
   ```
3. (Optional) Pre-download to avoid race conditions when launching multi-GPU training:
   ```bash
   python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov3-vitl16-pretrain-lvd1689m')"
   ```

The model id is wired into `configs/ad_bev_v4.yaml`:
```yaml
dinov3_pretrained: facebook/dinov3-vitl16-pretrain-lvd1689m
```

---

## 6. Install `flash-attn`

flash-attn needs PyTorch already installed, and must build **without isolation** to link against your installed torch's CUDA ABI.

```bash
uv pip install wheel packaging ninja
uv pip install flash-attn==2.8.1 --no-build-isolation
```

This takes a few minutes (compiles CUDA kernels). Don't interrupt.

---

## 7. Install the project itself (editable)

```bash
uv pip install -e . --no-deps
```

Verify:
```bash
python -c "import llamafactory; print(llamafactory.__file__)"
# should point to src/llamafactory/__init__.py in this repo
```

---

## 8. Download Bench2Drive-mini

The mini dataset is not bundled. Get it from the [Bench2Drive official release](https://github.com/Thinklab-SJTU/Bench2Drive). You want the 10 scene archives (around 2.7 GB total).

Place the `.tar.gz` files into `bench2drive/Bench2Drive-mini/`:
```
bench2drive/
└── Bench2Drive-mini/
    ├── Accident_Town03_Route156_Weather0.tar.gz
    ├── AccidentTwoWays_Town12_Route1444_Weather0.tar.gz
    ├── ...
    └── YieldToEmergencyVehicle_Town04_Route165_Weather7.tar.gz
```

Then extract them (into a sibling folder so the originals stay untouched):
```bash
mkdir -p bench2drive/Bench2Drive-mini-extracted
cd bench2drive/Bench2Drive-mini-extracted
for f in ../Bench2Drive-mini/*.tar.gz; do
  echo "Extracting $(basename $f)"
  tar -xzf "$f"
done
cd ../..
```

Sanity check:
```bash
ls bench2drive/Bench2Drive-mini-extracted/Accident_Town03_Route156_Weather0/camera/
# expected: anno  camera  expert_assessment  lidar  radar ...
```

---

## 9. Prepare training data

Two scripts run in sequence:

### 9.1 Crop BEV time-series images

```bash
python src/tools/crop_bev_for_bench2drive.py
```

For each scene this generates 5 sub-folders inside `camera/`:
`rgb_bev_0th-hz/`, `rgb_bev_5th-hz/`, `rgb_bev_10th-hz/`, `rgb_bev_15th-hz/`, `rgb_bev_20th-hz/`
— each containing 512×512 BEV crops at increasing future offsets (0 / +5 / +10 / +15 / +20 frames).

### 9.2 Generate a placeholder for missing historical frames

The first ~20 frames of each scene don't have a full 2-second history, so the data builder uses a black placeholder image. Create it once:

```bash
python -c "
import cv2, numpy as np
cv2.imwrite('bench2drive/Bench2Drive-mini-extracted/hisblack.jpg',
            np.zeros((900, 1600, 3), dtype=np.uint8))
"
```

### 9.3 Decompress per-frame annotations

`crop_bev_for_bench2drive.py` reads `.json.gz` directly. But `targetpointgen.py` (next step) expects `.json`. Unpack them:

```bash
cd bench2drive/Bench2Drive-mini-extracted
for d in */anno; do
  ( cd "$d" && gunzip -k *.json.gz )
done
cd ../..
```

### 9.4 Build the training JSONL

```bash
python bench2drive/dataprocess/targetpointgen.py
```

This writes ~2085 samples to `data/train_bev_mini.jsonl` (≈ 56 MB), pre-registered as the `bench2drive_bev_train` dataset in `data/dataset_info.json`.

The script **skips CoT** by default (each sample's answer uses `<think>None.</think>`). To enable CoT, run `bench2drive/dataprocess/jsonopenai.py` first with Qwen3-VL API credentials, merge the results, and point `RESULT_JSONL_PATH` at it inside `targetpointgen.py`.

Verify one sample:
```bash
python -c "
import json
with open('data/train_bev_mini.jsonl') as f:
    d = json.loads(f.readline())
print('keys:', list(d.keys()))
print('msgs:', len(d['messages']))
print('imgs:', len(d['images']))   # should be 15 (4 hist + 6 surround + 5 BEV)
"
```

---

## 10. Run training

```bash
bash nebula.sh
```

By default this uses **all visible GPUs** via `torchrun`. To restrict, edit `nebula.sh` and uncomment the `CUDA_VISIBLE_DEVICES` line.

**Strongly recommended first pass** — smoke-test on a tiny budget:
1. Edit `configs/ad_bev_v4.yaml`, uncomment:
   ```yaml
   max_samples: 100
   max_steps: 20
   ```
2. Run `bash nebula.sh` — should finish in a few minutes and produce loss values around `loss_rec ~ 3-5`, `loss_gen ~ 12`.
3. Re-comment those two lines for the real run.

### What you should see in the logs

```
Add special tokens <|start_bev_token|>,<|end_bev_token|>,<|bev_token_0|>,... to tokenizer's vocabulary.
Loaded pretrained DINOv3 weights from facebook/dinov3-vitl16-pretrain-lvd1689m
[INFO|trainer.py] Total optimization steps = 116        # for 2 epochs on 2085 samples / global batch 36
label_len: 4XXX, loss: XX, loss_rec: XX, loss_gen: XX   # printed per step from the modified Qwen forward
```

The custom Qwen2.5-VL forward (`src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`) prints `loss_rec` (CE for trajectory + CoT text tokens) and `loss_gen` (MSE between predicted BEV latents and DINOv3-extracted target features).

Checkpoints land in `saves/qwen2_5vl-3b/deepsight/ad_bev_v4/`. The loss curve plot is saved to `training_loss.png` in that directory.

---

## 11. Common gotchas

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'utils.obj_utils'` | You're on an older clone — pull latest; `road_collator.py` ships with stub imports |
| `ValueError: None is not in list` in ad_collator | Special tokens not registered — check `add_special_tokens` is present in `configs/ad_bev_v4.yaml` |
| `Image features and image tokens do not match: tokens X, features Y` | `cutoff_len` too small — bump from 4096 to 8192 (already done in the shipped config) |
| `AttributeError: 'Qwen2_5_VLConfig' object has no attribute 'dinov3_config'` | `dinov3_pretrained` not set — make sure that line is in `configs/ad_bev_v4.yaml` |
| `GatedRepoError` when downloading DINOv3 | Approval not granted yet — wait or check `huggingface-cli whoami` |
| OOM | (a) Increase grad_accum, decrease per_device_train_batch_size. (b) Switch deepspeed config to `examples/deepspeed/ds_z3_offload_config.json`. (c) Lower `image_max_pixels` to `131072`. |

---

## 12. Inference

### Native transformers inference (debug, slow)
```bash
python src/infer_for_debug.py --ckpt saves/qwen2_5vl-3b/deepsight/ad_bev_v4
```

### vLLM inference (faster, no DINOv3 — see README §3.2)
Requires a separate venv with `vllm` installed. Merge weights first via `src/tools/merge_model_weight.py`.

---

## 13. Closed-loop evaluation on Bench2Drive

See [README.md §4](README.md). Requires CARLA 0.9.16 + a separate `b2d` Python 3.10 venv. Not covered here.

---

## Appendix: What changed vs upstream DeepSight code

Files modified in this repo vs the original upstream:

| File | Change |
|---|---|
| `src/transformers/` | Swapped from broken partial fork to **full working transformers 4.56.2 + DeepSight DINOv3 patches** (62 MB) |
| `src/llamafactory/model/loader.py` | Inject `config.dinov3_config` before model construction; load pretrained DINOv3 weights post-init |
| `src/llamafactory/hparams/model_args.py` | New `dinov3_pretrained` config field |
| `src/llamafactory/data/road_collator.py` | Stubbed 3 missing util imports (`utils.obj_utils`, `utils.vis_utils`, `utils.cls_utils`) — RoadCollector is exported but never instantiated, ADCollector is what training uses |
| `bench2drive/dataprocess/targetpointgen.py` | (a) skip CoT when `RESULT_JSONL_PATH` is empty; (b) relative paths replace `/mnt/nas-data-1/...` hardcoded ones; (c) per-scene JSONL written inside the scene folder; (d) cleaner main block |
| `src/tools/crop_bev_for_bench2drive.py` | (a) replace `sudo gzip -d` with Python `gzip.open` (no root needed); (b) relative `base_folder`; (c) cap Pool processes at scene count |
| `data/dataset_info.json` | `bench2drive_bev_train.file_name` changed to relative `train_bev_mini.jsonl` |
| `nebula.sh` | YAML passed as positional arg (CLI doesn't accept `--config`); use `FORCE_TORCHRUN=1` |
| `configs/ad_bev_v4.yaml` | New file — full training config for our setup |
| `requirements.txt` | New file — pinned dependency list |
| `.gitignore` | New file — excludes `__pycache__`, `saves/`, `.venv`, training data outputs |
