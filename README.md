# DLoRA-Enhanced SparkVSR: Keyframe-Conditioned Video Super-Resolution

## Overview

This project integrates DLoRA (W4+SC) as a keyframe enhancer into SparkVSR's sparse-keyframe-conditioned VSR pipeline. DLoRA generates high-quality reference keyframes; SparkVSR propagates them to the full video. Three reference modes are supported: `no_ref` (blind), `pisasr` (PiSA-SR enhancement), and `dlora` (DLoRA W4+SC enhancement, our contribution).

## Directory Structure

```
submission/
├── README.md
├── sparkvsr_improved/        # SparkVSR with our modifications
│   ├── sparkvsr_inference_script.py   (*) modified: dlora ref_mode + SideChannel
│   ├── dloral_keyframe.py             (*) CLI wrapper for DLoRA single-frame SR
│   ├── side_channel/                  (*) SideChannel post-processing module
│   ├── run_eval_all.sh                evaluation launcher
│   ├── finetune/scripts/eval_all_metrics.py   evaluation core
│   └── requirements.txt               SparkVSR Python dependencies
└── dlora_improved/           # DLoRA W4+SC (code only, no weights)
    ├── src/                   core model + inference wrapper + CFR-RAFT
    └── ram/                   RAM semantic tagging model
```

(*) = files we created or modified.

## Dependencies

Two separate conda environments are required because SparkVSR and DLoRA need incompatible PyTorch versions.

### Environment 1: SparkVSR (main inference, torch 2.5.0)

```bash
conda create -n sparkvsr python=3.10
conda activate sparkvsr
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
cd sparkvsr_improved && pip install -r requirements.txt
```

### Environment 2: DLoRA (keyframe enhancer, torch 2.0.1)

```bash
conda create -n lora python=3.10
conda activate lora
pip install torch==2.0.1 torchvision==0.15.2
pip install diffusers==0.25.0 transformers==4.28.1 accelerate xformers==0.0.20
pip install peft==0.9.0 open-clip-torch==2.20.0 einops Pillow PyYAML numpy
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
pip install mmengine==0.10.7 scipy
pip install 'setuptools<80'   # pkg_resources was removed in 80+
```

Adjust the mmcv wheel URL for your CUDA version. The example above is for CUDA 11.7.

### Additional packages for evaluation

```bash
conda activate sparkvsr
pip install pyiqa   # CLIPIQA, MUSIQ, LPIPS, DISTS
```

## Weights

All weights must be downloaded separately. Placeholder links below -- replace with actual download URLs.

### SparkVSR Model (~40 GB)

```
# Download from HuggingFace:
#   https://huggingface.co/JiongzeYu/SparkVSR
# Place at: sparkvsr_improved/checkpoints/SparkVSR/
```

### DLoRA W4+SC Weights

| File | Size | Google Drive Link |
|---|---|---|
| model_52001.pkl (RAFT) | 3.3 GB | [PLACEHOLDER] |
| sidechannel_step005000.pt | 78 MB | [PLACEHOLDER] |
| stable-diffusion-2-1-base/ | 14 GB | [PLACEHOLDER] |
| ram_swin_large_14m.pth | 5.3 GB | [PLACEHOLDER] |



### Evaluation Metrics Weights

| File | Size | Google Drive Link |
|---|---|---|
| DOVER.pth | 229 MB | [PLACEHOLDER] |
| FAST_VQA_3D_1_1.pth | 122 MB | [PLACEHOLDER] |
| swin_tiny_patch244_window877_kinetics400_1k.pth | 122 MB | [PLACEHOLDER] |

## Usage

All commands assume weights are downloaded and environments are set up.

### 1. no_ref (blind VSR, baseline)

```bash
conda activate sparkvsr
cd sparkvsr_improved
CUDA_VISIBLE_DEVICES=0 python sparkvsr_inference_script.py \
    --input_dir datasets/test/UDM10/LQ-Video \
    --model_path checkpoints/SparkVSR \
    --output_path results/UDM10/no_ref \
    --is_vae_st --ref_mode no_ref --upscale 4
```

### 2. DLoRA reference mode (our main contribution)

```bash
export DLORA_HOME=/path/to/dlora_improved
conda activate sparkvsr
CUDA_VISIBLE_DEVICES=0 python sparkvsr_inference_script.py \
    --input_dir datasets/test/UDM10/LQ-Video \
    --model_path checkpoints/SparkVSR \
    --output_path results/UDM10/dlora \
    --is_vae_st --ref_mode dlora --ref_indices 0 --upscale 4 \
    --dlora_python /path/to/lora_env/bin/python \
    --dlora_script_path dloral_keyframe.py \
    --dlora_pretrained_path /path/to/model_52001.pkl \
    --dlora_sidechannel_ckpt /path/to/sidechannel_step005000.pt \
    --dlora_gpu 1
```

`DLORA_HOME` must point to the `dlora_improved/` directory. `dlora_python` must point to the `lora` conda environment's Python. `dlora_gpu` specifies which GPU the DLoRA subprocess uses.

### 4. Memory optimization for large videos

```bash
    --chunk_len 32 --tile_size_hw 512 640
```

### 5. Run evaluation

```bash
export HF_ENDPOINT=https://hf-mirror.com   # if HuggingFace blocked
bash run_eval_all.sh \
    --pred results/UDM10/dlora \
    --gt datasets/test/UDM10/GT-Video \
    --out results/UDM10/dlora \
    --metrics psnr,ssim,lpips,dists,clipiqa,musiq,dover,fastvqa \
    --gpu_id 0
```

## Results

### UDM10 (10 clips, synthetic BD degradation)

| Metric | no_ref | + PiSA-SR | + DLoRA (ours) |
|---|---|---|---|
| PSNR | 29.66 | 28.72 | 26.73 |
| SSIM | 0.868 | 0.841 | 0.790 |
| CLIPIQA | 0.454 | 0.294 | **0.593** |
| MUSIQ | 59.57 | 50.86 | **67.93** |
| DOVER Overall | 0.618 | 0.511 | **0.687** |

### SPMCS (30 clips, mixed degradation)

| Metric | no_ref | + PiSA-SR | + DLoRA (ours) |
|---|---|---|---|
| PSNR | 18.99 | 17.12 | 18.67 |
| CLIPIQA | 0.545 | **0.706** | 0.607 |
| MUSIQ | 67.57 | **73.92** | 70.18 |
| DOVER Overall | 0.498 | **0.548** | 0.490 |

DLoRA achieves best NR metrics on UDM10; PiSA-SR leads on SPMCS real degradation. For DLoRA standalone results and keyframe-count ablation, see our `eval_log.md`.




