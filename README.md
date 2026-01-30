# SLIM-BRAIN: A DATA- AND TRAINING-EFFICIENT FOUNDATION MODEL FOR FMRI DATA ANALYSIS

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv-2512.21881-b31b1b.svg?style=flat-square)](https://www.arxiv.org/abs/2512.21881)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github)](https://github.com/OneMore1/SLIM-Brain2026)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/OneMore1/Slim-Brain)

</div>

This repository contains the official implementation of SLIM-Brain. SLIM-Brain is a two-stage, selective-compute pipeline for voxel-level fMRI representation learning. A lightweight global branch ranks informative temporal windows; a high-capacity 4D Hiera–JEPA encoder processes only those windows, focusing compute on brain voxels and drastically reducing memory.


<p align="center">
  <img src="pipeline.png" width="800" alt="framework">
</p>

---

## Installation

Setting up the environment requires Python 3.13 and CUDA-compatible PyTorch for GPU acceleration:

```bash
conda create -n hiera-jepa python=3.13.5
conda activate hiera-jepa

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

The codebase is organized into modular components for easy navigation and extension:

```
hiera-jepa/
├── configs/                    # YAML configuration files for training and model parameters
├── checkpoints/                # Saved model weights and training checkpoints
├── hiera/                      # Hierarchical Vision Transformer backbone implementation
├── scripts/                   # Bash....
├── finetune.py               # Downstream task training and feature extraction script
└── requirements.txt            # Python package dependencies
```

## Downstream evaluation

1. Ensure your pre-train data structure as follow:

```
data_root/
├── ABIDE_train/                
├── ABIDE_val/                  
├── HCP_val/              
└── HCP_train/              
    ├── 0010001/                # Subject ID
    └── 0010002/                
        ├── 0010002_run-1_0000-0199_1.npz  # Data chunk 1 
        ├── 0010002_run-1_0000-0199_2.npz  # Data chunk 2
```

2. Loading downstream datasets as following data structure:

```yaml
task:
  csv: "/path/to/data_csv"

data:
  data_root: /path/to/data_root
  datasets: ["HCP"]
  mode: "directory"
```

3. Start downstream training:

```bash
# running downstream training
sh scripts/finetune.sh
```

#### Model Checkpoints

Our pre-trained model weights can be found in https://huggingface.co/OneMore1/Slim-Brain




