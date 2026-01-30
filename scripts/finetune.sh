#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=3  
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Configuration
CONFIG_FILE="/vePFS-0x0d/home/yewh/Hiera_MAE/configs/finetune_config.yaml"
NUM_GPUS=1  # Fixed: Changed from 0 to 2 (number of available GPUs)
MASTER_PORT=29503

# Optional: Output directory
OUTPUT_DIR="/vePFS-0x0d/home/yewh/Hiera_MAE/output/downstream/nki/age-lp3"

# Optional: Resume from checkpoint
# RESUME_CHECKPOINT="output/hiera_finetune/checkpoints/checkpoint_epoch_10.pth"

echo "Starting DDP fine-tuning with $NUM_GPUS GPUs..."
echo "Config: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"

# Launch training with torchrun (recommended for PyTorch >= 1.10)
if [ -z "$RESUME_CHECKPOINT" ]; then
    # Start from scratch (or from pretrained MAE)
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        /vePFS-0x0d/home/yewh/Hiera_MAE/finetune.py \
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR
else
    # Resume from checkpoint
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        /vePFS-0x0d/home/yewh/Hiera_MAE/finetune.py \
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR \
        --resume $RESUME_CHECKPOINT
fi

echo "Fine-tuning completed!"
