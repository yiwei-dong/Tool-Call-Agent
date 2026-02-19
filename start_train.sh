#!/bin/bash
# Quick Start Training Script
# 快速开始训练脚本

set -e  # Exit on error

echo "=================================="
echo "🚀 Qwen2.5-VL Agent Training"
echo "=================================="
echo ""

# ============================================================================
# Configuration (修改这些参数以适配你的环境)
# ============================================================================

# Data paths
DATA_FILE="dataset/VTS_SFT_ToolHeads_new_1K.jsonl"
VAL_DATA_FILE="dataset/debug_mini_20.jsonl"  # Optional
IMAGE_ROOT="dataset/VTS_SFT"

# Model and config
CONFIG_PATH="configs/qwen2_5_vl.json"
OUTPUT_DIR="checkpoints/$(date +%Y%m%d_%H%M%S)"

# DeepSpeed (optional, for multi-GPU)
USE_DEEPSPEED=true
NUM_GPUS=3
DEEPSPEED_CONFIG="configs/config.json"

# Training params
LOG_INTERVAL=50
SAVE_INTERVAL=50  # Save every N steps, -1 for epoch only
USE_VALIDATION=false
MAX_LENGTH=4096

# WandB (set to false to disable)
USE_WANDB=true
WANDB_PROJECT="qwen2.5-vl-agent"
WANDB_RUN_NAME="run-v1-hybrid"


# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"

# ============================================================================
# Build Training Command (构建训练命令)
# ============================================================================

echo ""
echo "🔧 Building training command..."

BASE_CMD="python -m src.train \
    --data_file $DATA_FILE \
    --image_root $IMAGE_ROOT \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --log_interval $LOG_INTERVAL"

# Add optional parameters
if [ "$SAVE_INTERVAL" != "-1" ]; then
    BASE_CMD="$BASE_CMD --save_interval $SAVE_INTERVAL"
fi

if [ "$USE_VALIDATION" = true ] && [ -f "$VAL_DATA_FILE" ]; then
    BASE_CMD="$BASE_CMD --validate --val_data_file $VAL_DATA_FILE"
fi

# DeepSpeed multi-GPU training
if [ "$USE_DEEPSPEED" = true ]; then
    if [ ! -f "$DEEPSPEED_CONFIG" ]; then
        echo "❌ Error: DeepSpeed config not found: $DEEPSPEED_CONFIG"
        exit 1
    fi
    
    echo "🚀 Using DeepSpeed with $NUM_GPUS GPUs"
    TRAIN_CMD="deepspeed --num_gpus=$NUM_GPUS src/train.py \
        --data_file $DATA_FILE \
        --image_root $IMAGE_ROOT \
        --config_path $CONFIG_PATH \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name $WANDB_RUN_NAME \
        --deepspeed \
        --deepspeed_config $DEEPSPEED_CONFIG \
        --output_dir $OUTPUT_DIR \
        --log_interval $LOG_INTERVAL"
    
    if [ "$SAVE_INTERVAL" != "-1" ]; then
        TRAIN_CMD="$TRAIN_CMD --save_interval $SAVE_INTERVAL"
    fi
    
    if [ "$USE_VALIDATION" = true ] && [ -f "$VAL_DATA_FILE" ]; then
        TRAIN_CMD="$TRAIN_CMD --validate --val_data_file $VAL_DATA_FILE"
    fi
else
    TRAIN_CMD="$BASE_CMD"
fi

# ============================================================================
# Start Training (开始训练)
# ============================================================================

echo ""
echo "🚀 Starting training..."
echo "Command: $TRAIN_CMD"
echo ""

# Save command to file for reference
echo "$TRAIN_CMD" > "$OUTPUT_DIR/train_command.sh"
chmod +x "$OUTPUT_DIR/train_command.sh"

# Start training
eval $TRAIN_CMD

# ============================================================================
# Post-Training (训练完成后)
# ============================================================================

echo "=================================="
echo "✅ Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=================================="