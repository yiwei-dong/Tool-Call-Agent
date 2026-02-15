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
DATA_FILE="./dataset/debug_mini_20.jsonl"
VAL_DATA_FILE="./dataset/debug_mini_20.jsonl"  # Optional
IMAGE_ROOT=".dataset/VTS_SFT/images"

# Model and config
CONFIG_PATH="configs/qwen2_5_vl.yaml"
OUTPUT_DIR="checkpoints/$(date +%Y%m%d_%H%M%S)"

# DeepSpeed (optional, for multi-GPU)
USE_DEEPSPEED=true
NUM_GPUS=1
DEEPSPEED_CONFIG="configs/config.json"

# Training params
LOG_INTERVAL=10
SAVE_INTERVAL=500  # Save every N steps, -1 for epoch only
USE_VALIDATION=false


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
# Display Configuration (显示配置)
# ============================================================================

echo ""
echo "=================================="
echo "📊 Training Configuration"
echo "=================================="
echo "Data File: $DATA_FILE"
echo "Image Root: $IMAGE_ROOT"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "DeepSpeed: $USE_DEEPSPEED"
if [ "$USE_DEEPSPEED" = true ]; then
    echo "  GPUs: $NUM_GPUS"
    echo "  Config: $DEEPSPEED_CONFIG"
fi
echo "Validation: $USE_VALIDATION"
echo "=================================="
echo ""

# Ask for confirmation
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
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

echo ""
echo "=================================="
echo "✅ Training completed!"
echo "=================================="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Check training logs in $OUTPUT_DIR"
echo "2. Test the model:"
echo "   python -c \"from src.reasoner import Reasoner; r = Reasoner('$OUTPUT_DIR/final_model'); print(r.run_task('test.jpg', 'What is in the image?'))\""
echo "3. Continue training from checkpoint:"
echo "   bash $OUTPUT_DIR/train_command.sh"
echo ""
