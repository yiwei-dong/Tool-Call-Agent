"""
Training Script - FULLY FIXED VERSION
All critical, strongly recommended, and nice-to-have fixes applied.
"""
import argparse
import sys
import os
import yaml
import torch
import deepspeed
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from tqdm import tqdm
import json

# 1. 获取 src 目录的绝对路径
src_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取项目根目录 (dissertation)
project_root = os.path.dirname(src_dir)

# 3. 将这两个路径加入 Python 搜索列表
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)  # 优先搜索 src
if project_root not in sys.path:
    sys.path.insert(0, project_root) # 其次搜索根目录

from agent_model import Qwen2_5_VL_Agent
from utils.heads import AgentConfig
from dataset import ToolHeadsDataset, custom_collate_fn


def parse_args():
    """✅ NEW: Improved argument parsing"""
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL Agent")
    parser.add_argument("--config_path", type=str, default="configs/models/qwen2_5_vl.yaml",
                       help="Path to model config YAML")
    parser.add_argument("--data_file", type=str, required=True,
                       help="Path to training JSONL file")
    parser.add_argument("--image_root", type=str, default="./images",
                       help="Root directory for images")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="DeepSpeed local rank")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=-1,
                       help="Save checkpoint every N steps (-1 = only at epoch end)")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation after each epoch")
    parser.add_argument("--val_data_file", type=str, default=None,
                       help="Path to validation JSONL file")
    
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def validate_config(config_data):
    """✅ NEW: Validate configuration"""
    required_keys = ["model_id"]
    for key in required_keys:
        if key not in config_data:
            raise ValueError(f"Missing required config key: {key}")
    
    # Set defaults
    config_data.setdefault("epochs", 3)
    config_data.setdefault("lm_loss_weight", 1.0)
    config_data.setdefault("agent_config", {})
    
    return config_data


def setup_model_and_processor(config_data, args):
    """✅ NEW: Separate model setup for clarity"""
    # Setup Processor
    print("Setting up Processor...")
    processor = AutoProcessor.from_pretrained(
        config_data["model_id"],
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28
    )

    # Add special tokens
    special_tokens = ["[EXECUTE]", "<textlist>", "</textlist>", "<text>", "</text>"]
    num_added = processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
    print(f"Added {num_added} special tokens")
    
    execute_token_id = processor.tokenizer.convert_tokens_to_ids("[EXECUTE]")
    print(f"[EXECUTE] token ID: {execute_token_id}")
    
    if execute_token_id == processor.tokenizer.unk_token_id:
        raise ValueError("[EXECUTE] token was not added correctly!")

    # Setup Model
    print("Setting up Model...")
    agent_config = AgentConfig(**config_data.get("agent_config", {}))

    model = Qwen2_5_VL_Agent(
        model_path=config_data["model_id"],
        agent_config=agent_config,
        execute_token_id=execute_token_id,
        lora_config=config_data.get("lora", None),
        lm_loss_weight=config_data.get("lm_loss_weight", 1.0),  # ✅ FIX: Configurable
        freeze_backbone=config_data.get("freeze_backbone", True)  # ✅ NEW: Configurable
    )

    # Resize embeddings to account for new tokens
    print(f"Resizing token embeddings from {len(processor.tokenizer)} to match tokenizer...")
    model.backbone.resize_token_embeddings(len(processor.tokenizer))

    # ✅ FIX: Enable gradient checkpointing to save memory
    if config_data.get("gradient_checkpointing", True):
        print("Enabling gradient checkpointing...")
        model.backbone.gradient_checkpointing_enable()

    return model, processor, execute_token_id


def log_training_progress(epoch, step, global_step, loss, outputs, grad_norm=None):
    """✅ NEW: Centralized logging"""
    log_str = f"Epoch {epoch} | Step {step} | Global {global_step} | Loss: {loss:.4f}"
    if grad_norm is not None:
        log_str += f" | Grad Norm: {grad_norm:.4f}"
    print(log_str)
    
    # Print detailed loss breakdown
    loss_dict = outputs.get('loss_dict', {})
    if loss_dict:
        loss_items = [f"{k}={v:.4f}" for k, v in loss_dict.items()]
        print(f"  └─ {' | '.join(loss_items)}")


def compute_gradient_norm(model_engine):
    """✅ NEW: Compute gradient norm for monitoring"""
    total_norm = 0.0
    num_params = 0
    for p in model_engine.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num_params += 1
    
    if num_params > 0:
        total_norm = (total_norm ** 0.5)
    return total_norm


def validate_model(model_engine, val_dataloader, device):
    """✅ NEW: Validation loop"""
    model_engine.eval()
    total_loss = 0.0
    num_batches = 0
    
    print("\n" + "="*50)
    print("Running Validation...")
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            pixel_values = batch['pixel_values'].to(device) if batch['pixel_values'] is not None else None
            image_grid_thw = batch['image_grid_thw'].to(device) if batch['image_grid_thw'] is not None else None
            
            targets = {
                'action_ids': batch['targets']['action_ids'].to(device),
                'box_targets': batch['targets']['box_targets'].to(device),
                'box_masks': batch['targets']['box_masks'].to(device),
                'img1_labels': batch['targets']['img1_labels'].to(device),
                'img2_labels': batch['targets']['img2_labels'].to(device),
                'img_multi_labels': batch['targets']['img_multi_labels'].to(device)
            }
            num_valid_images = batch['targets']['num_valid_images'].to(device)

            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                num_valid_images=num_valid_images,
                targets=targets,
                labels=labels
            )

            loss = outputs['loss']
            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1
    
    avg_val_loss = total_loss / max(num_batches, 1)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print("="*50 + "\n")
    
    model_engine.train()
    return avg_val_loss


def save_checkpoint(model_engine, processor, save_path, epoch, global_step, config_data):
    """✅ NEW: Enhanced checkpoint saving"""
    if model_engine.local_rank == 0:
        print(f"\n💾 Saving checkpoint to {save_path}...")
        
        # Save training metadata
        metadata = {
            "epoch": epoch,
            "global_step": global_step,
            "config": config_data
        }
        
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        processor.save_pretrained(save_path)
        print(f"✅ Checkpoint saved successfully!")


def main():
    args = parse_args()

    # 1. Load and validate config
    print("Loading configuration...")
    with open(args.config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    config_data = validate_config(config_data)
    print(f"Config loaded: {json.dumps(config_data, indent=2)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Setup model and processor
    model, processor, execute_token_id = setup_model_and_processor(config_data, args)

    # 3. Prepare Dataset
    print(f"\nLoading training data from {args.data_file}...")
    train_dataset = ToolHeadsDataset(args.data_file, processor, args.image_root)
    
    # ✅ FIX: Validate dataset is not empty
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty! Check data file: {args.data_file}")
    
    print(f"✅ Training dataset loaded: {len(train_dataset)} samples")

    # ✅ NEW: Optional validation dataset
    val_dataset = None
    if args.validate and args.val_data_file:
        print(f"Loading validation data from {args.val_data_file}...")
        val_dataset = ToolHeadsDataset(args.val_data_file, processor, args.image_root)
        print(f"✅ Validation dataset loaded: {len(val_dataset)} samples")

    # 4. Initialize DeepSpeed
    print("\nInitializing DeepSpeed...")
    parameters_to_train = [p for p in model.parameters() if p.requires_grad]
    
    num_trainable = sum(p.numel() for p in parameters_to_train)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_trainable:,} / {num_total:,} "
          f"({100.0 * num_trainable / num_total:.2f}%)")

    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=parameters_to_train,
        collate_fn=custom_collate_fn,
        training_data=train_dataset
    )

    # ✅ NEW: Setup validation dataloader if needed
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=model_engine.train_micro_batch_size_per_gpu(),
            collate_fn=custom_collate_fn,
            shuffle=False
        )

    # 5. Training Loop
    print("\n" + "="*70)
    print("🚀 Starting DeepSpeed Training...")
    print("="*70 + "\n")
    
    global_step = 0
    num_epochs = config_data.get("epochs", 3)
    device = model_engine.device

    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"📊 EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*70}\n")
        
        model_engine.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # ✅ NEW: Use tqdm for progress bar
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(pbar):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # ✅ FIX: Now we use labels
            pixel_values = batch['pixel_values'].to(device) if batch['pixel_values'] is not None else None
            image_grid_thw = batch['image_grid_thw'].to(device) if batch['image_grid_thw'] is not None else None
            
            # ✅ FIX: Properly restructure targets to match model expectations
            targets = {
                'action_ids': batch['targets']['action_ids'].to(device),
                'box_targets': batch['targets']['box_targets'].to(device),
                'box_masks': batch['targets']['box_masks'].to(device),
                'img1_labels': batch['targets']['img1_labels'].to(device),
                'img2_labels': batch['targets']['img2_labels'].to(device),
                'img_multi_labels': batch['targets']['img_multi_labels'].to(device)
            }
            num_valid_images = batch['targets']['num_valid_images'].to(device)

            # Forward pass
            try:
                outputs = model_engine(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    num_valid_images=num_valid_images,
                    targets=targets,
                    labels=labels  # ✅ FIX: Now passes labels
                )
            except Exception as e:
                print(f"\n❌ Forward pass failed at step {global_step}: {e}")
                print(f"Batch shapes: input_ids={input_ids.shape}, "
                      f"pixel_values={pixel_values.shape if pixel_values is not None else None}")
                raise

            loss = outputs['loss']
            
            # ✅ FIX: Check for NaN loss
            if torch.isnan(loss):
                print(f"\n⚠️  WARNING: NaN loss detected at step {global_step}")
                print(f"Loss dict: {outputs['loss_dict']}")
                print("Skipping this batch...")
                continue

            # Backward and optimize
            model_engine.backward(loss)
            model_engine.step()

            # Track statistics
            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})

            # ✅ FIX: Enhanced logging with gradient norm
            if global_step % args.log_interval == 0 and model_engine.local_rank == 0:
                grad_norm = compute_gradient_norm(model_engine) if global_step % 50 == 0 else None
                log_training_progress(epoch, step, global_step, loss_val, outputs, grad_norm)
            
            # ✅ NEW: Mid-epoch checkpointing
            if args.save_interval > 0 and global_step % args.save_interval == 0:
                save_path = os.path.join(args.output_dir, f"step_{global_step}")
                model_engine.save_checkpoint(save_path)
                save_checkpoint(model_engine, processor, save_path, epoch, global_step, config_data)

        # End of epoch
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        
        if model_engine.local_rank == 0:
            print(f"\n{'='*70}")
            print(f"✅ Epoch {epoch} completed!")
            print(f"   Average Loss: {avg_epoch_loss:.4f}")
            print(f"   Total Batches: {num_batches}")
            print(f"   Global Steps: {global_step}")
            print(f"{'='*70}\n")

        # ✅ NEW: Run validation if requested
        if args.validate and val_dataloader is not None:
            val_loss = validate_model(model_engine, val_dataloader, device)

        # Save epoch checkpoint
        save_path = os.path.join(args.output_dir, f"epoch_{epoch}")
        model_engine.save_checkpoint(save_path)
        save_checkpoint(model_engine, processor, save_path, epoch, global_step, config_data)

    # ✅ NEW: Print final statistics
    if model_engine.local_rank == 0:
        print("\n" + "="*70)
        print("🎉 Training Completed!")
        print("="*70)
        model.print_training_stats()
        
        # Save final model
        final_path = os.path.join(args.output_dir, "final_model")
        model_engine.save_checkpoint(final_path)
        save_checkpoint(model_engine, processor, final_path, num_epochs, global_step, config_data)
        print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
