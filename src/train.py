"""
Training Script for Qwen2.5-VL Agent using DeepSpeed.
"""
import argparse
import os
import sys
import json

import torch
import deepspeed
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from tqdm import tqdm

import torch.distributed as dist

# Ensure both src/ and project root are importable
_src_dir     = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_src_dir)
for _p in (_src_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_model import Qwen2_5_VL_Agent
from utils.heads import AgentConfig
from dataset import ToolHeadsDataset, custom_collate_fn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL Agent")
    parser.add_argument("--config_path",   type=str, default="configs/qwen2_5_vl.json")
    parser.add_argument("--data_file",     type=str, required=True)
    parser.add_argument("--val_data_file", type=str, default=None)
    parser.add_argument("--image_root",    type=str, default="./images")
    parser.add_argument("--output_dir",    type=str, default="./checkpoints")
    parser.add_argument("--local_rank",    type=int, default=-1)
    parser.add_argument("--log_interval",  type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=-1,
                        help="Save every N steps; -1 = epoch end only")
    parser.add_argument("--validate",      action="store_true")
    parser.add_argument("--wandb_project", type=str, default="qwen2.5-vl-agent")
    parser.add_argument("--wandb_run_name",type=str, default=None)
    parser.add_argument("--wandb_entity",  type=str, default=None)
    parser.add_argument("--no_wandb",      action="store_true")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_config(cfg: dict) -> dict:
    if "model_id" not in cfg:
        raise ValueError("Config must contain 'model_id'.")
    cfg.setdefault("epochs", 3)
    cfg.setdefault("lm_loss_weight", 1.0)
    cfg.setdefault("agent_config", {})
    cfg.setdefault("freeze_backbone", True)
    cfg.setdefault("gradient_checkpointing", True)
    cfg.setdefault("max_length", 2048)
    cfg.setdefault("head_lr_multiplier", 5.0)   # head LR = backbone LR × multiplier
    cfg.setdefault("loss_weights", None)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Model and processor setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_model_and_processor(cfg: dict):
    print("Setting up processor...")
    processor = AutoProcessor.from_pretrained(
        cfg["model_id"],
        min_pixels=256  * 28 * 28,
        max_pixels=768 * 28 * 28,
    )

    special_tokens = [
        "[EXECUTE]", "<textlist>", "</textlist>",
        "<text>", "</text>", "<action>", "</action>",
    ]
    n_added = processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
    print(f"Added {n_added} special tokens.")

    execute_token_id = processor.tokenizer.convert_tokens_to_ids("[EXECUTE]")
    if execute_token_id == processor.tokenizer.unk_token_id:
        raise ValueError("[EXECUTE] token was not added correctly.")
    print(f"[EXECUTE] token ID: {execute_token_id}")

    # FIX: compute vocabulary size here and pass it into the model constructor
    # so that resize_token_embeddings() runs BEFORE get_peft_model() is called.
    # Previously resize happened after LoRA wrapping, which meant the new
    # embedding rows were frozen and could never be trained.
    vocab_size = len(processor.tokenizer)

    print("Setting up model...")
    agent_config = AgentConfig(**cfg.get("agent_config", {}))
    model = Qwen2_5_VL_Agent(
        model_path=cfg["model_id"],
        agent_config=agent_config,
        execute_token_id=execute_token_id,
        lora_config=cfg.get("lora"),
        lm_loss_weight=cfg.get("lm_loss_weight", 1.0),
        freeze_backbone=cfg.get("freeze_backbone", True),
        vocab_size=vocab_size,                   # ← pass here, resize before LoRA
        loss_weights=cfg.get("loss_weights"),
    )

    if cfg.get("gradient_checkpointing", True):
        try:
            # Access underlying model when backbone is a PeftModel
            base = (
                model.backbone.base_model.model
                if hasattr(model.backbone, "base_model")
                else model.backbone
            )
            base.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled.")
        except Exception as e:
            print(f"Warning: Could not enable gradient checkpointing: {e}")

    return model, processor, execute_token_id


# ─────────────────────────────────────────────────────────────────────────────
# Trainable parameter setup
# ─────────────────────────────────────────────────────────────────────────────

def get_param_groups(model: Qwen2_5_VL_Agent, base_lr: float, head_lr_multiplier: float):
    """
    FIX: Previously `trainable_params` was built from `p.requires_grad` AFTER
    get_peft_model, which missed agent_head (frozen by PEFT by default).

    Now we explicitly separate backbone parameters from agent_head parameters
    and optionally apply a higher learning rate to the head for faster
    convergence of the newly initialised head weights.
    """
    # Ensure agent_head is always trainable (safety net)
    for p in model.agent_head.parameters():
        p.requires_grad = True

    backbone_params = [
        p for p in model.backbone.parameters() if p.requires_grad
    ]
    head_params = list(model.agent_head.parameters())

    n_backbone = sum(p.numel() for p in backbone_params)
    n_head     = sum(p.numel() for p in head_params)
    n_total    = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable — backbone: {n_backbone:,}  head: {n_head:,}  "
        f"total trainable: {n_backbone + n_head:,} / {n_total:,}"
    )

    return [
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params,     "lr": base_lr * head_lr_multiplier},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint saving
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model_engine,
    processor,
    save_path: str,
    epoch: int,
    global_step: int,
    cfg: dict,
    merge_lora: bool = False,
):
    """
    Save agent head weights, optional LoRA adapters (or merged backbone),
    processor, and training-state metadata.

    Args:
        merge_lora:  When True (recommended for the final checkpoint), merge
                     LoRA adapters into the base model and save the merged
                     weights so inference requires no PEFT dependency.
    """
    if model_engine.local_rank != 0:
        return

    os.makedirs(save_path, exist_ok=True)
    raw_model = model_engine.module

    # ── Agent head ────────────────────────────────────────────────────────
    torch.save(
        raw_model.agent_head.state_dict(),
        os.path.join(save_path, "agent_head.pth"),
    )

    # ── Backbone / LoRA ───────────────────────────────────────────────────
    lora_enabled = cfg.get("lora", {}).get("enabled", False)
    if lora_enabled:
        if merge_lora:
            # FIX: merge LoRA into base model for clean, PEFT-free inference.
            # merge_and_unload() returns the merged base model; capture it.
            try:
                print("Merging LoRA weights into backbone for deployment...")
                merged = raw_model.backbone.merge_and_unload()
                merged.save_pretrained(os.path.join(save_path, "merged_backbone"))
                print(f"Merged backbone saved → {save_path}/merged_backbone")
            except Exception as e:
                print(f"Warning: LoRA merge failed ({e}); saving raw adapters instead.")
                raw_model.backbone.save_pretrained(
                    os.path.join(save_path, "lora_adapters")
                )
        else:
            try:
                raw_model.backbone.save_pretrained(
                    os.path.join(save_path, "lora_adapters")
                )
            except Exception as e:
                print(f"Warning: Could not save LoRA adapters: {e}")

    # ── Processor, config, metadata ───────────────────────────────────────
    processor.save_pretrained(os.path.join(save_path, "processor"))
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(save_path, "training_state.json"), "w") as f:
        json.dump(
            {"epoch": epoch, "global_step": global_step, "model_id": cfg["model_id"]},
            f,
            indent=2,
        )

    print(f"Checkpoint saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_step(epoch, step, global_step, loss_val, loss_dict, lr=None):
    parts = [
        f"Epoch {epoch} | Step {step} | Global {global_step} | Loss: {loss_val:.4f}"
    ]
    if lr is not None:
        parts.append(f"LR: {lr:.2e}")
    print(" | ".join(parts))
    if loss_dict:
        # Show lm loss and head losses clearly; total_loss = lm + all heads
        keys_order = ["loss_lm", "loss_action", "loss_box_l1", "loss_box_giou",
                      "loss_img1", "loss_img2", "loss_img_multi",
                      "loss_head_total", "total_loss"]
        ordered = {k: loss_dict[k] for k in keys_order if k in loss_dict}
        remainder = {k: v for k, v in loss_dict.items() if k not in ordered}
        ordered.update(remainder)
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in ordered.items()))

    if WANDB_AVAILABLE and wandb.run is not None:
        log_data = {
            "train/total_loss": loss_val,
            "train/epoch": epoch,
            "train/global_step": global_step,
        }
        if lr is not None:
            log_data["train/learning_rate"] = lr
        for k, v in loss_dict.items():
            log_data[f"train/{k}"] = v
        wandb.log(log_data, step=global_step)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_model(model_engine, val_dataloader, device, epoch):
    model_engine.eval()
    total_loss, num_batches = 0.0, 0
    loss_dict_sum: dict[str, float] = {}

    print("Running validation...")
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            outputs = _forward_batch(model_engine, batch, device)
            total_loss  += outputs["loss"].item()
            num_batches += 1
            for k, v in outputs.get("loss_dict", {}).items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + v

    avg     = total_loss / max(num_batches, 1)
    avg_dict = {k: v / num_batches for k, v in loss_dict_sum.items()}
    print(
        f"Validation loss: {avg:.4f}  "
        + "  ".join(f"{k}={v:.4f}" for k, v in avg_dict.items())
    )

    if WANDB_AVAILABLE and wandb.run is not None:
        log_data = {"val/total_loss": avg, "val/epoch": epoch}
        for k, v in avg_dict.items():
            log_data[f"val/{k}"] = v
        wandb.log(log_data)

    model_engine.train()
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Shared batch forward helper
# ─────────────────────────────────────────────────────────────────────────────

def _forward_batch(model_engine, batch, device):
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["labels"].to(device)
    pixel_values   = (
        batch["pixel_values"].to(device)  if batch["pixel_values"]   is not None else None
    )
    image_grid_thw = (
        batch["image_grid_thw"].to(device) if batch["image_grid_thw"] is not None else None
    )

    # Separate num_valid_images (not a loss target) from head supervision targets
    targets = {
        k: v.to(device)
        for k, v in batch["targets"].items()
        if k != "num_valid_images"
    }
    num_valid_images = batch["targets"]["num_valid_images"].to(device)

    return model_engine(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        num_valid_images=num_valid_images,
        targets=targets,
        labels=labels,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("Loading configuration...")
    with open(args.config_path) as f:
        cfg = validate_config(json.load(f))
    print(json.dumps(cfg, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)

    # is_main is True only for rank 0 (or single-GPU runs)
    is_main = args.local_rank in (-1, 0)

    # ── WandB ─────────────────────────────────────────────────────────────
    if WANDB_AVAILABLE and not args.no_wandb and is_main:
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                config=cfg,
            )
            print(f"WandB run: {wandb.run.name}")
        except Exception as e:
            print(f"Warning: WandB init failed: {e}")

    # ── Model + processor ─────────────────────────────────────────────────
    model, processor, _ = setup_model_and_processor(cfg)

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = ToolHeadsDataset(
        args.data_file, processor, args.image_root,
        max_length=cfg.get("max_length", 2048),
    )
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty: {args.data_file}")

    val_dataset = None
    if args.validate and args.val_data_file:
        val_dataset = ToolHeadsDataset(
            args.val_data_file, processor, args.image_root,
            max_length=cfg.get("max_length", 2048),
        )

    # ── Trainable parameters with per-group learning rates ────────────────
    # Read base LR from the DeepSpeed optimizer config so everything stays
    # in sync with the DS scheduler.
    base_lr = cfg.get("optimizer", {}).get("params", {}).get("lr", 2e-5)
    # Fallback: try to read from the DS config file if present
    if hasattr(args, "deepspeed_config") and args.deepspeed_config:
        try:
            with open(args.deepspeed_config) as f:
                ds_cfg = json.load(f)
            base_lr = ds_cfg.get("optimizer", {}).get("params", {}).get("lr", base_lr)
        except Exception:
            pass

    param_groups = get_param_groups(
        model,
        base_lr=base_lr,
        head_lr_multiplier=cfg.get("head_lr_multiplier", 5.0),
    )

    # ── DeepSpeed init ────────────────────────────────────────────────────
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=param_groups,
        training_data=train_dataset,
        collate_fn=custom_collate_fn,
    )
    device = model_engine.device

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=model_engine.train_micro_batch_size_per_gpu(),
            collate_fn=custom_collate_fn,
            shuffle=False,
        )

    # ── Training loop ─────────────────────────────────────────────────────
    global_step = 0
    num_epochs  = cfg.get("epochs", 3)

    for epoch in range(num_epochs):
        model_engine.train()
        epoch_loss, num_batches = 0.0, 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for step, batch in enumerate(pbar):
            outputs  = _forward_batch(model_engine, batch, device)
            loss     = outputs["loss"]

            model_engine.backward(loss)
            model_engine.step()

            loss_val     = loss.item()
            epoch_loss  += loss_val
            num_batches += 1
            global_step += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            if global_step % args.log_interval == 0 and is_main:
                lr = None
                if hasattr(model_engine, "lr_scheduler"):
                    lr = model_engine.lr_scheduler.get_last_lr()[0]
                elif hasattr(optimizer, "param_groups"):
                    lr = optimizer.param_groups[0]["lr"]
                log_step(epoch + 1, step, global_step, loss_val,
                         outputs.get("loss_dict", {}), lr)

            # Mid-epoch checkpoint (no LoRA merge to avoid slowdown)
            # if args.save_interval > 0 and global_step % args.save_interval == 0:
            #     ckpt_path = os.path.join(args.output_dir, f"step_{global_step}")
            #     model_engine.save_checkpoint(ckpt_path)
            #     save_checkpoint(
            #         model_engine, processor, ckpt_path,
            #         epoch, global_step, cfg, merge_lora=False,
            #     )

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        if is_main:
            print(f"Epoch {epoch + 1} complete — avg loss: {avg_epoch_loss:.4f}")

        if args.validate and val_dataloader is not None and is_main:
            validate_model(model_engine, val_dataloader, device, epoch + 1)

        # End-of-epoch checkpoint (no merge; keep adapters separate)
        # if is_main:
        #     ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
        #     model_engine.save_checkpoint(ckpt_path)
        #     save_checkpoint(
        #         model_engine, processor, ckpt_path,
        #         epoch + 1, global_step, cfg, merge_lora=False,
        #     )

    # ── Final model — merge LoRA into backbone for clean inference ─────────
    final_path = os.path.join(args.output_dir, "final_model")
    model_engine.save_checkpoint(final_path)
    
    if dist.is_initialized():
        dist.barrier()
    
    if is_main:
        # FIX: merge_lora=True here produces a standalone model that the
        # Reasoner can load without any PEFT dependency.
        save_checkpoint(
            model_engine, processor, final_path,
            num_epochs, global_step, cfg, merge_lora=True,
        )
        print(f"Training complete. Final model saved → {final_path}")
    
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
    
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()