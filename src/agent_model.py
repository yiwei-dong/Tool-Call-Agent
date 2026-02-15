"""
Agent Model - FULLY FIXED VERSION
All critical, strongly recommended, and nice-to-have fixes applied.
"""
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from utils.heads import ProgramSlotAgentHead, AgentConfig
from utils.loss import DecoupledAgentLoss


class Qwen2_5_VL_Agent(nn.Module):
    """Wrapper that combines Qwen2.5-VL_3B backbone with the Agent Head."""

    def __init__(
        self, 
        model_path, 
        agent_config: AgentConfig, 
        execute_token_id: int, 
        lora_config=None, 
        lm_loss_weight=1.0,  # ✅ FIX: Made configurable
        freeze_backbone=True  # ✅ NEW: Explicit control
    ):
        super().__init__()
        self.config = agent_config
        self.execute_token_id = execute_token_id
        self.lm_loss_weight = lm_loss_weight
        self.freeze_backbone = freeze_backbone

        # 1. Load Backbone
        print(f"Loading backbone from {model_path}...")
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Synchronize hidden sizes
        self.config.hidden_size = self.backbone.config.hidden_size
        if hasattr(self.backbone, 'visual') and hasattr(self.backbone.visual, "config"):
            # Try different possible attribute names for visual hidden size
            visual_config = self.backbone.visual.config
            if hasattr(visual_config, 'embed_dim'):
                self.config.visual_hidden_size = visual_config.embed_dim
            elif hasattr(visual_config, 'hidden_size'):
                self.config.visual_hidden_size = visual_config.hidden_size
            elif hasattr(visual_config, 'd_model'):
                self.config.visual_hidden_size = visual_config.d_model
            else:
                # Fallback: use a default value
                print("⚠️  Warning: Could not determine visual hidden size, using default 1280")
                self.config.visual_hidden_size = 1280
        
        print(f"Model config: hidden_size={self.config.hidden_size}, "
              f"visual_hidden_size={self.config.visual_hidden_size}")

        # 2. Initialize Agent Head
        print("Initializing Agent Head...")
        self.agent_head = ProgramSlotAgentHead(self.config)
        self.agent_head.to(dtype=self.backbone.dtype)

        # 3. Apply LoRA if enabled
        if lora_config and lora_config.get("enabled", False):
            print("Applying LoRA to backbone...")
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("alpha", 32),
                lora_dropout=lora_config.get("dropout", 0.05),
                target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"])
            )
            self.backbone = get_peft_model(self.backbone, peft_cfg)
            self.backbone.print_trainable_parameters()
        elif freeze_backbone:
            # Freeze backbone, train head only
            print("Freezing backbone, training head only.")
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.agent_head.parameters():
                param.requires_grad = True
        else:
            print("Training full model (backbone + head).")

        self.loss_fn = DecoupledAgentLoss()
        
        # ✅ NEW: Track training statistics
        self.training_stats = {
            'total_steps': 0,
            'samples_without_execute': 0,
            'nan_losses': 0
        }

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        pixel_values=None, 
        image_grid_thw=None, 
        num_valid_images=None, 
        targets=None, 
        labels=None,  # ✅ FIX: Added labels parameter
        return_logits=False,  # ✅ NEW: Option to return logits for inference
        **kwargs
    ):
        """
        Forward pass with comprehensive error handling.
        
        Args:
            input_ids: [B, Seq] token IDs
            attention_mask: [B, Seq] attention mask
            pixel_values: Flattened image tensors
            image_grid_thw: Image grid dimensions
            num_valid_images: [B] or list, number of valid images per sample
            targets: Dict with keys: action_ids, box_targets, box_masks, img1_labels, img2_labels, img_multi_labels
            labels: [B, Seq] labels for LM loss (pad tokens should be -100)
            return_logits: Whether to return logits for inference
        
        Returns:
            Dict with 'loss', 'loss_dict', and optionally 'logits'
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # ✅ NEW: Input validation
        if self.training:
            assert targets is not None, "targets must be provided during training"
        
        # 1. Backbone Forward
        try:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,  # ✅ FIX: Now passes labels for LM loss
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
        except Exception as e:
            print(f"❌ Backbone forward failed: {e}")
            raise

        sequence_output = outputs.hidden_states[-1]  # [B, Seq, Hidden]
        lm_loss = outputs.loss

        # 2. Locate [EXECUTE] token
        is_execute = (input_ids == self.execute_token_id)
        
        # ✅ FIX: Better handling when no execute token exists
        if not is_execute.any():
            self.training_stats['samples_without_execute'] += batch_size
            
            if lm_loss is not None:
                return {
                    "loss": lm_loss, 
                    "loss_dict": {"loss_lm": lm_loss.item()},
                    "logits": None
                }
            else:
                # ✅ FIX: Maintain gradient graph with dummy loss
                dummy_loss = 0.0 * sequence_output.sum()
                return {
                    "loss": dummy_loss, 
                    "loss_dict": {"loss_lm": 0.0},
                    "logits": None
                }

        # Get indices of last [EXECUTE] token per sequence
        # Flip to find last occurrence
        flipped_is_exec = torch.flip(is_execute, dims=[1])
        last_idx_flipped = flipped_is_exec.int().argmax(dim=1)
        seq_len = input_ids.size(1)
        execute_indices = seq_len - 1 - last_idx_flipped

        # ✅ NEW: Validate execute indices
        if (execute_indices >= seq_len).any() or (execute_indices < 0).any():
            print(f"⚠️  Warning: Invalid execute indices detected: {execute_indices}")
            execute_indices = torch.clamp(execute_indices, 0, seq_len - 1)

        # Extract features at [EXECUTE] position: [B, 1, Hidden]
        batch_indices = torch.arange(batch_size, device=device)
        execute_features = sequence_output[batch_indices, execute_indices].unsqueeze(1)

        # ✅ FIX: Ensure head is on correct device
        if self.agent_head.action_queries.device != device:
            print(f"Moving agent_head to {device}")
            self.agent_head.to(device)

        # 3. Head Forward
        try:
            action_logits, box_preds, img1_logits, img2_logits, img_multi_logits = self.agent_head(
                memory=execute_features,
                num_valid_images=num_valid_images
            )
        except Exception as e:
            print(f"❌ Agent head forward failed: {e}")
            print(f"   execute_features shape: {execute_features.shape}")
            print(f"   num_valid_images: {num_valid_images}")
            raise

        logits = {
            "action_logits": action_logits,
            "box_preds": box_preds,
            "img1_logits": img1_logits,
            "img2_logits": img2_logits,
            "img_multi_logits": img_multi_logits
        }

        # 4. Calculate Loss
        # ✅ FIX: Proper loss initialization without requires_grad
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # Add LM loss if available
        if lm_loss is not None:
            total_loss = total_loss + self.lm_loss_weight * lm_loss
            loss_dict['loss_lm'] = lm_loss.item()
        else:
            loss_dict['loss_lm'] = 0.0

        # Add Head losses if targets provided
        if targets is not None:
            # ✅ FIX: Changed from 'action_id' to 'action_ids' to match dataset
            if 'action_ids' in targets:
                try:
                    head_loss, head_loss_dict = self.loss_fn(logits, targets)
                    total_loss = total_loss + head_loss
                    
                    # ✅ NEW: Convert tensors to scalars for logging
                    for k, v in head_loss_dict.items():
                        loss_dict[k] = v.item() if isinstance(v, torch.Tensor) else v
                        
                except Exception as e:
                    print(f"❌ Loss computation failed: {e}")
                    print(f"   Logit shapes: action={action_logits.shape}, "
                          f"box={box_preds.shape}, img1={img1_logits.shape}")
                    print(f"   Target keys: {targets.keys()}")
                    raise

        # ✅ NEW: Check for NaN loss
        if torch.isnan(total_loss):
            self.training_stats['nan_losses'] += 1
            print(f"⚠️  NaN loss detected! Loss dict: {loss_dict}")
            # Return a dummy loss to continue training
            total_loss = torch.tensor(1.0, device=device, requires_grad=True)

        self.training_stats['total_steps'] += 1

        result = {
            "loss": total_loss, 
            "loss_dict": loss_dict
        }
        
        if return_logits:
            result["logits"] = logits
            
        return result

    def print_training_stats(self):
        """✅ NEW: Print training statistics"""
        print("\n" + "="*50)
        print("Training Statistics:")
        print(f"  Total steps: {self.training_stats['total_steps']}")
        print(f"  Samples without [EXECUTE]: {self.training_stats['samples_without_execute']}")
        print(f"  NaN losses encountered: {self.training_stats['nan_losses']}")
        print("="*50 + "\n")
