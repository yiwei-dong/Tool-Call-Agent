"""
Agent Model: Qwen2.5-VL backbone + ProgramSlot Agent Head.
"""
import os
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from utils.heads import ProgramSlotAgentHead, AgentConfig
from utils.loss import DecoupledAgentLoss


class Qwen2_5_VL_Agent(nn.Module):
    """
    Wraps the Qwen2.5-VL backbone with a specialized agent head that predicts
    action type, bounding boxes, and image pointer indices.
    """

    def __init__(
        self,
        model_path: str,
        agent_config: AgentConfig,
        execute_token_id: int,
        lora_config: dict = None,
        lm_loss_weight: float = 1.0,
        freeze_backbone: bool = True,
        vocab_size: int = None,
        loss_weights: dict = None,
    ):
        """
        Args:
            model_path:       Path to the pretrained Qwen2.5-VL model.
            agent_config:     AgentConfig dataclass with head hyperparameters.
            execute_token_id: Token ID for the [EXECUTE] special token.
            lora_config:      Dict with LoRA settings; None / not-enabled falls
                              back to freeze_backbone logic.
            lm_loss_weight:   Weight for the language-model loss term.
            freeze_backbone:  Freeze all backbone params when LoRA is disabled.
            vocab_size:       Target vocabulary size AFTER adding special tokens.
                              Resize is performed BEFORE LoRA so the new rows
                              participate in training right away.
            loss_weights:     Optional dict overriding DecoupledAgentLoss defaults.
        """
        super().__init__()
        self.config = agent_config
        self.execute_token_id = execute_token_id
        self.lm_loss_weight = lm_loss_weight

        # ── 1. Load backbone ──────────────────────────────────────────────
        print(f"Loading backbone from {model_path}...")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
        )

        # ── 2. Resize embeddings BEFORE LoRA ─────────────────────────────
        # FIX: get_peft_model freezes all base-model params. If we resize
        # after applying LoRA, the newly added token rows are frozen and can
        # never be learned. Resizing here ensures they exist before LoRA
        # wraps the model, and we explicitly unfreeze them below.
        if vocab_size and vocab_size > self.backbone.config.vocab_size:
            self.backbone.resize_token_embeddings(vocab_size)
            print(f"Token embeddings resized: {self.backbone.config.vocab_size} → {vocab_size}")

        # Sync hidden sizes from backbone config
        self.config.hidden_size = self.backbone.config.hidden_size
        if hasattr(self.backbone, "visual") and hasattr(self.backbone.visual, "config"):
            vcfg = self.backbone.visual.config
            vis_size = (
                getattr(vcfg, "embed_dim", None)
                or getattr(vcfg, "hidden_size", None)
                or getattr(vcfg, "d_model", None)
            )
            if vis_size:
                self.config.visual_hidden_size = vis_size
            else:
                print("Warning: visual hidden size unknown; using default 1280.")
        print(
            f"hidden_size={self.config.hidden_size}, "
            f"visual_hidden_size={self.config.visual_hidden_size}"
        )

        # ── 3. Initialize agent head ──────────────────────────────────────
        self.agent_head = ProgramSlotAgentHead(self.config)
        self.agent_head.to(dtype=self.backbone.dtype)

        # ── 4. LoRA / freeze strategy ─────────────────────────────────────
        lora_enabled = bool(lora_config and lora_config.get("enabled", False))

        if lora_enabled:
            print("Applying LoRA to backbone...")
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("alpha", 32),
                lora_dropout=lora_config.get("dropout", 0.05),
                target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
                modules_to_save=lora_config.get("modules_to_save", None) # 👈 ADD THIS LINE
            )
            self.backbone = get_peft_model(self.backbone, peft_cfg)
            self.backbone.print_trainable_parameters()

            # FIX (critical): get_peft_model freezes every base-model param by
            # default, which also locks out (a) the newly resized embed_tokens /
            # lm_head rows and (b) the agent_head entirely since it lives outside
            # the PeftModel wrapper. Unfreeze them explicitly here.
            for name, param in self.backbone.named_parameters():
                if "embed_tokens" in name or "lm_head" in name:
                    param.requires_grad = True
            for param in self.agent_head.parameters():
                param.requires_grad = True
            print("Also unfroze embed_tokens, lm_head, and agent_head.")

        elif freeze_backbone:
            print("Backbone frozen; training agent head only.")
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.agent_head.parameters():
                param.requires_grad = True
        else:
            print("Full model training (backbone + agent head).")
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.agent_head.parameters():
                param.requires_grad = True

        # ── 5. Loss ───────────────────────────────────────────────────────
        self.loss_fn = DecoupledAgentLoss(loss_weights=loss_weights)

    # ─────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        num_valid_images=None,
        targets=None,
        labels=None,
        return_logits=False,
    ):
        """
        Args:
            input_ids:         [B, L]
            attention_mask:    [B, L]
            pixel_values:      Flattened image tensors from the processor.
            image_grid_thw:    Image-grid meta from the processor.
            num_valid_images:  [B] number of valid images per sample.
            targets:           Dict with head supervision labels (see loss.py).
            labels:            [B, L] for LM loss; non-assistant / pad = -100.
            return_logits:     Include raw head logits in the output dict.

        Returns:
            Dict: {'loss', 'loss_dict', optionally 'logits'}
        """
        device = input_ids.device
        B = input_ids.size(0)

        # ── Backbone ──────────────────────────────────────────────────────
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]   # [B, L, H]
        lm_loss = outputs.loss               # scalar or None

        # Use a proper zero tensor so the computation graph is always valid.
        total_loss = torch.zeros(1, device=device, dtype=hidden.dtype).squeeze()
        loss_dict: dict[str, float] = {}

        if lm_loss is not None and not torch.isnan(lm_loss):
            total_loss = total_loss + self.lm_loss_weight * lm_loss
            loss_dict["loss_lm"] = lm_loss.item()
        else:
            loss_dict["loss_lm"] = 0.0

        # ── Locate last [EXECUTE] token per sequence ──────────────────────
        is_exec = input_ids == self.execute_token_id
        if not is_exec.any():
            return {"loss": total_loss, "loss_dict": loss_dict, "logits": None}

        seq_len = input_ids.size(1)
        # argmax on reversed mask gives position of the LAST True from the right
        last_idx = seq_len - 1 - torch.flip(is_exec, dims=[1]).int().argmax(dim=1)
        last_idx = torch.clamp(last_idx, 0, seq_len - 1)

        batch_idx = torch.arange(B, device=device)
        exec_feats = hidden[batch_idx, last_idx].unsqueeze(1)  # [B, 1, H]

        # ── Agent head ────────────────────────────────────────────────────
        (
            action_logits,
            box_preds,
            img1_logits,
            img2_logits,
            img_multi_logits,
        ) = self.agent_head(memory=exec_feats, num_valid_images=num_valid_images)

        head_preds = {
            "action_logits":    action_logits,
            "box_preds":        box_preds,
            "img1_logits":      img1_logits,
            "img2_logits":      img2_logits,
            "img_multi_logits": img_multi_logits,
        }

        # ── Head losses ───────────────────────────────────────────────────
        if targets is not None and "action_ids" in targets:
            head_loss, head_loss_dict = self.loss_fn(head_preds, targets)
            if not torch.isnan(head_loss):
                total_loss = total_loss + head_loss
            loss_dict.update(head_loss_dict)

        # Log the TRUE combined total (lm_loss + all head losses) so the
        # displayed "total_loss" matches the actual tensor used for backprop.
        loss_dict["total_loss"] = total_loss.item()

        result = {"loss": total_loss, "loss_dict": loss_dict}
        if return_logits:
            result["logits"] = head_preds
        return result