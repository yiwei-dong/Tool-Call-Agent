import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from utils.heads import ProgramSlotAgentHead, AgentConfig
from utils.loss import DecoupledAgentLoss

class Qwen2_5_VL_Agent(nn.Module):
    """Wrapper that combines Qwen2.5-VL_3B backbone with the Agent Head."""

    def __init__(self, model_path, agent_config: AgentConfig, execute_token_id: int, lora_config=None):
        super().__init__()
        self.config = agent_config
        self.execute_token_id = execute_token_id  # 保存触发词的 Token ID

        # 1. Load Backbone
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Synchronize hidden sizes
        self.config.hidden_size = self.backbone.config.hidden_size
        if hasattr(self.backbone.visual, "config"):
            self.config.visual_hidden_size = self.backbone.visual.config.embed_dim

        # 2. Initialize Agent Head
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
        else:
            # Freeze backbone, train head only
            print("Freezing backbone, training head only.")
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.agent_head.parameters():
                param.requires_grad = True

        self.loss_fn = DecoupledAgentLoss()
        self.lm_loss_weight = 1.0

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, num_valid_images=None,
                targets=None, **kwargs):
        # 1. Backbone Forward
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False  # Training usually doesn't use cache
        )

        sequence_output = outputs.hidden_states[-1]  # [B, Seq, Hidden]
        lm_loss = outputs.loss

        # 2. Locate [EXECUTE] token
        # logic: find the *last* occurrence of execute_token_id in each sequence
        # or rely on the specific position provided by dataset if needed.
        # Here we assume the dataset constructs the prompt such that [EXECUTE] is present.

        is_execute = (input_ids == self.execute_token_id)

        # If no execute token in batch (rare but possible), skip head loss
        if not is_execute.any():
            return {"loss": lm_loss if lm_loss is not None else torch.tensor(0.0), "loss_dict": {}}

        # Get indices: We take the last [EXECUTE] token for prediction
        # shape [B] containing index
        # Note: argmax on boolean returns first True, so we flip to find last
        flipped_is_exec = torch.flip(is_execute, dims=[1])
        last_idx_flipped = flipped_is_exec.int().argmax(dim=1)
        seq_len = input_ids.size(1)
        execute_indices = seq_len - 1 - last_idx_flipped

        # Extract features: [B, 1, Hidden]
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        execute_features = sequence_output[batch_indices, execute_indices].unsqueeze(1)

        # Ensure head is on correct device
        if self.agent_head.action_queries.device != execute_features.device:
            self.agent_head.to(execute_features.device)

        # 3. Head Forward
        action_logits, box_preds, img1_logits, img2_logits, img_multi_logits = self.agent_head(
            memory=execute_features,
            num_valid_images=num_valid_images
        )

        logits = {
            "action_logits": action_logits,
            "box_preds": box_preds,
            "img1_logits": img1_logits,
            "img2_logits": img2_logits,
            "img_multi_logits": img_multi_logits
        }

        # 4. Calculate Loss
        total_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        loss_dict = {}

        if lm_loss is not None:
            total_loss = total_loss + self.lm_loss_weight * lm_loss
            loss_dict['loss_lm'] = lm_loss

        if targets is not None:
            # Check if targets are valid (not dummy)
            if 'action_id' in targets:
                head_loss, head_loss_dict = self.loss_fn(logits, targets)
                total_loss = total_loss + head_loss
                loss_dict.update(head_loss_dict)

        return {"loss": total_loss, "loss_dict": loss_dict, "logits": logits}