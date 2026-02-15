import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class AgentConfig:
    hidden_size: int = 1536
    visual_hidden_size: int = 1280
    num_actions: int = 13
    max_history_images: int = 10
    max_boxes: int = 5
    num_heads: int = 8
    dropout: float = 0.1

class ProgramSlotAgentHead(nn.Module):
    """The specialized head that predicts actions and visual slots."""

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        self.program_embeddings = nn.Embedding(config.num_actions, config.hidden_size)
        self.action_queries = nn.Parameter(torch.randn(1, 4, config.hidden_size))
        self.box_queries = nn.Parameter(torch.randn(1, config.max_boxes, config.hidden_size))
        self.image_queries = nn.Parameter(torch.randn(1, 4, config.hidden_size))
        self.multi_image_queries = nn.Parameter(torch.randn(1, 4, config.hidden_size))

        self.visual_context_proj = nn.Sequential(
            nn.Linear(config.visual_hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size, nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4, dropout=config.dropout,
            batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.action_head = nn.Linear(config.hidden_size, config.num_actions)
        self.box_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2), nn.GELU(),
            nn.Linear(config.hidden_size // 2, 4), nn.Sigmoid()
        )
        self.image_ptr1_head = nn.Linear(config.hidden_size, config.max_history_images)
        self.image_ptr2_head = nn.Linear(config.hidden_size, config.max_history_images)
        self.image_multi_head = nn.Linear(config.hidden_size, config.max_history_images)

    def forward(self, memory, attention_mask=None, pure_visual_features=None, num_valid_images=None):
        B = memory.size(0)

        # Visual Bias
        visual_bias = 0
        if pure_visual_features is not None:
            visual_bias = self.visual_context_proj(pure_visual_features).unsqueeze(1)

        # Phase 1: Action Prediction
        action_queries = self.action_queries.repeat(B, 1, 1)
        # memory shape: [B, Seq, Hidden] (Usually Seq=1 for execute token)
        action_feats = self.decoder(tgt=action_queries, memory=memory)
        action_feature = action_feats.mean(dim=1)  # Pooling
        action_logits = self.action_head(action_feature)

        # Get Action Embedding (Support Gumbel Softmax for End-to-End training)
        if self.training:
            action_probs = F.gumbel_softmax(action_logits, tau=1.0, hard=True)
            program_embed = torch.matmul(action_probs, self.program_embeddings.weight)
        else:
            action_id = action_logits.argmax(dim=-1)
            program_embed = self.program_embeddings(action_id)

        # Phase 2: Slot Query Construction
        def condition_query(base_query):
            base = base_query.repeat(B, 1, 1)
            intent = program_embed.unsqueeze(1)
            return base + intent + visual_bias

        slot_queries = torch.cat([
            condition_query(self.box_queries),
            condition_query(self.image_queries),
            condition_query(self.multi_image_queries)
        ], dim=1)

        slot_feats = self.decoder(tgt=slot_queries, memory=memory)

        # Phase 3: Slice and Predict
        offset = 0
        box_feats = slot_feats[:, offset: offset + self.config.max_boxes]
        offset += self.config.max_boxes
        img_feats = slot_feats[:, offset: offset + 4]
        offset += 4

        box_preds = self.box_head(box_feats)
        img1_logits = self.image_ptr1_head(img_feats[:, 0])
        img2_logits = self.image_ptr2_head(img_feats[:, 1])
        img_multi_logits = self.image_multi_head(slot_feats[:, -4:].mean(dim=1))

        # Mask invalid images
        if num_valid_images is not None:
            # num_valid_images: list or tensor of size [B]
            max_hist = img1_logits.size(-1)
            # Create mask: [B, max_hist] where True means INVALID
            mask_indices = torch.arange(max_hist, device=img1_logits.device).expand(B, max_hist)
            if isinstance(num_valid_images, list):
                num_valid_images = torch.tensor(num_valid_images, device=img1_logits.device)

            valid_limit = num_valid_images.unsqueeze(1)
            invalid_mask = mask_indices >= valid_limit

            img1_logits.masked_fill_(invalid_mask, float('-inf'))
            img2_logits.masked_fill_(invalid_mask, float('-inf'))
            img_multi_logits.masked_fill_(invalid_mask, float('-inf'))

        return action_logits, box_preds, img1_logits, img2_logits, img_multi_logits