"""
Agent Head Module for Qwen2.5-VL Agent.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    hidden_size: int = 1536
    visual_hidden_size: int = 1280
    num_actions: int = 13
    max_history_images: int = 100
    max_boxes: int = 50
    num_heads: int = 8
    dropout: float = 0.1


class ProgramSlotAgentHead(nn.Module):
    """
    Predicts action type, bounding boxes, and image pointer indices from
    the hidden state at the [EXECUTE] token position.

    Architecture (three-phase):
      Phase 1 – Action prediction via transformer decoder on action queries.
      Phase 2 – Slot queries conditioned on the predicted action embedding
                (and optional visual bias) decoded through the same decoder.
      Phase 3 – Per-slot linear projections for boxes and image pointers.
    """

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        H = config.hidden_size

        # Learnable query tensors
        self.program_embeddings   = nn.Embedding(config.num_actions, H)
        self.action_queries       = nn.Parameter(torch.randn(1, 4, H) * 0.02)
        self.box_queries          = nn.Parameter(torch.randn(1, config.max_boxes, H) * 0.02)
        self.image_queries        = nn.Parameter(torch.randn(1, 4, H) * 0.02)
        self.multi_image_queries  = nn.Parameter(torch.randn(1, 4, H) * 0.02)

        # Optional visual context projection
        self.visual_context_proj = nn.Sequential(
            nn.Linear(config.visual_hidden_size, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Linear(H, H),
        )

        # Shared transformer decoder (used for both phases)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=H,
            nhead=config.num_heads,
            dim_feedforward=H * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,     # pre-LN for better training stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Output heads
        self.action_head = nn.Linear(H, config.num_actions)

        self.box_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, 4),
            nn.Sigmoid(),        # output in [0, 1] for normalized coords
        )

        self.image_ptr1_head  = nn.Linear(H, config.max_history_images)
        self.image_ptr2_head  = nn.Linear(H, config.max_history_images)
        self.image_multi_head = nn.Linear(H, config.max_history_images)

    def forward(
        self,
        memory: torch.Tensor,
        pure_visual_features: Optional[torch.Tensor] = None,
        num_valid_images=None,
    ):
        """
        Args:
            memory:               [B, Seq, H] – usually the single [EXECUTE] feature.
            pure_visual_features: Optional [B, visual_hidden_size] extra visual bias.
            num_valid_images:     [B] int tensor or list; masks out invalid image slots.

        Returns:
            action_logits:    [B, num_actions]
            box_preds:        [B, max_boxes, 4]  – (cx, cy, w, h) normalised
            img1_logits:      [B, max_history_images]
            img2_logits:      [B, max_history_images]
            img_multi_logits: [B, max_history_images]
        """
        B = memory.size(0)
        device = memory.device
        dtype = memory.dtype

        # ── Optional visual bias (correctly typed) ────────────────────────
        # FIX: was `visual_bias = 0` (int), causing implicit broadcast issues
        # when tensors have non-standard dtypes (e.g. bfloat16).
        if pure_visual_features is not None:
            visual_bias = self.visual_context_proj(pure_visual_features).unsqueeze(1)
            # [B, 1, H] – broadcasts over the query sequence dimension
        else:
            visual_bias = torch.zeros(1, 1, self.config.hidden_size,
                                      device=device, dtype=dtype)

        # ── Phase 1: Action prediction ────────────────────────────────────
        action_queries = self.action_queries.expand(B, -1, -1)
        action_feats   = self.decoder(tgt=action_queries, memory=memory)
        action_logits  = self.action_head(action_feats.mean(dim=1))   # [B, num_actions]

        # Differentiable program embedding:
        #   • Training  → Gumbel-Softmax (hard=True keeps discrete, straight-through
        #                 estimator passes gradient through).
        #   • Inference → hard argmax (deterministic, no randomness).
        if self.training:
            action_probs  = F.gumbel_softmax(action_logits, tau=1.0, hard=True)
            program_embed = action_probs @ self.program_embeddings.weight  # [B, H]
        else:
            action_ids    = action_logits.argmax(dim=-1)
            program_embed = self.program_embeddings(action_ids)            # [B, H]

        # ── Phase 2: Slot queries conditioned on action + visual context ──
        intent = program_embed.unsqueeze(1)  # [B, 1, H]

        def _make_query(param: nn.Parameter) -> torch.Tensor:
            """Expand a learnable query parameter and inject intent + visual bias."""
            return param.expand(B, -1, -1) + intent + visual_bias

        slot_queries = torch.cat(
            [
                _make_query(self.box_queries),
                _make_query(self.image_queries),
                _make_query(self.multi_image_queries),
            ],
            dim=1,
        )
        slot_feats = self.decoder(tgt=slot_queries, memory=memory)

        # ── Phase 3: Decode each slot group ──────────────────────────────
        n_box = self.config.max_boxes
        box_feats   = slot_feats[:, :n_box]           # [B, max_boxes, H]
        img_feats   = slot_feats[:, n_box: n_box + 4] # [B, 4, H]
        multi_feats = slot_feats[:, n_box + 4:]       # [B, 4, H]

        box_preds        = self.box_head(box_feats)                   # [B, max_boxes, 4]
        img1_logits      = self.image_ptr1_head(img_feats[:, 0])      # [B, max_hist]
        img2_logits      = self.image_ptr2_head(img_feats[:, 1])      # [B, max_hist]
        img_multi_logits = self.image_multi_head(multi_feats.mean(1)) # [B, max_hist]

        # ── Mask invalid image slots ──────────────────────────────────────
        if num_valid_images is not None:
            max_hist = img1_logits.size(-1)
            if not isinstance(num_valid_images, torch.Tensor):
                num_valid_images = torch.tensor(
                    num_valid_images, device=device, dtype=torch.long
                )
            # Clamp so we never mask everything (at least slot 0 stays valid)
            num_valid_images = num_valid_images.reshape(-1).clamp(min=1)
            idx     = torch.arange(max_hist, device=device).unsqueeze(0)  # [1, max_hist]
            invalid = idx >= num_valid_images.unsqueeze(1)                 # [B, max_hist]
            NEG_INF = torch.finfo(img1_logits.dtype).min
            img1_logits      = img1_logits.masked_fill(invalid, NEG_INF)
            img2_logits      = img2_logits.masked_fill(invalid, NEG_INF)
            img_multi_logits = img_multi_logits.masked_fill(invalid, NEG_INF)

        return action_logits, box_preds, img1_logits, img2_logits, img_multi_logits