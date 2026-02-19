"""
Loss Module for the Hybrid Agent Head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss


class DecoupledAgentLoss(nn.Module):
    """
    Computes all training losses for the agent head:
      1. Action classification     (cross-entropy, with configurable class weights)
      2. Box regression            (L1 + GIoU, applied only to active box slots)
      3. Image pointer 1           (cross-entropy)
      4. Image pointer 2           (cross-entropy)
      5. Multi-image selection     (BCE with logits, applied only to samples
                                   that have at least one positive label)

    All individual loss terms are guarded against NaN / Inf before being
    accumulated so that a single bad batch cannot corrupt the run.
    """

    DEFAULT_WEIGHTS: dict = {
        "action":    1.0,
        "box_l1":    2.0,   # reduced from 5.0 to avoid box dominating multi-task
        "box_giou":  1.0,   # reduced from 2.0
        "img1":      1.0,
        "img2":      1.0,
        "img_multi": 1.0,
    }

    def __init__(
        self,
        loss_weights: dict = None,
        terminate_class_idx: int = 12,
        terminate_class_weight: float = 2.0,
    ):
        """
        Args:
            loss_weights:            Dict overriding DEFAULT_WEIGHTS keys.
            terminate_class_idx:     Index of the Terminate action class.
            terminate_class_weight:  Up-weight factor for the Terminate class
                                     in the action CE loss.
        """
        super().__init__()
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if loss_weights:
            self.weights.update(loss_weights)

        self.terminate_class_idx    = terminate_class_idx
        self.terminate_class_weight = terminate_class_weight

        # ignore_index=-100 matches the label convention used in dataset.py
        self.ce_loss  = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # ─────────────────────────────────────────────────────────────────────
    # Static helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """[..., 4]: (cx, cy, w, h) → (x1, y1, x2, y2)."""
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack(
            [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
        )

    @staticmethod
    def _ensure_valid_boxes(boxes: torch.Tensor) -> torch.Tensor:
        """Guarantee x2 > x1 and y2 > y1 (required for GIoU stability)."""
        x1, y1, x2, y2 = boxes.unbind(-1)
        x1, x2 = torch.minimum(x1, x2), torch.maximum(x1, x2)
        y1, y2 = torch.minimum(y1, y2), torch.maximum(y1, y2)
        eps = 1e-6
        x2 = torch.where(x2 <= x1, x1 + eps, x2)
        y2 = torch.where(y2 <= y1, y1 + eps, y2)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _safe(loss: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
        """Return (loss, ok) where ok=False means the value was nan/inf."""
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: {name} produced {'nan' if torch.isnan(loss) else 'inf'}; skipping.")
            return loss, False
        return loss, True

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(self, preds: dict, targets: dict):
        """
        Args:
            preds: {
                'action_logits':    [B, num_actions],
                'box_preds':        [B, max_boxes, 4],
                'img1_logits':      [B, max_imgs],
                'img2_logits':      [B, max_imgs],
                'img_multi_logits': [B, max_imgs],
            }
            targets: {
                'action_ids':       [B],
                'box_targets':      [B, max_boxes, 4],
                'box_masks':        [B, max_boxes],
                'img1_labels':      [B],
                'img2_labels':      [B],
                'img_multi_labels': [B, max_imgs],
            }

        Returns:
            total_loss (Tensor), loss_dict (dict of float values for logging)
        """
        device = preds["action_logits"].device
        total_loss = torch.zeros(1, device=device,
                                 dtype=preds["action_logits"].dtype).squeeze()
        loss_dict: dict[str, float] = {}

        # ── 1. Action classification ──────────────────────────────────────
        if "action_ids" in targets:
            n_cls = preds["action_logits"].shape[-1]
            class_weights = torch.ones(n_cls, device=device,
                                       dtype=preds["action_logits"].dtype)
            # FIX: was hardcoded [12]; now uses configurable attribute so it
            # stays correct if num_actions changes.
            if 0 <= self.terminate_class_idx < n_cls:
                class_weights[self.terminate_class_idx] = self.terminate_class_weight

            loss_act = F.cross_entropy(
                preds["action_logits"],
                targets["action_ids"],
                weight=class_weights,
                ignore_index=-100,
                reduction="mean",
            )
            loss_act, ok = self._safe(loss_act, "loss_action")
            if ok:
                total_loss = total_loss + self.weights["action"] * loss_act
            loss_dict["loss_action"] = loss_act.item() if ok else 0.0

        # ── 2. Box regression (L1 + GIoU) ────────────────────────────────
        if "box_targets" in targets and "box_masks" in targets:
            mask = targets["box_masks"].bool().view(-1)
            if mask.sum() > 0:
                p_flat = preds["box_preds"].view(-1, 4)[mask]
                t_flat = targets["box_targets"].view(-1, 4)[mask]
                p_flat = torch.clamp(p_flat, 0.0, 1.0)

                loss_l1 = F.l1_loss(p_flat, t_flat, reduction="mean")
                loss_l1, ok = self._safe(loss_l1, "loss_box_l1")
                if ok:
                    total_loss = total_loss + self.weights["box_l1"] * loss_l1
                loss_dict["loss_box_l1"] = loss_l1.item() if ok else 0.0

                p_xyxy  = self._ensure_valid_boxes(self._cxcywh_to_xyxy(p_flat))
                t_xyxy  = self._ensure_valid_boxes(self._cxcywh_to_xyxy(t_flat))
                loss_giou = generalized_box_iou_loss(p_xyxy, t_xyxy, reduction="mean")
                loss_giou, ok = self._safe(loss_giou, "loss_box_giou")
                if ok:
                    total_loss = total_loss + self.weights["box_giou"] * loss_giou
                loss_dict["loss_box_giou"] = loss_giou.item() if ok else 0.0
            else:
                loss_dict["loss_box_l1"]  = 0.0
                loss_dict["loss_box_giou"] = 0.0

        # ── 3. Image pointer 1 ────────────────────────────────────────────
        if "img1_labels" in targets:
            valid = targets["img1_labels"] != -100
            if valid.any():
                # FIX: guard against all-inf logits (can happen when all image
                # slots are masked), which produces nan from cross-entropy.
                logits_1 = preds["img1_logits"][valid]
                labels_1 = targets["img1_labels"][valid]
                if logits_1.isfinite().any(dim=-1).all():
                    loss_img1 = self.ce_loss(logits_1, labels_1)
                    loss_img1, ok = self._safe(loss_img1, "loss_img1")
                    if ok:
                        total_loss = total_loss + self.weights["img1"] * loss_img1
                    loss_dict["loss_img1"] = loss_img1.item() if ok else 0.0
                else:
                    loss_dict["loss_img1"] = 0.0
            else:
                loss_dict["loss_img1"] = 0.0

        # ── 4. Image pointer 2 ────────────────────────────────────────────
        if "img2_labels" in targets:
            valid = targets["img2_labels"] != -100
            if valid.any():
                logits_2 = preds["img2_logits"][valid]
                labels_2 = targets["img2_labels"][valid]
                if logits_2.isfinite().any(dim=-1).all():
                    loss_img2 = self.ce_loss(logits_2, labels_2)
                    loss_img2, ok = self._safe(loss_img2, "loss_img2")
                    if ok:
                        total_loss = total_loss + self.weights["img2"] * loss_img2
                    loss_dict["loss_img2"] = loss_img2.item() if ok else 0.0
                else:
                    loss_dict["loss_img2"] = 0.0
            else:
                loss_dict["loss_img2"] = 0.0

        # ── 5. Multi-image BCE ────────────────────────────────────────────
        if "img_multi_labels" in targets:
            labels_m = targets["img_multi_labels"].float()
            logits_m = torch.clamp(preds["img_multi_logits"], -50, 50)
            has_sel  = labels_m.sum(dim=-1) > 0
            if has_sel.any():
                sel_logits = logits_m[has_sel]
                sel_labels = labels_m[has_sel]
                if sel_logits.isfinite().any(dim=-1).all():
                    loss_multi = self.bce_loss(sel_logits, sel_labels)
                    loss_multi, ok = self._safe(loss_multi, "loss_img_multi")
                    if ok:
                        total_loss = total_loss + self.weights["img_multi"] * loss_multi
                    loss_dict["loss_img_multi"] = loss_multi.item() if ok else 0.0
                else:
                    loss_dict["loss_img_multi"] = 0.0
            else:
                loss_dict["loss_img_multi"] = 0.0

        # "loss_head_total" = sum of all HEAD losses only (excludes lm_loss).
        # The true training total is logged separately by agent_model.py.
        loss_dict["loss_head_total"] = total_loss.item()
        return total_loss, loss_dict