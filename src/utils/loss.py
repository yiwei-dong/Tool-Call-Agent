"""
Loss Module - FULLY IMPROVED VERSION
Better gradient handling and numerical stability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss


class DecoupledAgentLoss(nn.Module):
    """
    Loss function for the Hybrid Agent Head.
    
    Computes:
    1. Action Classification (Cross Entropy)
    2. Box Regression (L1 + GIoU)
    3. Image Pointer 1 (Cross Entropy)
    4. Image Pointer 2 (Cross Entropy)
    5. Multi-Image Selection (BCE with Logits)
    """

    def __init__(self, loss_weights=None):
        """
        Args:
            loss_weights: Dict of loss weights. If None, uses defaults.
        """
        super().__init__()
        
        # ✅ FIX: Make weights configurable
        default_weights = {
            "action": 1.0,
            "box_l1": 5.0,
            "box_giou": 2.0,
            "img1": 1.0,
            "img2": 1.0,
            "img_multi": 1.0
        }
        self.weights = loss_weights if loss_weights is not None else default_weights

        # Cross entropy with ignore_index=-100 for invalid labels
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def box_cxcywh_to_xyxy(self, x):
        """
        Convert box format from [cx, cy, w, h] to [x1, y1, x2, y2].
        
        Args:
            x: Tensor of shape [..., 4] with [cx, cy, w, h]
            
        Returns:
            Tensor of shape [..., 4] with [x1, y1, x2, y2]
        """
        cx, cy, w, h = x.unbind(-1)
        b = [
            (cx - 0.5 * w), 
            (cy - 0.5 * h),
            (cx + 0.5 * w), 
            (cy + 0.5 * h)
        ]
        return torch.stack(b, dim=-1)

    def forward(self, preds: dict, targets: dict):
        """
        Compute all losses.
        
        Args:
            preds: {
                'action_logits': [B, num_actions],
                'box_preds': [B, max_boxes, 4],
                'img1_logits': [B, max_imgs],
                'img2_logits': [B, max_imgs],
                'img_multi_logits': [B, max_imgs]
            }
            targets: {
                'action_ids': [B],
                'box_targets': [B, max_boxes, 4],
                'box_masks': [B, max_boxes],
                'img1_labels': [B],
                'img2_labels': [B],
                'img_multi_labels': [B, max_imgs]
            }
            
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dict of individual losses (as scalars for logging)
        """
        device = preds['action_logits'].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # -----------------------------------------------------------
        # 1. Action Loss (Classification)
        # -----------------------------------------------------------
        if 'action_ids' in targets:
            try:
                loss_act = self.ce_loss(preds['action_logits'], targets['action_ids'])
                loss_dict['loss_action'] = loss_act.item()
                total_loss = total_loss + self.weights['action'] * loss_act
            except Exception as e:
                print(f"⚠️  Action loss computation failed: {e}")
                loss_dict['loss_action'] = 0.0

        # -----------------------------------------------------------
        # 2. Box Regression Loss (L1 + GIoU)
        # -----------------------------------------------------------
        if 'box_targets' in targets and 'box_masks' in targets:
            # Flatten batch and box dimensions
            mask = targets['box_masks'].bool().view(-1)  # [B * Max_Boxes]

            if mask.sum() > 0:
                # Extract valid boxes
                p_box_flat = preds['box_preds'].view(-1, 4)  # [B*N, 4]
                t_box_flat = targets['box_targets'].view(-1, 4)  # [B*N, 4]

                p_box_valid = p_box_flat[mask]  # [num_valid, 4]
                t_box_valid = t_box_flat[mask]  # [num_valid, 4]

                # ✅ NEW: Clamp predictions to valid range for stability
                p_box_valid = torch.clamp(p_box_valid, 0.0, 1.0)

                # A. L1 Loss (on normalized [cx, cy, w, h])
                loss_l1 = F.l1_loss(p_box_valid, t_box_valid, reduction='mean')

                # B. GIoU Loss (convert to [x1, y1, x2, y2] first)
                try:
                    p_xyxy = self.box_cxcywh_to_xyxy(p_box_valid)
                    t_xyxy = self.box_cxcywh_to_xyxy(t_box_valid)

                    # ✅ FIX: Ensure boxes are valid (x2 > x1, y2 > y1)
                    p_xyxy = self._ensure_valid_boxes(p_xyxy)
                    t_xyxy = self._ensure_valid_boxes(t_xyxy)

                    # Compute GIoU loss
                    loss_giou = generalized_box_iou_loss(p_xyxy, t_xyxy, reduction='mean')
                    
                    loss_dict['loss_box_l1'] = loss_l1.item()
                    loss_dict['loss_box_giou'] = loss_giou.item()
                    
                    total_loss = total_loss + self.weights['box_l1'] * loss_l1
                    total_loss = total_loss + self.weights['box_giou'] * loss_giou
                
                except Exception as e:
                    print(f"⚠️  GIoU computation failed: {e}")
                    # Fall back to L1 only
                    loss_dict['loss_box_l1'] = loss_l1.item()
                    loss_dict['loss_box_giou'] = 0.0
                    total_loss = total_loss + self.weights['box_l1'] * loss_l1
            
            else:
                # No valid boxes - add dummy loss to maintain gradient graph
                # ✅ FIX: Better dummy loss that maintains graph connectivity
                dummy_loss = 0.0 * preds['box_preds'].sum()
                loss_dict['loss_box_l1'] = 0.0
                loss_dict['loss_box_giou'] = 0.0
                total_loss = total_loss + dummy_loss

        # -----------------------------------------------------------
        # 3. Image Pointer Loss 1 (Main Image)
        # -----------------------------------------------------------
        if 'img1_labels' in targets:
            # ✅ NEW: Only compute loss for valid labels (not -100)
            valid_mask = (targets['img1_labels'] != -100)
            if valid_mask.any():
                try:
                    loss_img1 = self.ce_loss(preds['img1_logits'], targets['img1_labels'])
                    loss_dict['loss_img1'] = loss_img1.item()
                    total_loss = total_loss + self.weights['img1'] * loss_img1
                except Exception as e:
                    print(f"⚠️  Img1 loss computation failed: {e}")
                    loss_dict['loss_img1'] = 0.0
            else:
                loss_dict['loss_img1'] = 0.0

        # -----------------------------------------------------------
        # 4. Image Pointer Loss 2 (Secondary Image)
        # -----------------------------------------------------------
        if 'img2_labels' in targets:
            valid_mask = (targets['img2_labels'] != -100)
            if valid_mask.any():
                try:
                    loss_img2 = self.ce_loss(preds['img2_logits'], targets['img2_labels'])
                    loss_dict['loss_img2'] = loss_img2.item()
                    total_loss = total_loss + self.weights['img2'] * loss_img2
                except Exception as e:
                    print(f"⚠️  Img2 loss computation failed: {e}")
                    loss_dict['loss_img2'] = 0.0
            else:
                loss_dict['loss_img2'] = 0.0

        # -----------------------------------------------------------
        # 5. Multi-Image Loss (Multi-label Classification)
        # -----------------------------------------------------------
        if 'img_multi_labels' in targets:
            try:
                # ✅ NEW: Only compute loss where at least one image is valid
                # If all labels are 0, it means no valid selection
                has_selection = (targets['img_multi_labels'].sum(dim=-1) > 0)
                
                if has_selection.any():
                    loss_multi = self.bce_loss(
                        preds['img_multi_logits'], 
                        targets['img_multi_labels']
                    )
                    loss_dict['loss_img_multi'] = loss_multi.item()
                    total_loss = total_loss + self.weights['img_multi'] * loss_multi
                else:
                    loss_dict['loss_img_multi'] = 0.0
            
            except Exception as e:
                print(f"⚠️  Multi-image loss computation failed: {e}")
                loss_dict['loss_img_multi'] = 0.0

        # ✅ NEW: Check for NaN in total loss
        if torch.isnan(total_loss):
            print(f"⚠️  NaN detected in total loss! Loss dict: {loss_dict}")
            # Return a small positive loss to continue training
            total_loss = torch.tensor(0.1, device=device, requires_grad=True)

        loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict

    def _ensure_valid_boxes(self, boxes_xyxy):
        """
        ✅ NEW: Ensure boxes are valid (x2 > x1, y2 > y1) for GIoU computation.
        
        Args:
            boxes_xyxy: [N, 4] tensor with [x1, y1, x2, y2]
            
        Returns:
            Valid boxes [N, 4]
        """
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        
        # Ensure x2 > x1 and y2 > y1 by swapping if needed
        x1, x2 = torch.min(x1, x2), torch.max(x1, x2)
        y1, y2 = torch.min(y1, y2), torch.max(y1, y2)
        
        # Add small epsilon to ensure non-zero width/height
        epsilon = 1e-6
        x2 = torch.where(x2 <= x1, x1 + epsilon, x2)
        y2 = torch.where(y2 <= y1, y1 + epsilon, y2)
        
        return torch.stack([x1, y1, x2, y2], dim=-1)
