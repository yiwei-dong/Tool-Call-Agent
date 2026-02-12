import torch
import torch.nn as nn
import torch.nn.functional as F

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU loss for bounding boxes.
    Expects boxes in [x1, y1, x2, y2] format.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)

    lt_enc = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]

    return iou - (area_enc - union) / (area_enc + 1e-6)


class DecoupledAgentLoss(nn.Module):
    """Loss function combining CrossEntropy for actions and L1/GIoU for boxes."""

    def __init__(self):
        super().__init__()
        self.weights = {"action": 1.0, "box_l1": 5.0, "box_giou": 2.0}
        self.action_criterion = nn.CrossEntropyLoss()

    def forward(self, preds: dict, targets: dict):
        total_loss = torch.tensor(0.0, device=preds['action_logits'].device)
        loss_dict = {}

        # 1. Action Loss
        if 'action_id' in targets:
            l_act = self.action_criterion(preds['action_logits'], targets['action_id'])
            loss_dict['loss_action'] = l_act
            total_loss += self.weights['action'] * l_act

        # 2. Box Regression Loss
        if 'boxes' in targets and 'boxes_mask' in targets:
            mask = targets['boxes_mask'] > 0
            if mask.sum() > 0:
                p_box = preds['box_preds'][mask]
                t_box = targets['boxes'][mask]

                # L1 Loss
                l1 = F.l1_loss(p_box, t_box)

                # GIoU Loss (Convert cxcywh to x1y1x2y2)
                p_cnr = torch.cat([p_box[:, :2] - p_box[:, 2:] / 2, p_box[:, :2] + p_box[:, 2:] / 2], dim=-1)
                t_cnr = torch.cat([t_box[:, :2] - t_box[:, 2:] / 2, t_box[:, :2] + t_box[:, 2:] / 2], dim=-1)
                giou_val = torch.diag(generalized_box_iou(p_cnr, t_cnr))
                giou_loss = (1 - giou_val).mean()

                loss_dict['loss_box_l1'] = l1
                loss_dict['loss_box_giou'] = giou_loss
                total_loss += self.weights['box_l1'] * l1 + self.weights['box_giou'] * giou_loss

        return total_loss, loss_dict