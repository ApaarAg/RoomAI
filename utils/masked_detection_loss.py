"""Class-conditional masked detection loss for partially annotated multi-dataset training (RoomAI Phase 20).

Addresses the critical partial-annotation defect in public datasets (e.g. Open Images):
When an image is annotated only for a subset of classes (e.g. porch), unannotated classes
(windows, doors, facade) must NOT be penalized as background negatives.

Provides:
- MaskedDetectionLoss(v8DetectionLoss): Masks out classification loss for unverified classes per image.
- VerifiedClassRegistry: Metadata lookup mapping image filenames to verified class masks [0.0 or 1.0].
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss, make_anchors


class MaskedDetectionLoss(v8DetectionLoss):
    """YOLO detection loss with per-image class-conditional masking for partially annotated datasets."""

    def __init__(
        self,
        model: torch.nn.Module,
        verified_mask_lookup: Optional[Dict[str, List[float]]] = None,
        tal_topk: int = 10,
        tal_topk2: Optional[int] = None,
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        # verified_mask_lookup maps image filename (e.g. 'oi_pilot_train_0001.jpg')
        # to a list/tensor of length nc with 1.0 for verified classes, 0.0 for unverified classes.
        self.verified_mask_lookup: Dict[str, List[float]] = verified_mask_lookup or {}

    def set_verified_mask_lookup(self, lookup: Dict[str, List[float]]) -> None:
        """Update or register verified mask lookup dictionary."""
        self.verified_mask_lookup = lookup

    def get_assigned_targets_and_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple:
        """Calculate detection loss with class-conditional masking for unannotated classes."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss with optional class weighting
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights

        # === CLASS-CONDITIONAL LOSS MASKING ===
        if self.verified_mask_lookup:
            im_files = batch.get("im_file", [])
            mask_list = []
            for f in im_files:
                fname = Path(f).name
                # Default to all ones (fully verified) if not explicitly registered
                m = self.verified_mask_lookup.get(fname, [1.0] * self.nc)
                mask_list.append(m)

            if len(mask_list) == batch_size:
                valid_mask = torch.tensor(mask_list, device=self.device, dtype=dtype).unsqueeze(1)
                bce_loss = bce_loss * valid_mask

        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        box_gain = self.hyp["box"] if isinstance(self.hyp, dict) else getattr(self.hyp, "box", 7.5)
        cls_gain = self.hyp["cls"] if isinstance(self.hyp, dict) else getattr(self.hyp, "cls", 0.5)
        dfl_gain = self.hyp["dfl"] if isinstance(self.hyp, dict) else getattr(self.hyp, "dfl", 1.5)

        loss[0] *= box_gain
        loss[1] *= cls_gain
        loss[2] *= dfl_gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )
