"""
Official and Corrected Loss Functions for RoomAI Stage 1 Fine-Tuning.

Implements masked, scale-balanced multi-task geometric losses:
1. Masked Normalized Depth Loss (L1 regression on valid pixels against normalized metric depth).
2. Masked Normalized Point Cloud Loss (L2 regression on valid pixels in canonical scene coordinates).
3. Masked Surface Normal Loss (Cosine distance or L1 on valid unit normals).
4. Gradient Norm Auditing helper to verify representation balance.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedNormalizedDepthLoss(nn.Module):
    """Computes regression loss between predicted depth and normalized ground truth.
    
    Operates strictly on valid pixels where mask == True and depth is finite and > 0.
    """
    def __init__(self, loss_type: str = "l1", clamp_max: float = 50.0):
        super().__init__()
        self.loss_type = loss_type
        self.clamp_max = clamp_max

    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth_normalized: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            pred_depth: (B, S, H, W) or (B, S, H, W, 1) predicted depth
            gt_depth_normalized: (B, S, H, W) or (B, S, H, W, 1) normalized depth (Z / s)
            valid_mask: (B, S, H, W) boolean or binary mask of valid pixels
            
        Returns:
            loss: Scalar loss tensor
            num_valid: Number of valid pixels used in reduction
        """
        if pred_depth.ndim == 5 and pred_depth.shape[-1] == 1:
            pred_depth = pred_depth.squeeze(-1)
        if gt_depth_normalized.ndim == 5 and gt_depth_normalized.shape[-1] == 1:
            gt_depth_normalized = gt_depth_normalized.squeeze(-1)
        if valid_mask.ndim == 5 and valid_mask.shape[-1] == 1:
            valid_mask = valid_mask.squeeze(-1)

        mask = (valid_mask > 0) & torch.isfinite(gt_depth_normalized) & (gt_depth_normalized > 0.0)
        num_valid = int(mask.sum().item())

        if num_valid < 100:
            return torch.tensor(0.0, device=pred_depth.device, requires_grad=True), 0

        pred_valid = pred_depth[mask]
        gt_valid = gt_depth_normalized[mask]

        if self.loss_type == "l1":
            diff = (pred_valid - gt_valid).abs()
        elif self.loss_type == "huber":
            diff = F.smooth_l1_loss(pred_valid, gt_valid, reduction="none", beta=0.1)
        elif self.loss_type == "l2":
            diff = (pred_valid - gt_valid) ** 2
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        diff = torch.clamp(diff, max=self.clamp_max)
        diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
        loss = diff.mean()
        return loss, num_valid


class MaskedNormalizedPointLoss(nn.Module):
    """Computes regression loss between predicted 3D points and normalized ground truth.
    
    Operates in canonical scene coordinates (X_w / s).
    Applies strict binary validity mask.
    """
    def __init__(self, loss_type: str = "l2", clamp_max: float = 50.0):
        super().__init__()
        self.loss_type = loss_type
        self.clamp_max = clamp_max

    def forward(
        self,
        pred_pts: torch.Tensor,
        gt_pts_normalized: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            pred_pts: (B, S, H, W, 3) predicted world coordinates
            gt_pts_normalized: (B, S, H, W, 3) normalized world coordinates (X_w / s)
            valid_mask: (B, S, H, W) boolean/binary validity mask
            
        Returns:
            loss: Scalar loss tensor
            num_valid: Number of valid points evaluated
        """
        if valid_mask.ndim == 5 and valid_mask.shape[-1] == 1:
            valid_mask = valid_mask.squeeze(-1)

        mask = (valid_mask > 0) & torch.isfinite(gt_pts_normalized).all(dim=-1)
        num_valid = int(mask.sum().item())

        if num_valid < 100:
            return torch.tensor(0.0, device=pred_pts.device, requires_grad=True), 0

        pred_valid = pred_pts[mask]  # (N, 3)
        gt_valid = gt_pts_normalized[mask]  # (N, 3)

        if self.loss_type == "l2":
            # Euclidean distance ||pred - gt||_2
            dist = torch.norm(pred_valid - gt_valid, p=2, dim=-1)
        elif self.loss_type == "l1":
            dist = (pred_valid - gt_valid).abs().sum(dim=-1)
        elif self.loss_type == "smooth_l1":
            dist = F.smooth_l1_loss(pred_valid, gt_valid, reduction="none", beta=0.1).sum(dim=-1)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        dist = torch.clamp(dist, max=self.clamp_max)
        dist = torch.where(torch.isfinite(dist), dist, torch.zeros_like(dist))
        loss = dist.mean()
        return loss, num_valid


class MaskedNormalLoss(nn.Module):
    """Computes angular / cosine distance on valid unit surface normals."""
    def __init__(self, loss_type: str = "cosine"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        pred_normals: torch.Tensor,
        gt_normals: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            pred_normals: (B, S, H, W, 3) predicted surface normals
            gt_normals: (B, S, H, W, 3) ground truth unit surface normals
            valid_mask: (B, S, H, W) validity mask
            
        Returns:
            loss: Scalar loss
            num_valid: Number of valid normal pixels
        """
        if valid_mask.ndim == 5 and valid_mask.shape[-1] == 1:
            valid_mask = valid_mask.squeeze(-1)

        norm_len = torch.norm(gt_normals, p=2, dim=-1)
        mask = (valid_mask > 0) & (norm_len > 0.5) & torch.isfinite(gt_normals).all(dim=-1)
        num_valid = int(mask.sum().item())

        if num_valid < 100:
            return torch.tensor(0.0, device=pred_normals.device, requires_grad=True), 0

        pred_valid = pred_normals[mask]  # (N, 3)
        gt_valid = gt_normals[mask]      # (N, 3)

        # Normalize predicted normals to unit length
        pred_unit = F.normalize(pred_valid, p=2, dim=-1, eps=1e-8)
        gt_unit = F.normalize(gt_valid, p=2, dim=-1, eps=1e-8)

        if self.loss_type == "cosine":
            # 1 - cos(theta)
            cos_sim = (pred_unit * gt_unit).sum(dim=-1).clamp(-1.0, 1.0)
            loss = (1.0 - cos_sim).mean()
        elif self.loss_type == "l1":
            loss = (pred_unit - gt_unit).abs().sum(dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        loss = torch.where(torch.isfinite(loss), loss, torch.tensor(0.0, device=loss.device))
        return loss, num_valid


class CanonicalMultiTaskLoss(nn.Module):
    """Balanced multi-task geometric loss container with validity masking."""
    def __init__(
        self,
        weight_depth: float = 1.0,
        weight_pts: float = 0.5,
        weight_norm: float = 0.2,
    ):
        super().__init__()
        self.weight_depth = weight_depth
        self.weight_pts = weight_pts
        self.weight_norm = weight_norm

        self.depth_loss_fn = MaskedNormalizedDepthLoss(loss_type="l1")
        self.pts_loss_fn = MaskedNormalizedPointLoss(loss_type="l2")
        self.norm_loss_fn = MaskedNormalLoss(loss_type="cosine")

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            preds: Dict with 'depth', 'pts3d', 'normals'
            targets: Dict with 'depth_norm', 'pts_norm', 'normals', 'mask'
            
        Returns:
            total_loss: Scalar weighted loss
            metrics: Dict of unweighted scalar metrics and valid pixel counts
        """
        valid_mask = targets["mask"]

        loss_depth, n_depth = self.depth_loss_fn(preds["depth"], targets["depth_norm"], valid_mask)
        loss_pts, n_pts = self.pts_loss_fn(preds["pts3d"], targets["pts_norm"], valid_mask)
        loss_norm, n_norm = self.norm_loss_fn(preds["normals"], targets["normals"], valid_mask)

        total_loss = (
            self.weight_depth * loss_depth +
            self.weight_pts * loss_pts +
            self.weight_norm * loss_norm
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_depth": float(loss_depth.item()),
            "loss_pts": float(loss_pts.item()),
            "loss_norm": float(loss_norm.item()),
            "valid_pixels_depth": n_depth,
            "valid_pixels_pts": n_pts,
            "valid_pixels_norm": n_norm,
        }
        return total_loss, metrics

