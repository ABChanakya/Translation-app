"""
Advanced Non-Maximum Suppression (NMS) implementations
Provides Soft-NMS and DIoU-NMS for better handling of overlapping text regions
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: Tensor of shape (N, 4) in format [x1, y1, x2, y2]
        boxes2: Tensor of shape (M, 4) in format [x1, y1, x2, y2]
        
    Returns:
        IoU matrix of shape (N, M)
    """
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # Compute union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    union = area1[:, None] + area2 - inter  # (N, M)
    
    iou = inter / union.clamp(min=1e-6)
    return iou


def compute_diou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Distance-IoU between two sets of boxes
    DIoU = IoU - (d^2 / c^2)
    where d is center distance and c is diagonal length of smallest enclosing box
    
    Args:
        boxes1: Tensor of shape (N, 4) in format [x1, y1, x2, y2]
        boxes2: Tensor of shape (M, 4) in format [x1, y1, x2, y2]
        
    Returns:
        DIoU matrix of shape (N, M)
    """
    # Compute standard IoU
    iou = compute_iou(boxes1, boxes2)  # (N, M)
    
    # Compute center points
    center1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2  # (N, 2)
    center2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2  # (M, 2)
    
    # Center distance squared
    center_dist2 = torch.sum((center1[:, None, :] - center2) ** 2, dim=2)  # (N, M)
    
    # Smallest enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    # Diagonal length squared
    wh = (rb - lt).clamp(min=0)
    diag2 = torch.sum(wh ** 2, dim=2)  # (N, M)
    
    # DIoU = IoU - (center_distance^2 / diagonal^2)
    diou = iou - (center_dist2 / diag2.clamp(min=1e-6))
    
    return diou


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    method: str = 'gaussian'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft-NMS: Improving Object Detection with One Line of Code
    Paper: https://arxiv.org/abs/1704.04503
    
    Instead of eliminating overlapping boxes, Soft-NMS reduces their scores
    based on overlap, allowing better handling of densely packed objects.
    
    Args:
        boxes: Tensor of shape (N, 4) in format [x1, y1, x2, y2]
        scores: Tensor of shape (N,) with confidence scores
        iou_threshold: IoU threshold for score decay
        sigma: Gaussian smoothing parameter (for gaussian method)
        score_threshold: Minimum score to keep a box
        method: 'linear' or 'gaussian' score decay function
        
    Returns:
        Tuple of (kept_boxes, kept_scores) after Soft-NMS
    """
    device = boxes.device
    N = boxes.shape[0]
    
    if N == 0:
        return boxes, scores
    
    # Work with copies
    boxes = boxes.clone()
    scores = scores.clone()
    
    # Track which boxes to keep
    kept_indices = []
    kept_scores = []
    
    # Process in descending score order
    for _ in range(N):
        # Find box with highest score
        idx = scores.argmax()
        max_score = scores[idx]
        
        if max_score < score_threshold:
            break
        
        kept_indices.append(idx.item())
        kept_scores.append(max_score)
        max_box = boxes[idx:idx+1]  # (1, 4)
        
        # Compute IoU with remaining boxes
        if idx < N - 1:
            remaining_boxes = boxes[idx+1:]
            ious = compute_iou(max_box, remaining_boxes).squeeze(0)  # (N-idx-1,)
            
            # Apply score decay based on IoU
            if method == 'linear':
                # Linear decay: s = s * (1 - iou) if iou > threshold
                weights = torch.ones_like(ious)
                weights[ious > iou_threshold] = 1 - ious[ious > iou_threshold]
                scores[idx+1:] = scores[idx+1:] * weights
                
            elif method == 'gaussian':
                # Gaussian decay: s = s * exp(-(iou^2) / sigma)
                weights = torch.exp(-(ious ** 2) / sigma)
                scores[idx+1:] = scores[idx+1:] * weights
            
            else:
                raise ValueError(f"Unknown method: {method}. Use 'linear' or 'gaussian'")
        
        # Mark processed box with very low score
        scores[idx] = -1
    
    # Return kept boxes
    kept_indices = torch.tensor(kept_indices, dtype=torch.long, device=device)
    return boxes[kept_indices], torch.tensor(kept_scores, dtype=scores.dtype, device=device)


def diou_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    DIoU-NMS: Distance-IoU based Non-Maximum Suppression
    Paper: https://arxiv.org/abs/1911.08287
    
    Uses DIoU instead of IoU for better handling of boxes at different scales
    and overlapping patterns. Better for manga where bubbles can nest or overlap.
    
    Args:
        boxes: Tensor of shape (N, 4) in format [x1, y1, x2, y2]
        scores: Tensor of shape (N,) with confidence scores
        iou_threshold: DIoU threshold for suppression
        
    Returns:
        Tensor of indices of kept boxes
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    # Sort by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    while sorted_indices.numel() > 0:
        # Keep box with highest score
        idx = sorted_indices[0]
        keep.append(idx)
        
        if sorted_indices.numel() == 1:
            break
        
        # Compute DIoU with remaining boxes
        max_box = boxes[idx:idx+1]  # (1, 4)
        remaining_boxes = boxes[sorted_indices[1:]]  # (N-1, 4)
        
        dious = compute_diou(max_box, remaining_boxes).squeeze(0)  # (N-1,)
        
        # Keep boxes with DIoU below threshold
        mask = dious <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: Optional[torch.Tensor] = None,
    iou_threshold: float = 0.5,
    method: str = 'standard',
    class_agnostic: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply NMS with configurable method
    
    Args:
        boxes: Tensor of shape (N, 4) in format [x1, y1, x2, y2]
        scores: Tensor of shape (N,) with confidence scores
        classes: Optional tensor of shape (N,) with class IDs
        iou_threshold: IoU/DIoU threshold
        method: 'standard', 'soft', or 'diou'
        class_agnostic: If True, apply NMS across all classes
        **kwargs: Additional method-specific parameters
                  For soft-NMS: sigma, score_threshold, decay_method
        
    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_classes)
    """
    if boxes.numel() == 0:
        empty = torch.empty((0,), dtype=torch.int64, device=boxes.device)
        return boxes, scores, classes if classes is not None else None
    
    # Class-agnostic NMS: apply to all boxes together
    if class_agnostic or classes is None:
        if method == 'soft':
            sigma = kwargs.get('sigma', 0.5)
            score_threshold = kwargs.get('score_threshold', 0.001)
            decay_method = kwargs.get('decay_method', 'gaussian')
            
            kept_boxes, kept_scores = soft_nms(
                boxes, scores,
                iou_threshold=iou_threshold,
                sigma=sigma,
                score_threshold=score_threshold,
                method=decay_method
            )
            kept_classes = classes if classes is None else classes[:len(kept_boxes)]
            
        elif method == 'diou':
            keep_indices = diou_nms(boxes, scores, iou_threshold=iou_threshold)
            kept_boxes = boxes[keep_indices]
            kept_scores = scores[keep_indices]
            kept_classes = classes[keep_indices] if classes is not None else None
            
        else:  # standard
            keep_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
            kept_boxes = boxes[keep_indices]
            kept_scores = scores[keep_indices]
            kept_classes = classes[keep_indices] if classes is not None else None
        
        return kept_boxes, kept_scores, kept_classes
    
    # Per-class NMS: apply separately for each class
    unique_classes = torch.unique(classes)
    all_kept_boxes = []
    all_kept_scores = []
    all_kept_classes = []
    
    for cls in unique_classes:
        mask = classes == cls
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        
        if method == 'soft':
            sigma = kwargs.get('sigma', 0.5)
            score_threshold = kwargs.get('score_threshold', 0.001)
            decay_method = kwargs.get('decay_method', 'gaussian')
            
            cls_kept_boxes, cls_kept_scores = soft_nms(
                cls_boxes, cls_scores,
                iou_threshold=iou_threshold,
                sigma=sigma,
                score_threshold=score_threshold,
                method=decay_method
            )
            
        elif method == 'diou':
            keep_indices = diou_nms(cls_boxes, cls_scores, iou_threshold=iou_threshold)
            cls_kept_boxes = cls_boxes[keep_indices]
            cls_kept_scores = cls_scores[keep_indices]
            
        else:  # standard
            keep_indices = torch.ops.torchvision.nms(cls_boxes, cls_scores, iou_threshold)
            cls_kept_boxes = cls_boxes[keep_indices]
            cls_kept_scores = cls_scores[keep_indices]
        
        all_kept_boxes.append(cls_kept_boxes)
        all_kept_scores.append(cls_kept_scores)
        all_kept_classes.append(torch.full((len(cls_kept_boxes),), cls, device=boxes.device))
    
    # Concatenate results
    kept_boxes = torch.cat(all_kept_boxes, dim=0)
    kept_scores = torch.cat(all_kept_scores, dim=0)
    kept_classes = torch.cat(all_kept_classes, dim=0)
    
    return kept_boxes, kept_scores, kept_classes


# Convenience functions
def soft_nms_linear(boxes, scores, iou_threshold=0.5, score_threshold=0.001):
    """Soft-NMS with linear decay"""
    return soft_nms(boxes, scores, iou_threshold, method='linear', score_threshold=score_threshold)


def soft_nms_gaussian(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """Soft-NMS with Gaussian decay"""
    return soft_nms(boxes, scores, iou_threshold, sigma=sigma, method='gaussian', score_threshold=score_threshold)
