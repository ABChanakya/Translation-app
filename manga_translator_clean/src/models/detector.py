"""
YOLO-based text region detector for manga pages.
"""

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from functools import lru_cache

from config.settings import (
    YOLO_MODEL_PATH,
    DEVICE,
    MODEL_CACHE_TTL,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESHOLD,
    ENABLE_CROSS_CLASS_DEDUP,
    CROSS_CLASS_DEDUP_IOU,
)

# Import advanced NMS methods
try:
    from .advanced_nms import apply_nms
    ADVANCED_NMS_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from advanced_nms import apply_nms
        ADVANCED_NMS_AVAILABLE = True
    except ImportError:
        ADVANCED_NMS_AVAILABLE = False
        print("⚠️ Advanced NMS not available, using standard NMS")


@lru_cache(maxsize=1)
def load_yolo_detector():
    """
    Load and cache the YOLO text detection model.
    
    This loads YOUR custom-trained model that detects 5 types of text regions:
    - DIALOGUE (0): Speech bubble outlines
    - SOUND_EFFECTS (1): SFX text
    - SIGNS (2): Background signs/labels
    - TEXT (3): General text
    - REMOVAL (4): Text to be replaced
    
    Returns:
        Loaded YOLO model ready for inference
    """
    print(f"📦 Loading YOLO model from: {YOLO_MODEL_PATH}")
    
    model = YOLO(YOLO_MODEL_PATH)
    model.fuse()  # Optimize for inference
    
    if DEVICE == "cuda":
        model.to("cuda").half()  # Use FP16 on GPU
        print("🚀 YOLO model loaded on GPU with FP16 precision")
    else:
        print("💻 YOLO model loaded on CPU")
    
    print("✅ YOLO model ready!")
    
    return model


class TextDetector:
    """Wrapper class for YOLO-based text detection"""
    
    def __init__(self, confidence: float = DEFAULT_CONFIDENCE,
                 iou_threshold: float = DEFAULT_IOU_THRESHOLD,
                 nms_method: str = 'diou'):
        """
        Initialize text detector
        
        Args:
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            nms_method: NMS method to use: 'standard', 'soft', or 'diou'
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.nms_method = nms_method if ADVANCED_NMS_AVAILABLE else 'standard'
        self.enable_cross_class_dedup = ENABLE_CROSS_CLASS_DEDUP
        self.cross_class_dedup_iou = CROSS_CLASS_DEDUP_IOU
        self.model = load_yolo_detector()
        
        print(f"🎯 Detector initialized with:")
        print(f"   Confidence: {self.confidence}")
        print(f"   IoU Threshold: {self.iou_threshold}")
        print(f"   NMS Method: {self.nms_method}")
        if self.enable_cross_class_dedup:
            print(f"   Cross-class dedup IoU: {self.cross_class_dedup_iou}")
    
    def detect(self, image, apply_advanced_nms: bool = True):
        """
        Detect text regions in an image.
        
        Args:
            image: PIL Image or numpy array
            apply_advanced_nms: Whether to apply advanced NMS post-processing
        
        Returns:
            YOLO detection result object
        """
        print(f"🔍 Running detection with conf={self.confidence}, iou={self.iou_threshold}")
        
        # Run YOLO detection with lower IoU to get more boxes
        # We'll apply advanced NMS afterwards
        detection_iou = self.iou_threshold if not apply_advanced_nms else 0.3
        
        # IMPORTANT: max_det controls maximum detections per image
        # agnostic_nms=False means NMS is applied per-class (better for manga)
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=detection_iou,
            max_det=300,  # Allow up to 300 detections (default is 300)
            agnostic_nms=False,  # Apply NMS per class, not across all classes
            verbose=False
        )
        
        result = results[0]
        
        # Apply advanced NMS if enabled and available
        if apply_advanced_nms and ADVANCED_NMS_AVAILABLE and self.nms_method != 'standard':
            result = self._apply_advanced_nms(result)

        if self.enable_cross_class_dedup:
            result = self._apply_cross_class_dedup(result)
        
        # Log final detection count
        total_detections = len(result.boxes)
        print(f"   📊 Final detections: {total_detections} (method: {self.nms_method})")
        
        return result

    def _apply_cross_class_dedup(self, result):
        """
        Remove near-identical overlapping boxes across classes.

        This keeps one best box for highly-overlapping duplicates that often appear
        as Dialogue/Text/Signs for the same region.
        """
        if len(result.boxes) <= 1:
            return result

        boxes = result.boxes.xyxy
        scores = result.boxes.conf
        classes = result.boxes.cls.long()
        device = boxes.device

        # Slight preference when scores are near-ties.
        # Dialogue > Text > Signs > Sound Effects > Removal
        class_bonus = {
            0: 0.030,
            3: 0.020,
            2: 0.010,
            1: 0.000,
            4: -0.010,
        }
        bonus = torch.tensor(
            [class_bonus.get(int(c.item()), 0.0) for c in classes],
            device=device,
            dtype=scores.dtype,
        )
        ranking_scores = scores + bonus
        order = torch.argsort(ranking_scores, descending=True)

        keep = []
        suppressed = torch.zeros(len(order), dtype=torch.bool, device=device)

        ordered_boxes = boxes[order]

        def _pair_iou(box: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
            x1 = torch.maximum(box[0], others[:, 0])
            y1 = torch.maximum(box[1], others[:, 1])
            x2 = torch.minimum(box[2], others[:, 2])
            y2 = torch.minimum(box[3], others[:, 3])
            inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            area_a = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
            area_b = (others[:, 2] - others[:, 0]).clamp(min=0) * (others[:, 3] - others[:, 1]).clamp(min=0)
            union = area_a + area_b - inter
            return inter / union.clamp(min=1e-6)

        for i in range(len(order)):
            if suppressed[i]:
                continue
            keep.append(order[i])
            if i == len(order) - 1:
                continue
            if suppressed[i + 1 :].all():
                continue

            cur_box = ordered_boxes[i]
            rest_boxes = ordered_boxes[i + 1 :]
            ious = _pair_iou(cur_box, rest_boxes)
            overlap_mask = ious >= self.cross_class_dedup_iou
            suppressed[i + 1 :] |= overlap_mask

        if len(keep) == len(boxes):
            return result

        keep = torch.stack(keep)
        deduped_boxes = boxes[keep]
        deduped_scores = scores[keep]
        deduped_classes = result.boxes.cls[keep]

        new_boxes_data = torch.cat([
            deduped_boxes,
            deduped_scores.unsqueeze(1),
            deduped_classes.unsqueeze(1),
        ], dim=1)
        result.boxes = Boxes(new_boxes_data, result.orig_shape)
        return result
    
    def _apply_advanced_nms(self, result):
        """
        Apply advanced NMS to YOLO results
        
        Args:
            result: YOLO result object
            
        Returns:
            Modified result with advanced NMS applied
        """
        if len(result.boxes) == 0:
            return result
        
        # Extract boxes, scores, and classes
        boxes = result.boxes.xyxy  # (N, 4) [x1, y1, x2, y2]
        scores = result.boxes.conf  # (N,)
        classes = result.boxes.cls  # (N,)
        
        # Apply advanced NMS
        if self.nms_method == 'soft':
            kept_boxes, kept_scores, kept_classes = apply_nms(
                boxes, scores, classes,
                iou_threshold=self.iou_threshold,
                method='soft',
                class_agnostic=False,
                sigma=0.5,
                score_threshold=self.confidence,
                decay_method='gaussian'
            )
        elif self.nms_method == 'diou':
            kept_boxes, kept_scores, kept_classes = apply_nms(
                boxes, scores, classes,
                iou_threshold=self.iou_threshold,
                method='diou',
                class_agnostic=False
            )
        else:
            return result
        
        # Create new Boxes object with filtered detections
        # We need to reconstruct the boxes tensor in the format [xyxy, conf, cls]
        if kept_boxes.numel() > 0:
            # Combine boxes, scores, and classes
            new_boxes_data = torch.cat([
                kept_boxes,  # xyxy coordinates
                kept_scores.unsqueeze(1),  # confidence scores
                kept_classes.unsqueeze(1)  # class IDs
            ], dim=1)
            
            # Create new Boxes object
            result.boxes = Boxes(new_boxes_data, result.orig_shape)
        else:
            # No boxes kept, create empty Boxes
            empty_data = torch.zeros((0, 6), device=boxes.device)
            result.boxes = Boxes(empty_data, result.orig_shape)
        
        return result
