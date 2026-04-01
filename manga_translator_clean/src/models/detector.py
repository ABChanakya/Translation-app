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
    DEFAULT_IOU_THRESHOLD
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
        self.model = load_yolo_detector()
        
        print(f"🎯 Detector initialized with:")
        print(f"   Confidence: {self.confidence}")
        print(f"   IoU Threshold: {self.iou_threshold}")
        print(f"   NMS Method: {self.nms_method}")
    
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
        
        # Log final detection count
        total_detections = len(result.boxes)
        print(f"   📊 Final detections: {total_detections} (method: {self.nms_method})")
        
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
