"""
Vertical Text Detection and Handling Module
Detects vertical Japanese text orientation and handles rotation for OCR/rendering
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from PIL import Image
import torch


@dataclass
class TextOrientation:
    """Text orientation information"""
    is_vertical: bool
    confidence: float
    rotation_angle: float  # Degrees to rotate for OCR (0, 90, 180, 270)
    bbox_aspect_ratio: float


class VerticalTextDetector:
    """
    Detects vertical text orientation in manga
    """
    
    def __init__(
        self,
        aspect_ratio_threshold: float = 2.0,
        use_ml_detection: bool = False
    ):
        """
        Initialize vertical text detector
        
        Args:
            aspect_ratio_threshold: Height/width ratio above which text is likely vertical
            use_ml_detection: Use ML-based orientation detection (more accurate but slower)
        """
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.use_ml_detection = use_ml_detection
    
    def detect_orientation(
        self,
        bbox: Tuple[int, int, int, int],
        image: Optional[np.ndarray] = None
    ) -> TextOrientation:
        """
        Detect text orientation from bounding box
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            image: Optional image crop for ML-based detection
            
        Returns:
            TextOrientation with detection results
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Avoid division by zero
        if width < 1:
            width = 1
        
        aspect_ratio = height / width
        
        # Simple aspect ratio heuristic
        # Japanese vertical text typically has height >> width
        is_vertical_by_ar = aspect_ratio > self.aspect_ratio_threshold
        confidence = min(aspect_ratio / self.aspect_ratio_threshold, 1.0)
        
        # ML-based detection if enabled and image provided
        if self.use_ml_detection and image is not None:
            is_vertical_ml, ml_confidence = self._detect_with_ml(image)
            # Combine heuristics
            is_vertical = is_vertical_ml
            confidence = ml_confidence
        else:
            is_vertical = is_vertical_by_ar
        
        # Determine rotation angle
        # Vertical text in manga is typically top-to-bottom, right-to-left
        rotation_angle = 0 if not is_vertical else 90
        
        return TextOrientation(
            is_vertical=is_vertical,
            confidence=confidence,
            rotation_angle=rotation_angle,
            bbox_aspect_ratio=aspect_ratio
        )
    
    def _detect_with_ml(
        self,
        image_crop: np.ndarray
    ) -> Tuple[bool, float]:
        """
        ML-based orientation detection using edge analysis
        
        Args:
            image_crop: Cropped image region
            
        Returns:
            Tuple of (is_vertical, confidence)
        """
        # Convert to grayscale if needed
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Analyze edge orientations using Sobel
        sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate dominant orientation
        horizontal_edges = np.sum(np.abs(sobelx))
        vertical_edges = np.sum(np.abs(sobely))
        
        # Vertical text has more horizontal edges (character strokes are horizontal)
        if vertical_edges == 0:
            return False, 0.5
        
        ratio = horizontal_edges / vertical_edges
        
        # Vertical text typically has ratio > 1.0
        is_vertical = ratio > 1.0
        confidence = min(ratio, 2.0) / 2.0 if is_vertical else min(1.0 / ratio, 2.0) / 2.0
        
        return is_vertical, confidence
    
    def batch_detect(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        image: Optional[np.ndarray] = None
    ) -> List[TextOrientation]:
        """
        Detect orientation for multiple bboxes
        
        Args:
            bboxes: List of bounding boxes
            image: Optional full image for ML detection
            
        Returns:
            List of TextOrientation objects
        """
        orientations = []
        
        for bbox in bboxes:
            # Extract crop if ML detection enabled
            crop = None
            if self.use_ml_detection and image is not None:
                x1, y1, x2, y2 = bbox
                crop = image[y1:y2, x1:x2]
            
            orientation = self.detect_orientation(bbox, crop)
            orientations.append(orientation)
        
        return orientations


class VerticalTextRotator:
    """
    Handles rotation of vertical text regions for OCR and rendering
    """
    
    @staticmethod
    def rotate_for_ocr(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        rotation_angle: float
    ) -> np.ndarray:
        """
        Rotate text region for horizontal OCR processing
        
        Args:
            image: Full image
            bbox: Bounding box of text region
            rotation_angle: Rotation angle in degrees (90, 180, 270)
            
        Returns:
            Rotated crop suitable for OCR
        """
        x1, y1, x2, y2 = bbox
        
        # Extract crop
        crop = image[y1:y2, x1:x2].copy()
        
        # Rotate crop
        if rotation_angle == 90:
            rotated = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_angle == 180:
            rotated = cv2.rotate(crop, cv2.ROTATE_180)
        elif rotation_angle == 270:
            rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        else:
            rotated = crop
        
        return rotated
    
    @staticmethod
    def rotate_bbox(
        bbox: Tuple[int, int, int, int],
        rotation_angle: float,
        image_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Rotate bounding box coordinates
        
        Args:
            bbox: Original bounding box (x1, y1, x2, y2)
            rotation_angle: Rotation angle in degrees
            image_shape: Image shape (height, width)
            
        Returns:
            Rotated bounding box
        """
        x1, y1, x2, y2 = bbox
        h, w = image_shape
        
        if rotation_angle == 90:
            # 90° CCW: (x, y) -> (y, w - x)
            new_x1, new_y1 = y1, w - x2
            new_x2, new_y2 = y2, w - x1
        elif rotation_angle == 180:
            # 180°: (x, y) -> (w - x, h - y)
            new_x1, new_y1 = w - x2, h - y2
            new_x2, new_y2 = w - x1, h - y1
        elif rotation_angle == 270:
            # 90° CW: (x, y) -> (h - y, x)
            new_x1, new_y1 = h - y2, x1
            new_x2, new_y2 = h - y1, x2
        else:
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
        
        # Ensure x1 < x2 and y1 < y2
        return (
            min(new_x1, new_x2),
            min(new_y1, new_y2),
            max(new_x1, new_x2),
            max(new_y1, new_y2)
        )


class VerticalTextRenderer:
    """
    Renders translated text vertically for manga
    """
    
    def __init__(
        self,
        font_path: Optional[str] = None,
        default_font_size: int = 24
    ):
        """
        Initialize vertical text renderer
        
        Args:
            font_path: Path to TrueType font file (Japanese font recommended)
            default_font_size: Default font size
        """
        self.font_path = font_path
        self.default_font_size = default_font_size
    
    def render_vertical_text(
        self,
        text: str,
        bbox: Tuple[int, int, int, int],
        font_size: Optional[int] = None,
        color: Tuple[int, int, int] = (0, 0, 0),
        background_color: Optional[Tuple[int, int, int]] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Render text vertically
        
        Args:
            text: Text to render
            bbox: Target bounding box (x1, y1, x2, y2)
            font_size: Font size (uses default if None)
            color: Text color (BGR)
            background_color: Background color (None for transparent)
            
        Returns:
            Rendered text image
        """
        from PIL import Image, ImageDraw, ImageFont
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if font_size is None:
            font_size = self.default_font_size
        
        # Create vertical image (swap width/height)
        img = Image.new('RGB', (height, width), background_color or (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Draw text vertically (character by character)
        y_offset = 10
        for char in text:
            # Get character size
            bbox = draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            
            # Center horizontally, stack vertically
            x_pos = (height - char_width) // 2
            draw.text((x_pos, y_offset), char, fill=color, font=font)
            y_offset += char_height + 5
        
        # Rotate 90° clockwise for final vertical orientation
        img_rotated = img.rotate(-90, expand=True)
        
        return np.array(img_rotated)
    
    def fit_text_to_bbox(
        self,
        text: str,
        bbox: Tuple[int, int, int, int],
        max_font_size: int = 36,
        min_font_size: int = 12
    ) -> int:
        """
        Calculate optimal font size to fit text in bbox
        
        Args:
            text: Text to fit
            bbox: Target bounding box
            max_font_size: Maximum font size
            min_font_size: Minimum font size
            
        Returns:
            Optimal font size
        """
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        
        # Estimate: each character takes ~font_size pixels + spacing
        estimated_height = len(text) * max_font_size * 1.2
        
        if estimated_height <= bbox_height:
            return max_font_size
        
        # Scale down
        optimal_size = int((bbox_height / estimated_height) * max_font_size)
        return max(min_font_size, optimal_size)


# Integration helper
class VerticalTextHandler:
    """
    Complete vertical text handling pipeline
    """
    
    def __init__(
        self,
        detector: Optional[VerticalTextDetector] = None,
        renderer: Optional[VerticalTextRenderer] = None
    ):
        """
        Initialize handler
        
        Args:
            detector: Vertical text detector (creates default if None)
            renderer: Vertical text renderer (creates default if None)
        """
        self.detector = detector or VerticalTextDetector()
        self.renderer = renderer or VerticalTextRenderer()
        self.rotator = VerticalTextRotator()
    
    def process_region(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        ocr_func,
        translate_func,
        render_horizontal: bool = False
    ) -> Tuple[str, np.ndarray, TextOrientation]:
        """
        Complete pipeline: detect orientation, rotate for OCR, translate, render
        
        Args:
            image: Full image
            bbox: Text region bounding box
            ocr_func: OCR function (takes image crop, returns text)
            translate_func: Translation function (takes text, returns translation)
            render_horizontal: Force horizontal rendering even for vertical text
            
        Returns:
            Tuple of (translated_text, rendered_image, orientation_info)
        """
        # Detect orientation
        orientation = self.detector.detect_orientation(bbox, image)
        
        # Rotate for OCR if vertical
        if orientation.is_vertical and orientation.rotation_angle > 0:
            ocr_crop = self.rotator.rotate_for_ocr(
                image, bbox, orientation.rotation_angle
            )
        else:
            x1, y1, x2, y2 = bbox
            ocr_crop = image[y1:y2, x1:x2]
        
        # Run OCR
        text = ocr_func(ocr_crop)
        
        # Translate
        translated = translate_func(text)
        
        # Render
        if orientation.is_vertical and not render_horizontal:
            rendered = self.renderer.render_vertical_text(translated, bbox)
        else:
            # Use horizontal rendering (implement separately or use existing)
            rendered = self._render_horizontal(translated, bbox)
        
        return translated, rendered, orientation
    
    def _render_horizontal(
        self,
        text: str,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Fallback horizontal text rendering"""
        from PIL import Image, ImageDraw, ImageFont
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        font = ImageFont.load_default()
        draw.text((10, height // 2), text, fill=(0, 0, 0), font=font)
        
        return np.array(img)
