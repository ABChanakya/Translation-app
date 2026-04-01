"""
ONNX/TensorRT Model Optimization Module
Exports and runs YOLO models with ONNX Runtime or TensorRT for faster inference
"""
import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union
import cv2

# Check for ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNXRuntime not installed. Install with: pip install onnxruntime onnxruntime-gpu")

# Check for TensorRT
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    # TensorRT is optional, only needed for maximum speed on NVIDIA GPUs


class ONNXModelExporter:
    """
    Exports PyTorch YOLO models to ONNX format
    """
    
    @staticmethod
    def export_yolo_to_onnx(
        model_path: str,
        output_path: Optional[str] = None,
        imgsz: int = 640,
        simplify: bool = True,
        opset: int = 12
    ) -> str:
        """
        Export YOLO model to ONNX format
        
        Args:
            model_path: Path to PyTorch .pt model
            output_path: Output ONNX path (auto-generated if None)
            imgsz: Input image size for export
            simplify: Apply ONNX simplification
            opset: ONNX opset version
            
        Returns:
            Path to exported ONNX model
        """
        from ultralytics import YOLO
        
        print(f"📦 Loading model from {model_path}")
        model = YOLO(model_path)
        
        # Generate output path if not provided
        if output_path is None:
            model_dir = Path(model_path).parent
            model_name = Path(model_path).stem
            output_path = str(model_dir / f"{model_name}.onnx")
        
        print(f"🔄 Exporting to ONNX format...")
        print(f"   Image size: {imgsz}")
        print(f"   Opset: {opset}")
        print(f"   Simplify: {simplify}")
        
        # Export using Ultralytics built-in export
        export_result = model.export(
            format='onnx',
            imgsz=imgsz,
            simplify=simplify,
            opset=opset
        )
        
        print(f"✅ ONNX model exported to: {export_result}")
        
        return str(export_result)
    
    @staticmethod
    def export_to_tensorrt(
        onnx_path: str,
        output_path: Optional[str] = None,
        fp16: bool = True,
        workspace: int = 4
    ) -> Optional[str]:
        """
        Convert ONNX model to TensorRT engine
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Output TensorRT engine path
            fp16: Use FP16 precision
            workspace: Workspace size in GB
            
        Returns:
            Path to TensorRT engine or None if TensorRT unavailable
        """
        if not TENSORRT_AVAILABLE:
            print("⚠️ TensorRT not available. Install TensorRT for NVIDIA GPUs.")
            return None
        
        from ultralytics import YOLO
        
        # Generate output path
        if output_path is None:
            output_path = str(Path(onnx_path).with_suffix('.engine'))
        
        print(f"🚀 Converting to TensorRT engine...")
        print(f"   Input: {onnx_path}")
        print(f"   Output: {output_path}")
        print(f"   FP16: {fp16}")
        print(f"   Workspace: {workspace}GB")
        
        # Load ONNX model and export to TensorRT
        model = YOLO(onnx_path)
        export_result = model.export(
            format='engine',
            half=fp16,
            workspace=workspace
        )
        
        print(f"✅ TensorRT engine created: {export_result}")
        
        return str(export_result)


class ONNXYOLODetector:
    """
    YOLO detector using ONNX Runtime for faster inference
    """
    
    def __init__(
        self,
        onnx_path: str,
        confidence: float = 0.1,
        iou_threshold: float = 0.55,
        imgsz: int = 640,
        use_gpu: bool = True
    ):
        """
        Initialize ONNX YOLO detector
        
        Args:
            onnx_path: Path to ONNX model file
            confidence: Confidence threshold
            iou_threshold: IoU threshold for NMS
            imgsz: Model input size
            use_gpu: Use GPU if available
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNXRuntime not installed")
        
        self.onnx_path = onnx_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        
        # Create ONNX Runtime session
        print(f"📦 Loading ONNX model: {onnx_path}")
        
        providers = []
        if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
            print("🚀 Using CUDA execution provider (GPU)")
        providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers
        )
        
        # Get model input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"✅ ONNX detector initialized")
        print(f"   Input: {self.input_name}")
        print(f"   Outputs: {self.output_names}")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for ONNX model
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Tuple of (preprocessed_image, original_shape)
        """
        original_shape = image.shape[:2]
        
        # Resize with padding
        img_resized = cv2.resize(image, (self.imgsz, self.imgsz))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and transpose to (C, H, W)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch, original_shape
    
    def postprocess(
        self,
        outputs: List[np.ndarray],
        original_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess ONNX outputs to get boxes, scores, classes
        
        Args:
            outputs: ONNX model outputs
            original_shape: Original image shape (H, W)
            
        Returns:
            Tuple of (boxes, scores, classes) as numpy arrays
        """
        # YOLO output format varies by version
        # Typically: [batch, num_predictions, 5+num_classes]
        # Format: [x_center, y_center, width, height, objectness, class_scores...]
        
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Extract components
        boxes_xywh = predictions[:, :4]
        objectness = predictions[:, 4]
        class_scores = predictions[:, 5:]
        
        # Get class with max score
        class_ids = np.argmax(class_scores, axis=1)
        class_confidences = np.max(class_scores, axis=1)
        
        # Combined confidence
        confidences = objectness * class_confidences
        
        # Filter by confidence
        mask = confidences > self.confidence
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Convert xywh to xyxy
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
        
        # Scale boxes to original image size
        scale_x = original_shape[1] / self.imgsz
        scale_y = original_shape[0] / self.imgsz
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y
        
        # Apply NMS
        keep_indices = self._nms(boxes_xyxy, confidences, self.iou_threshold)
        
        boxes_xyxy = boxes_xyxy[keep_indices]
        confidences = confidences[keep_indices]
        class_ids = class_ids[keep_indices]
        
        return boxes_xyxy, confidences, class_ids
    
    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float
    ) -> np.ndarray:
        """
        Non-Maximum Suppression
        
        Args:
            boxes: Boxes in xyxy format
            scores: Confidence scores
            iou_threshold: IoU threshold
            
        Returns:
            Indices of kept boxes
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
        
        # Sort by scores
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Compute IoU with remaining boxes
            ious = self._compute_iou(boxes[i], boxes[order[1:]])
            
            # Keep boxes with IoU below threshold
            mask = ious <= iou_threshold
            order = order[1:][mask]
        
        return np.array(keep, dtype=np.int32)
    
    def _compute_iou(
        self,
        box: np.ndarray,
        boxes: np.ndarray
    ) -> np.ndarray:
        """Compute IoU between one box and multiple boxes"""
        # Intersection
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Areas
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Union
        union = box_area + boxes_area - intersection
        
        # IoU
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def detect(self, image: np.ndarray):
        """
        Run detection on image
        
        Args:
            image: Input image (BGR numpy array)
            
        Returns:
            Tuple of (boxes, scores, classes)
        """
        # Preprocess
        input_tensor, original_shape = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )
        
        # Postprocess
        boxes, scores, classes = self.postprocess(outputs, original_shape)
        
        print(f"🔍 ONNX Detection: Found {len(boxes)} regions")
        
        return boxes, scores, classes


def benchmark_models(
    pytorch_model_path: str,
    onnx_model_path: Optional[str] = None,
    test_image_path: Optional[str] = None,
    num_runs: int = 100
):
    """
    Benchmark PyTorch vs ONNX inference speed
    
    Args:
        pytorch_model_path: Path to PyTorch .pt model
        onnx_model_path: Path to ONNX model (exports if None)
        test_image_path: Path to test image (generates random if None)
        num_runs: Number of inference runs for averaging
    """
    import time
    from ultralytics import YOLO
    
    print("=" * 60)
    print("BENCHMARK: PyTorch vs ONNX")
    print("=" * 60)
    
    # Export ONNX if needed
    if onnx_model_path is None:
        print("Exporting ONNX model...")
        onnx_model_path = ONNXModelExporter.export_yolo_to_onnx(pytorch_model_path)
    
    # Load test image
    if test_image_path:
        test_image = cv2.imread(test_image_path)
    else:
        # Random image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    print(f"\nTest image shape: {test_image.shape}")
    print(f"Number of runs: {num_runs}\n")
    
    # Benchmark PyTorch
    print("🔥 PyTorch Model:")
    pytorch_model = YOLO(pytorch_model_path)
    
    # Warmup
    _ = pytorch_model.predict(test_image, verbose=False)
    
    start = time.time()
    for _ in range(num_runs):
        _ = pytorch_model.predict(test_image, verbose=False)
    pytorch_time = (time.time() - start) / num_runs
    
    print(f"   Average time: {pytorch_time*1000:.2f} ms")
    print(f"   FPS: {1/pytorch_time:.2f}")
    
    # Benchmark ONNX
    if ONNX_AVAILABLE:
        print("\n⚡ ONNX Model:")
        onnx_detector = ONNXYOLODetector(onnx_model_path)
        
        # Warmup
        _ = onnx_detector.detect(test_image)
        
        start = time.time()
        for _ in range(num_runs):
            _ = onnx_detector.detect(test_image)
        onnx_time = (time.time() - start) / num_runs
        
        print(f"   Average time: {onnx_time*1000:.2f} ms")
        print(f"   FPS: {1/onnx_time:.2f}")
        
        # Speedup
        speedup = pytorch_time / onnx_time
        print(f"\n🚀 ONNX Speedup: {speedup:.2f}x faster")
    else:
        print("\n⚠️ ONNX Runtime not available for benchmarking")
    
    print("=" * 60)
