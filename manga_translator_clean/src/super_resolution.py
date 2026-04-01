"""
Real-ESRGAN Super-Resolution Module
Upscales low-resolution manga pages for better OCR accuracy
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import torch

# Try to import RealESRGAN
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    print("⚠️  Real-ESRGAN not installed. Install with: pip install realesrgan basicsr")


class SuperResolutionUpscaler:
    """
    Upscales images using Real-ESRGAN for improved OCR on low-resolution pages
    """
    
    def __init__(
        self,
        model_name: str = "RealESRGAN_x4plus_anime_6B",
        device: Optional[str] = None,
        tile_size: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        half: bool = False
    ):
        """
        Initialize Real-ESRGAN upscaler
        
        Args:
            model_name: Model to use ('RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealESRNet_x4plus')
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
            tile_size: Tile size for processing large images (0 = no tiling)
            tile_pad: Tile padding
            pre_pad: Pre-padding
            half: Use FP16 precision (faster on GPU)
        """
        if not REALESRGAN_AVAILABLE:
            raise ImportError(
                "Real-ESRGAN not available. Install with:\n"
                "pip install realesrgan basicsr"
            )
        
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Disable FP16 on CPU
        if device == 'cpu':
            half = False
        
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.half = half
        
        # Model configurations
        self.model_configs = {
            'RealESRGAN_x4plus': {
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                'scale': 4,
                'num_block': 23,
                'num_feat': 64
            },
            'RealESRGAN_x4plus_anime_6B': {
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
                'scale': 4,
                'num_block': 6,
                'num_feat': 64
            },
            'RealESRNet_x4plus': {
                'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
                'scale': 4,
                'num_block': 23,
                'num_feat': 64
            }
        }
        
        self.upsampler = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Real-ESRGAN model"""
        if self.model_name not in self.model_configs:
            raise ValueError(
                f"Unknown model: {self.model_name}. "
                f"Available: {list(self.model_configs.keys())}"
            )
        
        config = self.model_configs[self.model_name]
        
        # Create model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=config['num_feat'],
            num_block=config['num_block'],
            num_grow_ch=32,
            scale=config['scale']
        )
        
        # Download model weights if needed
        model_path = self._get_model_path(self.model_name, config['url'])
        
        # Create upsampler
        try:
            self.upsampler = RealESRGANer(
                scale=config['scale'],
                model_path=model_path,
                model=model,
                tile=self.tile_size,
                tile_pad=self.tile_pad,
                pre_pad=self.pre_pad,
                half=self.half,
                device=self.device
            )
            print(f"✅ Real-ESRGAN loaded: {self.model_name} on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load Real-ESRGAN: {e}")
            raise
    
    def _get_model_path(self, model_name: str, url: str) -> str:
        """Get or download model weights"""
        # Store models in ~/.cache/realesrgan
        cache_dir = Path.home() / '.cache' / 'realesrgan'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = cache_dir / f"{model_name}.pth"
        
        if not model_path.exists():
            print(f"📥 Downloading {model_name}...")
            import urllib.request
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"✅ Downloaded to {model_path}")
            except Exception as e:
                print(f"❌ Download failed: {e}")
                raise
        
        return str(model_path)
    
    def upscale(
        self,
        image: np.ndarray,
        outscale: Optional[float] = None
    ) -> np.ndarray:
        """
        Upscale an image
        
        Args:
            image: Input image (BGR format, numpy array)
            outscale: Output scale factor. If None, uses model's default scale.
            
        Returns:
            Upscaled image (BGR format, numpy array)
        """
        if self.upsampler is None:
            raise RuntimeError("Model not initialized")
        
        try:
            # Upscale
            output, _ = self.upsampler.enhance(image, outscale=outscale)
            return output
        except Exception as e:
            print(f"❌ Upscaling failed: {e}")
            # Return original image as fallback
            return image
    
    def upscale_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        outscale: Optional[float] = None
    ) -> str:
        """
        Upscale an image file
        
        Args:
            input_path: Path to input image
            output_path: Path to save output. If None, adds suffix to input name.
            outscale: Output scale factor
            
        Returns:
            Path to output file
        """
        # Read image
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {input_path}")
        
        # Upscale
        output = self.upscale(img, outscale=outscale)
        
        # Determine output path
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(
                input_path_obj.parent / 
                f"{input_path_obj.stem}_upscaled{input_path_obj.suffix}"
            )
        
        # Save
        cv2.imwrite(output_path, output)
        
        return output_path
    
    def should_upscale(
        self,
        image: np.ndarray,
        min_resolution: Tuple[int, int] = (1000, 1000)
    ) -> bool:
        """
        Determine if an image should be upscaled based on resolution
        
        Args:
            image: Input image
            min_resolution: Minimum (width, height) to skip upscaling
            
        Returns:
            True if image should be upscaled
        """
        height, width = image.shape[:2]
        return width < min_resolution[0] or height < min_resolution[1]


class AdaptiveUpscaler:
    """
    Adaptive upscaler that chooses the best strategy based on image characteristics
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize adaptive upscaler
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.esrgan_upscaler = None
        
        # Only initialize if Real-ESRGAN is available
        if REALESRGAN_AVAILABLE:
            try:
                self.esrgan_upscaler = SuperResolutionUpscaler(
                    model_name='RealESRGAN_x4plus_anime_6B',
                    device=self.device,
                    tile_size=256 if self.device == 'cuda' else 0,  # Use tiling on GPU
                    half=True if self.device == 'cuda' else False
                )
            except Exception as e:
                print(f"⚠️  Could not initialize Real-ESRGAN: {e}")
    
    def upscale_for_ocr(
        self,
        image: np.ndarray,
        target_height: int = 2000,
        max_upscale: float = 4.0
    ) -> np.ndarray:
        """
        Upscale image optimally for OCR
        
        Args:
            image: Input image
            target_height: Target height for OCR (default 2000px)
            max_upscale: Maximum upscale factor
            
        Returns:
            Upscaled image
        """
        height, width = image.shape[:2]
        
        # Calculate required scale
        scale_needed = target_height / height
        
        # Limit scale
        scale_needed = min(scale_needed, max_upscale)
        
        # If scale is small, use simple interpolation
        if scale_needed <= 1.5:
            new_width = int(width * scale_needed)
            new_height = int(height * scale_needed)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # For larger scales, use Real-ESRGAN if available
        if self.esrgan_upscaler:
            try:
                # Real-ESRGAN works in steps of 4x
                if scale_needed <= 4.0:
                    return self.esrgan_upscaler.upscale(image, outscale=scale_needed)
                else:
                    # First upscale with Real-ESRGAN, then simple resize
                    upscaled = self.esrgan_upscaler.upscale(image, outscale=4.0)
                    remaining_scale = scale_needed / 4.0
                    new_height = int(upscaled.shape[0] * remaining_scale)
                    new_width = int(upscaled.shape[1] * remaining_scale)
                    return cv2.resize(upscaled, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            except Exception as e:
                print(f"⚠️  Real-ESRGAN failed, falling back to interpolation: {e}")
        
        # Fallback: simple interpolation
        new_width = int(width * scale_needed)
        new_height = int(height * scale_needed)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    def preprocess_for_detection(
        self,
        image: np.ndarray,
        min_size: int = 640
    ) -> np.ndarray:
        """
        Preprocess image for object detection
        
        Args:
            image: Input image
            min_size: Minimum dimension size
            
        Returns:
            Preprocessed image
        """
        height, width = image.shape[:2]
        min_dim = min(height, width)
        
        if min_dim < min_size:
            scale = min_size / min_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return image
