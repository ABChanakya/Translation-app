"""
Image processing utilities.
"""

import numpy as np
from PIL import Image
from typing import Tuple


def find_whitest_pixel(image_array: np.ndarray) -> Tuple[int, int, int]:
    """
    Find the brightest (whitest) pixel in an image.
    
    Used to find a good background color for text removal.
    
    Args:
        image_array: NumPy array of shape (H, W, 3)
    
    Returns:
        RGB tuple of the whitest pixel
    """
    if image_array.ndim != 3 or image_array.shape[-1] != 3:
        raise ValueError("Image must have 3 color channels (RGB)")
    
    brightness = image_array.sum(axis=-1)
    brightest_idx = np.argmax(brightness)
    
    height, width, _ = image_array.shape
    y, x = divmod(brightest_idx, width)
    
    return tuple(map(int, image_array[y, x]))


def calculate_median_color(image_array: np.ndarray) -> Tuple[int, int, int]:
    """
    Calculate the median color of an image region.
    
    Args:
        image_array: NumPy array of shape (H, W, 3)
    
    Returns:
        RGB tuple of the median color
    """
    median = np.median(image_array.reshape(-1, 3), axis=0)
    return tuple(int(x) for x in median)


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """
    Resize image if it exceeds max dimension.
    
    Args:
        image: PIL Image
        max_size: Maximum dimension (width or height)
    
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    
    if max(width, height) <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.LANCZOS)
