#!/usr/bin/env python3
"""
Model Download Script for Manga Translation Developer Kit
Downloads required models: Gemma, manga-ocr, and ensures YOLO models are accessible
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import hashlib

def download_file(url, dest_path, expected_size=None):
    """Download a file with progress bar"""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        print(f"✓ {dest_path.name} already exists")
        return
    
    print(f"Downloading {dest_path.name}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    if expected_size and total_size != expected_size:
        print(f"Warning: Expected {expected_size} bytes, got {total_size}")
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"✓ Downloaded {dest_path.name}")

def setup_huggingface_models():
    """Setup HuggingFace models (Gemma, manga-ocr)"""
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import manga_ocr
    except ImportError as e:
        print(f"Error: Missing packages. Install with:")
        print("pip install huggingface-hub transformers manga-ocr")
        return False
    
    print("\n" + "="*60)
    print("Setting up HuggingFace Models")
    print("="*60)
    
    # Download Gemma model (choose smaller version for faster download)
    models_to_download = [
        "google/gemma-2b-it",  # Smaller, faster
        # "google/gemma-7b-it",  # Larger, better quality - uncomment if you want this
    ]
    
    for model_name in models_to_download:
        print(f"\nDownloading {model_name}...")
        try:
            # Download model files
            snapshot_download(
                repo_id=model_name,
                cache_dir="./models/huggingface",
                local_files_only=False
            )
            print(f"✓ {model_name} downloaded successfully")
            
            # Test loading
            print(f"Testing {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"✓ {model_name} tokenizer loaded successfully")
            
        except Exception as e:
            print(f"✗ Error downloading {model_name}: {e}")
            print("You may need to login to HuggingFace: huggingface-cli login")
    
    # Setup manga-ocr
    print(f"\nSetting up manga-ocr...")
    try:
        mocr = manga_ocr.MangaOcr()
        print("✓ manga-ocr initialized successfully")
    except Exception as e:
        print(f"✗ Error setting up manga-ocr: {e}")
    
    return True

def setup_yolo_models():
    """Check for your custom YOLO model"""
    print("\n" + "="*60)
    print("Checking Custom YOLO Model")
    print("="*60)
    
    # Check for your custom trained model
    custom_model_path = Path("yolo_train_run/full_finetune_60_20/weights/best.pt")
    
    if custom_model_path.exists():
        size_mb = custom_model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Custom YOLO model: {custom_model_path} ({size_mb:.1f} MB)")
        print("  This is your custom-trained model with 5 text region classes:")
        print("  - DIALOGUE, SOUND_EFFECTS, SIGNS, TEXT, REMOVAL")
    else:
        print(f"✗ Custom YOLO model not found: {custom_model_path}")
        print("  Run copy_my_models.sh to copy from the original workspace")
        
        # Check for backup custom model
        backup_path = Path("models/checkpoints/custom_yolo_best.pt")
        if backup_path.exists():
            size_mb = backup_path.stat().st_size / (1024 * 1024)
            print(f"✓ Backup custom model found: {backup_path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ Backup custom model not found: {backup_path}")
    
    print("\n💡 Note: This system uses ONLY your custom-trained model.")
    print("   Default YOLO models (yolo11n.pt, yolov8m.pt) are NOT used.")

def setup_colorization_models():
    """Check colorization models"""
    print("\n" + "="*60)
    print("Checking Colorization Models")
    print("="*60)
    
    colorizer_path = Path("colorization/colorizer.pth")
    if colorizer_path.exists():
        size_mb = colorizer_path.stat().st_size / (1024 * 1024)
        print(f"✓ Colorization model: {colorizer_path} ({size_mb:.1f} MB)")
    else:
        print(f"✗ Colorization model not found: {colorizer_path}")
        print("This should be available from the original workspace")

def main():
    print("🤖 Manga Translation Developer Kit - Model Setup")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("training").exists() or not Path("models").exists():
        print("Error: Please run this script from the manga_translator_clean directory")
        sys.exit(1)
    
    # Setup models
    success = True
    
    try:
        success &= setup_huggingface_models()
        setup_yolo_models()
        setup_colorization_models()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        success = False
    
    print("\n" + "="*60)
    if success:
        print("✅ Model setup completed!")
        print("\nNext steps:")
        print("1. Test the web interface: python web/app.py")
        print("2. Try translation: python examples/translator_standalone.py --help")
        print("3. Train a model: python training/advanced_train_yolo.py")
    else:
        print("⚠️ Model setup completed with some errors")
        print("Check the messages above and install missing dependencies")
    
    print("\n💡 Tip: Some models will auto-download when first used")

if __name__ == "__main__":
    main()
