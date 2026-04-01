#!/usr/bin/env python3
"""
Model check script - verifies your custom model is available
"""

import os
import sys
from pathlib import Path

def check_custom_model():
    """Check if your custom model is available"""
    print("🎯 Checking for your custom YOLO model...")
    
    # Check main model location
    main_model = Path("yolo_train_run/full_finetune_60_20/weights/best.pt")
    backup_model = Path("models/checkpoints/custom_yolo_best.pt")
    
    if main_model.exists():
        size_mb = main_model.stat().st_size / (1024 * 1024)
        print(f"✅ Custom model found: {main_model} ({size_mb:.1f} MB)")
        return True
    elif backup_model.exists():
        size_mb = backup_model.stat().st_size / (1024 * 1024)
        print(f"✅ Backup custom model found: {backup_model} ({size_mb:.1f} MB)")
        return True
    else:
        print("❌ Custom YOLO model not found!")
        print("   Run: ./copy_my_models.sh")
        print("   Or manually copy your model to:")
        print(f"   {main_model}")
        return False
    
def check_other_models():
    """Check other models but don't auto-download"""
    print("\n📖 Note: Other models (manga-ocr, Gemma) will auto-download when first used")
    print("   manga-ocr: Downloads automatically on first import")
    print("   Gemma: Downloads from HuggingFace when first used")
    print("   (You may need: huggingface-cli login)")

def download_gemma():
    """Downloads the Gemma model from Hugging Face."""
    print("Downloading Gemma model...")
    try:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("Error: huggingface_hub is not installed in this environment.")
            print("Please activate your virtual environment and run:")
            print("  pip install huggingface_hub")
            return
        
        model_id = "google/gemma-2b-it"
        
        # This will download the model files to a local cache directory
        # and return the path. The application will then load it from there.
        snapshot_download(
            repo_id=model_id,
            local_dir_use_symlinks=False, # Use copies instead of symlinks
            resume_download=True
        )
        
        print(f"✓ Gemma model ({model_id}) downloaded successfully.")
        print("  The model is stored in the Hugging Face cache directory.")

    except ImportError:
        print("Gemma download error: `huggingface_hub` is not installed.")
        print("Please run: pip install huggingface_hub")
    except Exception as e:
        print(f"Gemma download error: {e}")
        print("Note: You may need to login to HuggingFace: huggingface-cli login")

def download_all_models():
    """Downloads all models."""
    print("Downloading all models...")
    download_gemma()
    print("All models downloaded successfully.")

if __name__ == "__main__":
    found = check_custom_model()
    check_other_models()
    
    if found:
        print("\n✅ Ready to go!")
    else:
        print("\n❌ Setup incomplete - custom model needed")
        sys.exit(1)
