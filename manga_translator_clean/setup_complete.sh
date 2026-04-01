#!/bin/bash
# Complete setup script for Manga Translation Developer Kit
# Downloads dependencies and all required models

set -e  # Exit on error

echo "🎌 Manga Translation Developer Kit - Complete Setup"
echo "================================================="

# Check if we're in the right directory
if [ ! -d "training" ] || [ ! -d "models" ]; then
    echo "❌ Error: Please run this script from the manga_translator_clean directory"
    exit 1
fi

echo ""
echo "📦 Step 1: Installing Python dependencies..."
echo "================================================="

# Install requirements
if command -v uv &> /dev/null; then
    echo "Using uv for faster installation..."
    uv pip install -r requirements-minimal.txt
else
    echo "Using pip..."
    pip install -r requirements-minimal.txt
fi

echo ""
echo "🔄 Step 2: Copying your custom trained models..."
echo "================================================="

# Define source paths for your custom models
OLD_WORKSPACE="/home/chanakya/chanakya/UNI/translation_tool"
CUSTOM_MODELS_DIR="./models/custom"

# Create directory for custom models
mkdir -p "$CUSTOM_MODELS_DIR"

echo "Copying your custom trained YOLO models..."

# Copy your best custom models from training runs
if [ -f "$OLD_WORKSPACE/yolo_train_run/full_finetune_60_20/weights/best.pt" ]; then
    cp "$OLD_WORKSPACE/yolo_train_run/full_finetune_60_20/weights/best.pt" "$CUSTOM_MODELS_DIR/manga_yolo_60_20_best.pt"
    echo "✅ Copied: manga_yolo_60_20_best.pt"
fi

if [ -f "$OLD_WORKSPACE/yolo_train_run/full_finetune_phase40/weights/best.pt" ]; then
    cp "$OLD_WORKSPACE/yolo_train_run/full_finetune_phase40/weights/best.pt" "$CUSTOM_MODELS_DIR/manga_yolo_phase40_best.pt"
    echo "✅ Copied: manga_yolo_phase40_best.pt"
fi

if [ -f "$OLD_WORKSPACE/yolo_train_run/augmented/weights/best.pt" ]; then
    cp "$OLD_WORKSPACE/yolo_train_run/augmented/weights/best.pt" "$CUSTOM_MODELS_DIR/manga_yolo_augmented_best.pt"
    echo "✅ Copied: manga_yolo_augmented_best.pt"
fi

# Copy the latest/best performing model as the default
if [ -f "$OLD_WORKSPACE/yolo_train_run/full_finetune_60_20/weights/best.pt" ]; then
    cp "$OLD_WORKSPACE/yolo_train_run/full_finetune_60_20/weights/best.pt" "$CUSTOM_MODELS_DIR/manga_yolo_custom_best.pt"
    echo "✅ Copied: manga_yolo_custom_best.pt (default custom model)"
fi

# Copy colorization model if it exists
if [ -f "$OLD_WORKSPACE/colorizer.pth" ]; then
    cp "$OLD_WORKSPACE/colorizer.pth" "./colorization/"
    echo "✅ Copied: colorizer.pth"
elif [ -f "$OLD_WORKSPACE/manga_translator_clean/colorization/colorizer.pth" ]; then
    echo "✅ Colorizer model already present"
fi

# Copy any other custom models
echo "Checking for additional custom models..."
find "$OLD_WORKSPACE" -name "*.pt" -path "*/yolo_train_run/*/weights/best.pt" | while read -r model_path; do
    # Extract a meaningful name from the path
    run_name=$(echo "$model_path" | sed 's|.*/yolo_train_run/||' | sed 's|/weights/best.pt||')
    dest_name="manga_yolo_${run_name}_best.pt"
    
    if [ ! -f "$CUSTOM_MODELS_DIR/$dest_name" ]; then
        cp "$model_path" "$CUSTOM_MODELS_DIR/$dest_name"
        echo "✅ Copied: $dest_name"
    fi
done

echo ""
echo "🤖 Step 3: Downloading AI models..."
echo "================================================="

# Download models using Python
python3 << 'EOF'
import sys
import os
from pathlib import Path

print("Starting model downloads...")

# Create model directories
Path("models/huggingface").mkdir(parents=True, exist_ok=True)
Path("models/checkpoints").mkdir(parents=True, exist_ok=True)

try:
    print("\n🎯 Checking your custom YOLO model...")
    
    # Check if custom model exists
    import os
    custom_model_path = "yolo_train_run/full_finetune_60_20/weights/best.pt"
    if os.path.exists(custom_model_path):
        print(f"✅ Custom YOLO model found: {custom_model_path}")
        print("💡 Using your custom-trained model (no default YOLO downloads needed)")
    else:
        print(f"❌ Custom YOLO model not found: {custom_model_path}")
        print("   Run ./copy_my_models.sh to copy from original workspace")
        
        # Check backup location
        backup_path = "models/checkpoints/custom_yolo_best.pt"
        if os.path.exists(backup_path):
            print(f"✅ Backup custom model found: {backup_path}")
        else:
            print(f"❌ Backup model not found: {backup_path}")
    
except Exception as e:
    print(f"❌ Model check failed: {e}")

try:
    print("\n📖 Downloading manga-ocr model...")
    import manga_ocr
    
    # Initialize manga-ocr (downloads model on first use)
    mocr = manga_ocr.MangaOcr()
    print("✅ manga-ocr ready")
    
except Exception as e:
    print(f"❌ manga-ocr download failed: {e}")

try:
    print("\n🧠 Downloading Gemma model (this may take a while)...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_name = "google/gemma-2b-it"
    
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="./models/huggingface"
    )
    
    print(f"Downloading model {model_name} (2GB)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="./models/huggingface",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    
    print("✅ Gemma model ready")
    
except Exception as e:
    print(f"❌ Gemma download failed: {e}")
    print("💡 You may need to accept terms at: https://huggingface.co/google/gemma-2b-it")
    print("💡 And login with: huggingface-cli login")

print("\n✅ Model download process completed!")
EOF

echo ""
echo "🔍 Step 3: Verifying installation..."
echo "================================================="

# Test basic imports
python3 << 'EOF'
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except ImportError:
    print("❌ PyTorch not installed")

try:
    import ultralytics
    print(f"✅ Ultralytics {ultralytics.__version__}")
except ImportError:
    print("❌ Ultralytics not installed")

try:
    import manga_ocr
    print("✅ manga-ocr available")
except ImportError:
    print("❌ manga-ocr not installed")

try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except ImportError:
    print("❌ Transformers not installed")

try:
    import streamlit
    print(f"✅ Streamlit {streamlit.__version__}")
except ImportError:
    print("❌ Streamlit not installed")

print("\n🎯 Checking model files...")

from pathlib import Path
import os

# Check custom YOLO model (your trained model)
custom_model = Path("yolo_train_run/full_finetune_phase40/weights/best.pt")
if custom_model.exists():
    print(f"✅ Custom YOLO model found: {custom_model}")
else:
    print(f"❌ Custom YOLO model missing: {custom_model}")
    backup_model = Path("models/checkpoints/custom_yolo_best.pt")
    if backup_model.exists():
        print(f"✅ Backup custom model found: {backup_model}")
    else:
        print(f"❌ Backup custom model missing: {backup_model}")
        print("   💡 Run ./copy_my_models.sh to copy from original workspace")

# Check HuggingFace cache
hf_cache = Path("./models/huggingface")
if hf_cache.exists() and any(hf_cache.iterdir()):
    print("✅ HuggingFace models cached")
else:
    print("⚠️  HuggingFace models not cached locally")

EOF

echo ""
echo "🚀 Step 4: Ready to use!"
echo "================================================="
echo ""
echo "✅ Setup complete! You can now:"
echo ""
echo "   🌐 Start web interface:"
echo "      python web/app.py"
echo ""
echo "   🎯 Train a model:"
echo "      python training/advanced_train_yolo.py"
echo ""
echo "   📊 Analyze dataset:"
echo "      python evaluation/analyze_dataset.py --data training/datasets/custom_manga.yaml"
echo ""
echo "   🔍 Translate image:"
echo "      python examples/translator_standalone.py -i input.jpg -o output.jpg"
echo ""
echo "💡 All models will be automatically downloaded on first use if not already cached."
echo ""
echo "🎉 Happy translating!"
