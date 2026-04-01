#!/bin/bash
# Quick Start Guide for Manga Translator

echo "🎌 Manga Translation Developer Kit - Quick Start"
echo "=================================================="
echo ""

# Check if in correct directory
if [ ! -f "web/app.py" ]; then
    echo "❌ Error: Not in manga_translator_clean directory"
    echo "   Run: cd ~/chanakya/Translation_tool-2/manga_translator_clean"
    exit 1
fi

echo "✓ In correct directory"
echo ""

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "   Installing dependencies..."
    pip install -r requirements-minimal.txt
else
    echo "✓ Virtual environment found"
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

echo ""

# Check models
echo "Checking models..."
python3 << 'EOF'
from pathlib import Path

model_path = Path("yolo_train_run/full_finetune_60_20/weights/best.pt")
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✓ Custom YOLO model: {model_path} ({size_mb:.1f} MB)")
else:
    print(f"❌ Custom model not found!")
    print("   Run: ./copy_my_models.sh")
    exit(1)

colorizer = Path("colorization/colorizer.pth")
if colorizer.exists():
    print(f"✓ Colorizer model found")
else:
    print(f"⚠️  Colorizer not found (optional)")
EOF

echo ""
echo "=================================================="
echo "🚀 Ready to Start!"
echo "=================================================="
echo ""
echo "Choose an option:"
echo ""
echo "  1. Web UI (Flask):       python web/app.py"
echo "  2. Streamlit UI:         streamlit run web/streamlit_app.py"
echo "  3. CLI Translator:       python examples/translator_standalone.py --help"
echo "  4. Test System:          python test_translator.py"
echo ""
echo "Example CLI usage:"
echo "  python examples/translator_standalone.py \\"
echo "    --input data/test/page1.jpg \\"
echo "    --output translated/page1.jpg \\"
echo "    --source ja --target en"
echo ""
