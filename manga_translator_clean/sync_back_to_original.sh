#!/bin/bash
# Sync the updated code back to the original workspace

echo "🔄 Syncing updated code back to original workspace..."
echo ""

# Source and destination paths
NEW_WORKSPACE="/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean"
OLD_WORKSPACE="/home/chanakya/chanakya/UNI/translation_tool"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Source:      $NEW_WORKSPACE"
echo "Destination: $OLD_WORKSPACE"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Copying updated files..."
echo "═══════════════════════════════════════════════════════════"
echo ""

# Function to copy and report
copy_file() {
    local file="$1"
    local src="$NEW_WORKSPACE/$file"
    local dst="$OLD_WORKSPACE/$file"
    
    if [ -f "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        echo -e "${GREEN}✅${NC} Copied: $file"
        return 0
    else
        echo -e "${RED}❌${NC} Not found: $file"
        return 1
    fi
}

# Copy updated configuration files
echo "📋 Configuration files:"
copy_file "config/settings.py"
copy_file "requirements.txt"
copy_file "requirements-minimal.txt"

echo ""
echo "🔧 Setup scripts:"
copy_file "copy_my_models.sh"
copy_file "setup_models.py"
copy_file "download_models.py"
copy_file "setup_complete.sh"

echo ""
echo "🌐 Translator fixes:"
copy_file "src/translators/google.py"

echo ""
echo "📖 Documentation:"
copy_file "README.md"
copy_file "QUICKSTART.md"
copy_file "DEVELOPER_GUIDE.md"

echo ""
echo "🧪 Test scripts:"
copy_file "test_translator.py"
copy_file "quick_start.sh"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Copying updated models..."
echo "═══════════════════════════════════════════════════════════"
echo ""

# Copy the phase 60 model
if [ -f "$NEW_WORKSPACE/yolo_train_run/full_finetune_60_20/weights/best.pt" ]; then
    mkdir -p "$OLD_WORKSPACE/yolo_train_run/full_finetune_60_20/weights"
    cp "$NEW_WORKSPACE/yolo_train_run/full_finetune_60_20/weights/best.pt" \
       "$OLD_WORKSPACE/yolo_train_run/full_finetune_60_20/weights/best.pt"
    echo -e "${GREEN}✅${NC} Copied: yolo_train_run/full_finetune_60_20/weights/best.pt"
else
    echo -e "${YELLOW}⚠️${NC}  Model already in original location"
fi

# Copy backup model
if [ -f "$NEW_WORKSPACE/models/checkpoints/custom_yolo_best.pt" ]; then
    mkdir -p "$OLD_WORKSPACE/models/checkpoints"
    cp "$NEW_WORKSPACE/models/checkpoints/custom_yolo_best.pt" \
       "$OLD_WORKSPACE/models/checkpoints/custom_yolo_best.pt"
    echo -e "${GREEN}✅${NC} Copied: models/checkpoints/custom_yolo_best.pt"
fi

# Copy colorizer if updated
if [ -f "$NEW_WORKSPACE/colorization/colorizer.pth" ]; then
    mkdir -p "$OLD_WORKSPACE/colorization"
    cp "$NEW_WORKSPACE/colorization/colorizer.pth" \
       "$OLD_WORKSPACE/colorization/colorizer.pth"
    echo -e "${GREEN}✅${NC} Copied: colorization/colorizer.pth"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Sync Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Updated files are now in: $OLD_WORKSPACE"
echo ""
echo "Next steps in original workspace:"
echo "  1. cd $OLD_WORKSPACE"
echo "  2. source .venv/bin/activate  (or create new venv)"
echo "  3. pip install -r requirements.txt"
echo "  4. python test_translator.py"
echo ""
