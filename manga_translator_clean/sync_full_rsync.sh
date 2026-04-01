#!/bin/bash
# Full sync using rsync - syncs entire codebase

echo "🔄 Full Sync: Translation_tool-2 → UNI/translation_tool"
echo ""

NEW_WORKSPACE="/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean"
OLD_WORKSPACE="/home/chanakya/chanakya/UNI/translation_tool"

echo "Source:      $NEW_WORKSPACE"
echo "Destination: $OLD_WORKSPACE"
echo ""
echo "⚠️  This will sync ALL files (except excluded patterns)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Running rsync..."
echo ""

# Rsync with exclusions
rsync -av --progress \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='*.log' \
    --exclude='runs/' \
    --exclude='data/images/' \
    --exclude='data/labels/' \
    --exclude='translated_output/' \
    --exclude='translated_pages/' \
    --exclude='data1/' \
    --exclude='data2/' \
    "$NEW_WORKSPACE/" "$OLD_WORKSPACE/"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Full Sync Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "All updated files are now in: $OLD_WORKSPACE"
echo ""
echo "Next steps:"
echo "  cd $OLD_WORKSPACE"
echo "  source .venv/bin/activate"
echo "  pip install -r requirements.txt"
echo ""
