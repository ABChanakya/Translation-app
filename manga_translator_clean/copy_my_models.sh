#!/bin/bash
# Copy only YOUR specific models that you actually use

echo "📋 Copying your specific models..."

# Source paths
OLD_WORKSPACE="/home/chanakya/chanakya/UNI/translation_tool"
NEW_WORKSPACE="/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean"

# Create model directories
mkdir -p "$NEW_WORKSPACE/models/checkpoints"
mkdir -p "$NEW_WORKSPACE/colorization"

echo ""
echo "Copying your custom YOLO model (the one you actually use):"

# Copy YOUR custom trained model (full_finetune_60_20 - the best model)
if [ -f "$OLD_WORKSPACE/yolo_train_run/full_finetune_60_20/weights/best.pt" ]; then
    mkdir -p "$NEW_WORKSPACE/yolo_train_run/full_finetune_60_20/weights"
    cp "$OLD_WORKSPACE/yolo_train_run/full_finetune_60_20/weights/best.pt" "$NEW_WORKSPACE/yolo_train_run/full_finetune_60_20/weights/best.pt"
    echo "✅ Copied: yolo_train_run/full_finetune_60_20/weights/best.pt (your best trained model)"
    
    # Also copy as backup to models/checkpoints for compatibility
    cp "$OLD_WORKSPACE/yolo_train_run/full_finetune_60_20/weights/best.pt" "$NEW_WORKSPACE/models/checkpoints/custom_yolo_best.pt"
    echo "✅ Copied: custom_yolo_best.pt (backup copy)"
else
    echo "❌ Your main custom model not found at: $OLD_WORKSPACE/yolo_train_run/full_finetune_60_20/weights/best.pt"
    
    # Fallback: try phase20 if phase40 doesn't exist
    if [ -f "$OLD_WORKSPACE/yolo_train_run/full_finetune_phase20/weights/best.pt" ]; then
        mkdir -p "$NEW_WORKSPACE/yolo_train_run/full_finetune_phase20/weights"
        cp "$OLD_WORKSPACE/yolo_train_run/full_finetune_phase20/weights/best.pt" "$NEW_WORKSPACE/yolo_train_run/full_finetune_phase20/weights/best.pt"
        echo "✅ Copied: yolo_train_run/full_finetune_phase20/weights/best.pt (fallback model)"
        
        cp "$OLD_WORKSPACE/yolo_train_run/full_finetune_phase20/weights/best.pt" "$NEW_WORKSPACE/models/checkpoints/custom_yolo_best.pt"
        echo "✅ Copied: custom_yolo_best.pt (backup copy)"
    else
        echo "❌ No custom models found in phase20 or phase40"
    fi
fi

echo ""
echo "Copying colorization model:"

# Copy only the colorization model (skip default YOLO models)
if [ -f "$OLD_WORKSPACE/colorizer.pth" ]; then
    mkdir -p "$NEW_WORKSPACE/colorization"
    cp "$OLD_WORKSPACE/colorizer.pth" "$NEW_WORKSPACE/colorization/"
    echo "✅ Copied: colorizer.pth"
else
    echo "❌ Colorization model not found: colorizer.pth"
fi

echo ""
echo "✅ Done! Your custom model is ready:"
echo "   • Custom YOLO: yolo_train_run/full_finetune_60_20/weights/best.pt"
echo "   • Backup copy: models/checkpoints/custom_yolo_best.pt"
echo "   • Colorizer: colorization/colorizer.pth"
echo ""
echo "🎯 Your code uses: YOLO_MODEL_PATH = \"yolo_train_run/full_finetune_60_20/weights/best.pt\""
echo "💡 No default YOLO models needed - you have your own custom-trained model!"
