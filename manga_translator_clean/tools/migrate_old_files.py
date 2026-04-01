#!/usr/bin/env python3
"""
Automated migration script for moving files from old structure to new organized structure
"""
import os
import shutil
from pathlib import Path

def migrate_files():
    """Migrate all important files to new structure"""
    
    # Base directories
    old_base = Path("/home/chanakya/chanakya/UNI/translation_tool")
    # If this script is copied/moved, determine the new_base automatically
    # tools/ is located under <manga_translator_clean>/tools, so parents[2] -> manga_translator_clean
    new_base = Path(__file__).resolve().parents[2]
    
    # Create necessary directories
    directories = [
        "data/scrapers",
        "data/utils",
        "examples",
        "training/configs",
        "training/utils"
    ]
    
    for dir_path in directories:
        (new_base / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    # File migrations (source -> destination)
    file_migrations = {
        # Configs
        "config.yaml": "training/configs/config.yaml",
        "config1.yaml": "training/configs/config1.yaml",
        "classes.txt": "training/datasets/classes.txt",
        
        # Scrapers
        "rawkuma_scraper.py": "data/scrapers/rawkuma_scraper.py",
        
        # Utilities
        "dedupe.py": "data/utils/dedupe.py",
        "counts.py": "data/utils/counts.py",
        "deleter.py": "data/utils/deleter.py",
        "rename_seq.py": "data/utils/rename_seq.py",
        "page_upload.py": "data/utils/page_upload.py",
        "nvidia_gpu_monitor.py": "training/utils/nvidia_gpu_monitor.py",
        
        # Evaluation
        "evaluate_thresholds.py": "evaluation/threshold_sweep.py",
        "convert_yolo_to_labelstudio.py": "data/annotations/convert_yolo_to_labelstudio.py",
        
        # Examples/Demos
        "DEMO.py": "examples/demo_legacy.py",
        "DEMO_26.06.py": "examples/demo_2606.py",
        "DEMO_26.06_REFACTORED.py": "examples/demo_2606_refactored.py",
        "demo_lama.py": "examples/demo_lama.py",
        "colorize_manga_demo.py": "examples/colorize_demo.py",
        "translator.py": "examples/translator_standalone.py",
        "PIPELINE_VISUAL_FLOW.py": "examples/pipeline_visual_flow.py",
        
        # Models
        "yolo11n.pt": "models/checkpoints/yolo11n.pt",
        "yolov8m.pt": "models/checkpoints/yolov8m.pt",
        "colorizer.pth": "colorization/colorizer.pth",
        
        # Documentation
        "LAMA_INPAINTING_EXPLAINED.md": "colorization/readme.md",
        "WHERE_IS_LAMA.md": "colorization/readme.md",
        "manga_localizer_prompt.txt": "src/translators/manga_localizer_prompt.txt",
    }
    
    # Copy files
    copied = 0
    skipped = 0
    errors = 0
    
    for src_name, dest_path in file_migrations.items():
        src = old_base / src_name
        dest = new_base / dest_path
        
        if not src.exists():
            print(f"⚠ Source not found: {src_name}")
            skipped += 1
            continue
        
        if dest.exists():
            print(f"⊘ Already exists: {dest_path}")
            skipped += 1
            continue
        
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            print(f"✓ Copied: {src_name} → {dest_path}")
            copied += 1
        except Exception as e:
            print(f"✗ Error copying {src_name}: {e}")
            errors += 1
    
    # Create symlinks for large data directories
    symlinks = {
        "data1": "data/manga_chapters",
        "data2": "data/additional_data",
        "pseudo_labels": "data/pseudo_labels",
    }
    
    for src_name, dest_path in symlinks.items():
        src = old_base / src_name
        dest = new_base / dest_path
        
        if not src.exists():
            print(f"⚠ Source directory not found: {src_name}")
            continue
        
        if dest.exists():
            print(f"⊘ Link already exists: {dest_path}")
            continue
        
        try:
            # Create relative symlink
            rel_src = os.path.relpath(src, dest.parent)
            os.symlink(rel_src, dest)
            print(f"✓ Linked: {src_name} → {dest_path}")
        except Exception as e:
            print(f"✗ Error linking {src_name}: {e}")
            errors += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Migration Summary:")
    print(f"{'='*60}")
    print(f"✓ Files copied: {copied}")
    print(f"⊘ Files skipped: {skipped}")
    print(f"✗ Errors: {errors}")
    print(f"{'='*60}\n")
    
    if errors == 0:
        print("✅ Migration completed successfully!")
        print("\nNext steps:")
        print("1. Update config files in training/configs/ to point to your data")
        print("2. Test training: python training/advanced_train_yolo.py")
        print("3. Start web interface: cd web && python app.py")
    else:
        print("⚠ Migration completed with errors. Please check the messages above.")


if __name__ == "__main__":
    print("Starting migration...\n")
    migrate_files()
