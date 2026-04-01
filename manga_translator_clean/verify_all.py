#!/usr/bin/env python3
"""Comprehensive verification of all components"""

import os
import sys
from pathlib import Path

def check_file(path, description=""):
    exists = os.path.exists(path)
    icon = "✅" if exists else "❌"
    print(f"{icon} {path:<60} {description}")
    return exists

def check_symlink(path, target):
    if os.path.islink(path):
        real = os.path.realpath(path)
        exists = os.path.exists(real)
        icon = "✅" if exists else "⚠️"
        print(f"{icon} {path:<40} → {target} [{'OK' if exists else 'BROKEN'}]")
        return exists
    else:
        print(f"❌ {path:<40} [NOT A SYMLINK]")
        return False

print("=" * 80)
print("🔍 AUTO MANGA TRANSLATION - VERIFICATION")
print("=" * 80)

all_good = True

# Check data symlinks
print("\n📁 DATA DIRECTORIES:")
all_good &= check_symlink("data/manga_chapters", "../../data1")
all_good &= check_symlink("data/additional_data", "../../data2")
all_good &= check_symlink("data/pseudo_labels", "../../pseudo_labels")
all_good &= check_symlink("data/training_data", "../../data")

# Check training scripts
print("\n🏋️ TRAINING SCRIPTS:")
all_good &= check_file("training/advanced_train_yolo.py", "Advanced training")
all_good &= check_file("training/train_and_eval.py", "Train + eval pipeline")
all_good &= check_file("training/create_pseudo_labels.py", "Generate labels")
all_good &= check_file("training/train_yolo.py", "Basic training")

# Check configs
print("\n⚙️ CONFIGURATION FILES:")
all_good &= check_file("training/configs/config.yaml", "3-class config")
all_good &= check_file("training/configs/config1.yaml", "5-class config")
all_good &= check_file("training/datasets/custom_manga.yaml", "Dataset config")
all_good &= check_file("training/datasets/classes.txt", "Class labels")

# Check data tools
print("\n🛠️ DATA UTILITIES:")
all_good &= check_file("data/scrapers/rawkuma_scraper.py", "Web scraper")
all_good &= check_file("data/annotations/convert_yolo_to_labelstudio.py", "Format converter")
all_good &= check_file("data/utils/dedupe.py", "Deduplication")
all_good &= check_file("data/utils/counts.py", "Statistics")
all_good &= check_file("data/utils/deleter.py", "Batch operations")
all_good &= check_file("data/utils/rename_seq.py", "Sequential rename")
all_good &= check_file("data/utils/page_upload.py", "Upload utility")

# Check models
print("\n🤖 MODEL FILES:")
all_good &= check_file("models/checkpoints/yolo11n.pt", "YOLO11 nano")
all_good &= check_file("models/checkpoints/yolov8m.pt", "YOLOv8 medium")
all_good &= check_file("colorization/colorizer.pth", "Colorization model")

# Check evaluation
print("\n📊 EVALUATION TOOLS:")
all_good &= check_file("evaluation/threshold_sweep.py", "Threshold optimization")
all_good &= check_file("evaluation/analyze_dataset.py", "Dataset analysis")
all_good &= check_file("training/utils/nvidia_gpu_monitor.py", "GPU monitoring")

# Check examples
print("\n💻 EXAMPLE SCRIPTS:")
all_good &= check_file("examples/demo_legacy.py", "Original demo")
all_good &= check_file("examples/demo_2606.py", "June 26 demo")
all_good &= check_file("examples/demo_2606_refactored.py", "Refactored demo")
all_good &= check_file("examples/demo_lama.py", "LaMa demo")
all_good &= check_file("examples/colorize_demo.py", "Colorization demo")
all_good &= check_file("examples/translator_standalone.py", "Standalone translator")
all_good &= check_file("examples/pipeline_visual_flow.py", "Visual pipeline")

# Check web interface
print("\n🌐 WEB INTERFACE:")
all_good &= check_file("web/app.py", "Flask server")
all_good &= check_file("web/templates/index.html", "Homepage")
all_good &= check_file("web/static/css/style.css", "Stylesheets")
all_good &= check_file("app.py", "Root app launcher")

# Check documentation
print("\n📖 DOCUMENTATION:")
all_good &= check_file("README.md", "Main readme")
all_good &= check_file("QUICKSTART.md", "Quick start")
all_good &= check_file("DEVELOPER_GUIDE.md", "Developer guide")
all_good &= check_file("FUTURE_ACCESSIBILITY_AND_SCALING.md", "Future planning notes")
all_good &= check_file("data/README.md", "Data workflow")
all_good &= check_file("colorization/readme.md", "Colorization guide")

# Count available manga chapters
print("\n📚 DATA STATISTICS:")
if os.path.exists("data/manga_chapters"):
    chapter_count = len([d for d in os.listdir("data/manga_chapters") if os.path.isdir(f"data/manga_chapters/{d}")])
    print(f"   📖 Manga chapters available: {chapter_count}")

# Final status
print("\n" + "=" * 80)
if all_good:
    print("✅ ALL SYSTEMS OPERATIONAL - READY FOR DEVELOPMENT!")
else:
    print("⚠️  SOME COMPONENTS MISSING - CHECK ERRORS ABOVE")
print("=" * 80)

print("\n🚀 QUICK START:")
print("   • Web Interface:  python web/app.py")
print("   • Train Model:    python training/advanced_train_yolo.py")
print("   • Documentation:  See README.md, QUICKSTART.md, and DEVELOPER_GUIDE.md")
print()

sys.exit(0 if all_good else 1)
