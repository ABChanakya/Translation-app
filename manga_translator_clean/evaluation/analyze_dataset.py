"""Dataset statistics and analysis for YOLO-formatted manga datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency for plot output
    plt = None


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.common import (  # noqa: E402
    normalize_class_names,
    resolve_project_path,
    resolve_split_image_dir,
    resolve_split_label_dir,
)


def analyze_dataset(data_yaml: str, save_dir: str | None = None) -> dict:
    """Analyze a YOLO dataset and optionally save summary artifacts."""
    resolved_data_yaml = resolve_project_path(data_yaml)
    with open(resolved_data_yaml, "r", encoding="utf-8") as handle:
        data_config = yaml.safe_load(handle)

    class_names = normalize_class_names(data_config.get("names"))
    stats = {
        "total_images": 0,
        "total_annotations": 0,
        "class_distribution": Counter(),
        "image_sizes": [],
        "bbox_sizes": [],
        "annotations_per_image": [],
    }

    print("\n" + "=" * 60)
    print("Dataset Analysis")
    print("=" * 60)

    for split in ("train", "val", "test"):
        if split not in data_config:
            continue

        images_dir = resolve_split_image_dir(resolved_data_yaml, split)
        labels_dir = resolve_split_label_dir(images_dir)

        if not images_dir.exists():
            print(f"\nSkipping {split}: {images_dir} does not exist")
            continue

        print(f"\nAnalyzing {split} set...")
        image_files = sorted(
            path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )

        for image_file in image_files:
            stats["total_images"] += 1

            with Image.open(image_file) as image:
                width, height = image.size
                stats["image_sizes"].append((width, height))

            label_file = labels_dir / f"{image_file.stem}.txt"
            if not label_file.exists():
                stats["annotations_per_image"].append(0)
                continue

            with open(label_file, "r", encoding="utf-8") as handle:
                annotations = [line.strip().split() for line in handle if line.strip()]

            stats["annotations_per_image"].append(len(annotations))
            for annotation in annotations:
                if len(annotation) < 5:
                    continue
                class_id = int(float(annotation[0]))
                bbox_width = float(annotation[3]) * width
                bbox_height = float(annotation[4]) * height
                stats["class_distribution"][class_id] += 1
                stats["total_annotations"] += 1
                stats["bbox_sizes"].append((bbox_width, bbox_height))

    if not stats["image_sizes"]:
        raise RuntimeError(f"No images were found while analyzing {resolved_data_yaml}")

    widths = [size[0] for size in stats["image_sizes"]]
    heights = [size[1] for size in stats["image_sizes"]]
    bbox_widths = [size[0] for size in stats["bbox_sizes"]] or [0]
    bbox_heights = [size[1] for size in stats["bbox_sizes"]] or [0]
    average_annotations = float(np.mean(stats["annotations_per_image"])) if stats["annotations_per_image"] else 0.0

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total Images: {stats['total_images']}")
    print(f"Total Annotations: {stats['total_annotations']}")
    print(f"Average Annotations per Image: {average_annotations:.2f}")
    print("\nClass Distribution:")
    for class_id, count in sorted(stats["class_distribution"].items()):
        class_name = class_names.get(class_id, f"class_{class_id}")
        percentage = (count / stats["total_annotations"] * 100.0) if stats["total_annotations"] else 0.0
        print(f"  {class_name} (ID {class_id}): {count} ({percentage:.1f}%)")

    print("\nImage Size Statistics:")
    print(f"  Width  - Min: {min(widths)}, Max: {max(widths)}, Avg: {np.mean(widths):.0f}")
    print(f"  Height - Min: {min(heights)}, Max: {max(heights)}, Avg: {np.mean(heights):.0f}")

    print("\nBounding Box Size Statistics:")
    print(f"  Width  - Min: {min(bbox_widths):.0f}, Max: {max(bbox_widths):.0f}, Avg: {np.mean(bbox_widths):.0f}")
    print(f"  Height - Min: {min(bbox_heights):.0f}, Max: {max(bbox_heights):.0f}, Avg: {np.mean(bbox_heights):.0f}")

    report = {
        "dataset": str(resolved_data_yaml),
        "total_images": stats["total_images"],
        "total_annotations": stats["total_annotations"],
        "avg_annotations_per_image": average_annotations,
        "class_distribution": {class_names.get(key, f"class_{key}"): value for key, value in stats["class_distribution"].items()},
        "image_size_stats": {
            "width": {"min": int(min(widths)), "max": int(max(widths)), "avg": float(np.mean(widths))},
            "height": {"min": int(min(heights)), "max": int(max(heights)), "avg": float(np.mean(heights))},
        },
        "bbox_size_stats": {
            "width": {"min": float(min(bbox_widths)), "max": float(max(bbox_widths)), "avg": float(np.mean(bbox_widths))},
            "height": {"min": float(min(bbox_heights)), "max": float(max(bbox_heights)), "avg": float(np.mean(bbox_heights))},
        },
    }

    if save_dir:
        output_dir = resolve_project_path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "dataset_stats.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        if plt is not None:
            plt.figure(figsize=(10, 6))
            classes = [class_names.get(i, f"class_{i}") for i in sorted(stats["class_distribution"])]
            counts = [stats["class_distribution"][i] for i in sorted(stats["class_distribution"])]
            plt.bar(classes, counts)
            plt.title("Class Distribution")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / "class_distribution.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.hist(stats["annotations_per_image"], bins=20, edgecolor="black")
            plt.title("Annotations per Image Distribution")
            plt.xlabel("Number of Annotations")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(output_dir / "annotations_per_image.png")
            plt.close()

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a YOLO dataset")
    parser.add_argument(
        "--data",
        type=str,
        default="training/datasets/custom_manga.yaml",
        help="Path to the dataset YAML",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="evaluation/dataset_analysis",
        help="Directory to save analysis artifacts",
    )
    args = parser.parse_args()
    analyze_dataset(args.data, args.save_dir)


if __name__ == "__main__":
    main()
