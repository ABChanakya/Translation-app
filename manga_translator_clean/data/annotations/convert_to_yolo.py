"""
YOLO Format Annotation Converter
Convert annotations from various formats to YOLO format
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from typing import Dict, List, Tuple


def convert_coco_to_yolo(coco_json: str, output_dir: str, class_mapping: Dict[int, int] = None):
    """
    Convert COCO format annotations to YOLO format
    
    Args:
        coco_json: Path to COCO JSON file
        output_dir: Directory to save YOLO format labels
        class_mapping: Optional mapping of COCO class IDs to YOLO class IDs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    # Create image ID to filename mapping
    images = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Convert each image's annotations
    for img_id, annotations in annotations_by_image.items():
        img_info = images[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Create label file
        label_file = output_dir / f"{Path(img_info['file_name']).stem}.txt"
        
        with open(label_file, 'w') as f:
            for ann in annotations:
                # COCO bbox format: [x, y, width, height] (top-left corner)
                x, y, w, h = ann['bbox']
                category_id = ann['category_id']
                
                # Apply class mapping if provided
                if class_mapping:
                    category_id = class_mapping.get(category_id, category_id)
                
                # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Converted {len(annotations_by_image)} images to YOLO format")
    print(f"Labels saved to: {output_dir}")


def convert_labelstudio_to_yolo(json_file: str, output_dir: str, 
                                 class_names: List[str], image_dir: str = None):
    """
    Convert Label Studio annotations to YOLO format
    
    Args:
        json_file: Path to Label Studio JSON export
        output_dir: Directory to save YOLO format labels
        class_names: List of class names in order
        image_dir: Optional directory containing images (to get dimensions)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    class_to_id = {name: i for i, name in enumerate(class_names)}
    
    for item in data:
        # Get image info
        image_file = item['data']['image']
        image_name = Path(image_file).stem
        
        # Get image dimensions from annotation or file
        if 'original_width' in item:
            img_width = item['original_width']
            img_height = item['original_height']
        elif image_dir:
            from PIL import Image
            img_path = Path(image_dir) / Path(image_file).name
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        else:
            print(f"Warning: No image dimensions for {image_name}, skipping...")
            continue
        
        # Create label file
        label_file = output_dir / f"{image_name}.txt"
        
        with open(label_file, 'w') as f:
            if 'annotations' in item and len(item['annotations']) > 0:
                for result in item['annotations'][0]['result']:
                    if result['type'] == 'rectanglelabels':
                        # Get bounding box (Label Studio format: percentage)
                        value = result['value']
                        x_pct = value['x'] / 100
                        y_pct = value['y'] / 100
                        w_pct = value['width'] / 100
                        h_pct = value['height'] / 100
                        
                        # Convert to YOLO format (center coordinates)
                        x_center = x_pct + w_pct / 2
                        y_center = y_pct + h_pct / 2
                        
                        # Get class
                        class_name = value['rectanglelabels'][0]
                        class_id = class_to_id.get(class_name, 0)
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_pct:.6f} {h_pct:.6f}\n")
    
    print(f"Converted Label Studio annotations to YOLO format")
    print(f"Labels saved to: {output_dir}")


def convert_voc_to_yolo(xml_dir: str, output_dir: str, class_names: List[str]):
    """
    Convert Pascal VOC XML annotations to YOLO format
    
    Args:
        xml_dir: Directory containing VOC XML files
        output_dir: Directory to save YOLO format labels
        class_names: List of class names in order
    """
    xml_dir = Path(xml_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_to_id = {name: i for i, name in enumerate(class_names)}
    
    for xml_file in xml_dir.glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # Create label file
        label_file = output_dir / f"{xml_file.stem}.txt"
        
        with open(label_file, 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_id = class_to_id.get(class_name, 0)
                
                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to YOLO format
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Converted VOC annotations to YOLO format")
    print(f"Labels saved to: {output_dir}")


def split_dataset(images_dir: str, labels_dir: str, output_dir: str,
                  train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1,
                  seed: int = 42):
    """
    Split dataset into train/val/test sets
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        output_dir: Output directory for split dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for reproducibility
    """
    import shutil
    import random
    
    random.seed(seed)
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    random.shuffle(image_files)
    
    # Calculate split indices
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    # Create split directories and copy files
    for split_name, split_files in splits.items():
        split_img_dir = output_dir / split_name / 'images'
        split_lbl_dir = output_dir / split_name / 'labels'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in split_files:
            # Copy image
            shutil.copy(img_file, split_img_dir / img_file.name)
            
            # Copy label if exists
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, split_lbl_dir / label_file.name)
        
        print(f"{split_name}: {len(split_files)} images")
    
    print(f"\nDataset split complete!")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert annotations to YOLO format')
    parser.add_argument('--format', type=str, required=True, 
                       choices=['coco', 'labelstudio', 'voc'],
                       help='Input annotation format')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for YOLO labels')
    parser.add_argument('--classes', type=str, nargs='+',
                       help='Class names in order')
    parser.add_argument('--images', type=str,
                       help='Images directory (for Label Studio)')
    parser.add_argument('--split', action='store_true',
                       help='Split dataset after conversion')
    parser.add_argument('--split-output', type=str,
                       help='Output directory for split dataset')
    
    args = parser.parse_args()
    
    if args.format == 'coco':
        convert_coco_to_yolo(args.input, args.output)
    elif args.format == 'labelstudio':
        if not args.classes:
            print("Error: --classes required for Label Studio format")
            return
        convert_labelstudio_to_yolo(args.input, args.output, args.classes, args.images)
    elif args.format == 'voc':
        if not args.classes:
            print("Error: --classes required for VOC format")
            return
        convert_voc_to_yolo(args.input, args.output, args.classes)
    
    if args.split:
        if not args.split_output:
            print("Error: --split-output required when using --split")
            return
        if not args.images:
            print("Error: --images required when using --split")
            return
        split_dataset(args.images, args.output, args.split_output)


if __name__ == '__main__':
    main()
