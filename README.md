# Manga Translation Toolkit

**A comprehensive end-to-end solution for speech bubble detection, OCR, and translation in manga/manhwa images.**

<p align="center">
  <a href="#features"><strong>Explore the features »</strong></a>
  &nbsp;|&nbsp;
  <a href="#quickstart">Quickstart</a>
  &nbsp;|&nbsp;
  <a href="#installation">Installation</a>
  &nbsp;|&nbsp;
  <a href="#configuration">Configuration</a>
  &nbsp;|&nbsp;
  <a href="#data-preparation">Data Preparation</a>
  &nbsp;|&nbsp;
  <a href="#model-training">Model Training</a>
  &nbsp;|&nbsp;
  <a href="#usage">Usage</a>
  &nbsp;|&nbsp;
  <a href="#utility-scripts">Utility Scripts</a>
  &nbsp;|&nbsp;
  <a href="#project-structure">Project Structure</a>
  &nbsp;|&nbsp;
  <a href="#contributing">Contributing</a>
  &nbsp;|&nbsp;
  <a href="#license">License</a>
</p>

---

## Features

- **Accurate Speech Bubble Detection**: Custom YOLOv8 model (3 classes: dialogue, sound effects, signs) with mosaic, mixup, HSV jitter augmentations.
- **Image Preprocessing**: Automatic deskew, grayscale contrast enhancement, and cropping of detected regions for optimal OCR.
- **High‑Precision OCR**: Powered by [Manga‑OCR](https://github.com/TachibanaYoshino/Manga-OCR) (v0.1.14) tuned for Japanese typefaces.
- **Seamless Translation Pipeline**: Integrates Google Translate via `googletrans==4.0.0-rc1`; supports batch and interactive modes.
- **Stylized Overlays**: Word-wrapping, font scaling, and semi-transparent bounding boxes ensure readability on original pages.
- **Web Interface**: Real‑time previews using Streamlit (`page_upload.py`) with progress bars and history tracking.
- **Command‑Line Tool**: Flexible `translator.py` for single-image or directory translation with customizable confidence, output format (PNG, PDF), and threading.
- **Training & Fine‑Tuning**: `trainyolo.py` script built on Ultralytics API; supports custom datasets, dynamic image sizing, and learning‑rate scheduling.
- **Labeling Utilities**: Generate pseudo-labels (`create_pseudo_labels.py`), convert YOLO annotations to Label Studio format (`convert_yolo_to_labelstudio.py`), and a live editor (`realtime_label_editor.py`).
- **Diagnostics & Statistics**: `counts.py` for class-distribution histograms and error‑rate estimation.
- **Automated Testing**: Unit tests under `test/` verify OCR accuracy, overlay positioning, and translation consistency.

---

## Quickstart

1. **Clone & install** (see [Installation](#installation)).
2. **Generate pseudo‑labels** on unlabeled data:
   ```bash
   python create_pseudo_labels.py \
     --model yolov8n.pt \
     --input data/images/unlabeled \
     --output pseudo_labels/run1 \
     --conf 0.3 --device cpu
   ```
3. **Fine‑tune the detector** on corrected labels:
   ```bash
   python trainyolo.py --data config.yaml --epochs 200 --batch 32
   ```
4. **Translate a directory** of pages:
   ```bash
   python translator.py \
     --input scans/ \
     --model yolo_train_run/weights/best.pt \
     --lang en --conf 0.25 --threads 4 \
     --output translated/ --format png
   ```
5. **Start the web UI** for interactive translation:
   ```bash
   streamlit run page_upload.py -- --model yolo_train_run/weights/best.pt --device cuda
   ```

---

## Installation

### 1. Prerequisites

- **OS**: Linux, macOS, or Windows
- **Python**: 3.8+
- **CUDA** (optional): For GPU acceleration with `torch>=2.3.0+cu1x`
- **Git**: To clone and update the repo

### 2. Clone Repository

```bash
git clone https://github.com/ABChanakya/Tranlation-app.git
cd Tranlation-app
```

### 3. Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate    # Windows
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

*Key packages in `requirements.txt`:*  
`torch>=2.3.0`  
`ultralytics>=8.0.0`  
`manga-ocr>=0.1.14`  
`opencv-python>=4.7.0`  
`Pillow>=9.0.0`  
`requests>=2.25.0`  
`googletrans==4.0.0-rc1`  
`tqdm>=4.64.0`  
`streamlit>=1.15.0`

---

## Configuration

Edit `config.yaml` to reflect your dataset and class names:

```yaml
# Number of detection classes
nc: 3
# Path to training images (YOLO format)
train: "/home/chanakya/chanakya/UNI/translation_tool/data/images/train"
# Path to validation images
val:   "/home/chanakya/chanakya/UNI/translation_tool/data/images/val"
# Class names mapping
names:
  0: Dialogue
  1: Sound_Effects
  2: Signs_Labels
```

- **nc**: must match line count in `names`.  
- **train/val**: support absolute or relative paths.  
- **names**: keys are class indices; values are displayed labels.

---

## Data Preparation

### Directory Layout

```
translation_tool/
├── data/
│   ├── images/
│   │   ├── train/       # .jpg/.png
│   │   └── val/
│   └── labels/          # YOLO .txt per image
└── pseudo_labels/       # Outputs from pseudo-labeling
```

### Generate Pseudo‑Labels

Auto-label unlabeled images using a pretrained YOLOv8 model:

```bash
python create_pseudo_labels.py \
  --model yolov8n.pt \
  --input data/images/unlabeled \
  --output pseudo_labels/run1 \
  --conf 0.3 --device cpu
```

Outputs in `pseudo_labels/run1/labels/*.txt` (YOLO format).

### Convert to Label Studio Format

Create JSON tasks for manual correction in Label Studio:

```bash
python convert_yolo_to_labelstudio.py \
  --image-dir data/images/train \
  --label-dir pseudo_labels/run1/labels \
  --classes classes.txt \
  --output import_ls.json
```

Sample task entry:
```json
{
  "id": 1,
  "data": {"image": "http://.../train/img001.jpg"},
  "annotations": [],
  "predictions": [
    {"result": [ /* bounding box, class */ ]}
  ]
}
```

### Real‑Time Label Editor

Launch a Flask/Streamlit app to edit labels live:

```bash
python realtime_label_editor.py \
  --data-dir data/images/train \
  --labels-dir pseudo_labels/run1/labels \
  --port 8050
```

- Navigate to `http://localhost:8050`  
- Correct bounding boxes or class labels; changes saved instantly.

---

## Model Training

Fine‑tune the YOLOv8 detector on your labeled dataset:

```bash
python trainyolo.py \
  --data config.yaml \
  --epochs 200 \
  --batch 32 \
  --imgsz 640 \
  --device 0 \
  --project yolo_train_run/augmented \
  --name run1
```

- **--data**: path to config YAML  
- **--epochs**: training iterations (default: 200)  
- **--batch**: batch size (default: 16)  
- **--imgsz**: input image size, e.g., 640×640  
- **--device**: GPU index or `cpu`  
- **--project/name**: output directory under `yolo_train_run/`

Check `yolo_train_run/augmented/run1/weights/` for `best.pt` and `last.pt`.

---

## Usage

### Command‑Line Interface: `translator.py`

```bash
usage: translator.py [-h]
  [-i INPUT] [-u URL] [-m MODEL]
  [-l LANG] [-c CONF] [-t THREADS]
  [-o OUTPUT] [--format {png,pdf}]
  [--no-overlay] [--save-json]
```

| Flag           | Description                                                         | Default       |
| -------------- | ------------------------------------------------------------------- | ------------- |
| `-i, --input`  | Path to image file or directory                                     | **None**      |
| `-u, --url`    | URL of a single image (alternative to `--input`)                    | **None**      |
| `-m, --model`  | Path to YOLOv8 `.pt` detector                                       | `yolov8n.pt`  |
| `-l, --lang`   | Target ISO‑639‑1 language code                                      | `en`          |
| `-c, --conf`   | Detection confidence threshold [0.0–1.0]                             | `0.25`        |
| `-t, --threads`| Number of OCR/translation threads                                   | `1`           |
| `-o, --output` | Output directory or file path                                       | `translated`  |
| `--format`     | Output format: `png` or `pdf`                                       | `png`         |
| `--no-overlay` | Skip rendering translated text on images (outputs JSON only)        | *Flag*        |
| `--save-json`  | Save OCR + translation results in JSON alongside images             | *Flag*        |

**Example:**
```bash
python translator.py \
  --input pages/ \
  --model yolo_train_run/augmented/run1/weights/best.pt \
  --lang fr --conf 0.3 --threads 4 \
  --output outputs/ --format pdf --save-json
```

### Web Interface: `page_upload.py`

```bash
streamlit run page_upload.py -- \
  --model yolo_train_run/augmented/run1/weights/best.pt \
  --device cuda \
  --port 8503
```

1. **Open** `http://localhost:8503` in a browser.  
2. **Upload** files or paste image URLs.  
3. **Adjust** confidence slider and choose target language.  
4. **Inspect** detection, OCR results, and final overlay.  
5. **Download** translated pages or JSON logs.

---

## Utility Scripts

| Script                                   | Purpose                                                           |
| ---------------------------------------- | ----------------------------------------------------------------- |
| `counts.py`                              | Print class-wise label counts and save histogram plots.          |
| `create_pseudo_labels.py`                | Auto-generate YOLO labels on unlabeled datasets.                 |
| `convert_yolo_to_labelstudio.py`         | Convert YOLO `.txt` labels into Label Studio import JSON.        |
| `realtime_label_editor.py`               | Launch live editor for correcting YOLO labels.                   |
| `export_translations.py`                 | Gather all `.json` result files into a single CSV summary.       |

Run any script with `-h/--help` for detailed arguments.

---

## Project Structure

```text
translation_tool/
├── config.yaml                  # YOLO data config and class mapping
├── classes.txt                  # Newline-separated class names
├── requirements.txt             # Python package requirements
├── data/                        # Raw images and YOLO labels
│   ├── images/{train,val}/
│   └── labels/{train,val}/
├── pseudo_labels/               # Auto-generated pseudo-labels
├── convert_yolo_to_labelstudio.py  # YOLO→Label Studio conversion
├── create_pseudo_labels.py      # Generate pseudo-labels via YOLOv8
├── realtime_label_editor.py     # Live annotation editor
├── trainyolo.py                 # YOLOv8 training script
├── yolo_train_run/              # Training outputs (weights, logs)
├── translator.py                # CLI detection/OCR/translation pipeline
├── export_translations.py       # Compile JSON results into CSV
├── page_upload.py               # Streamlit web app for interactive translation
├── counts.py                    # Label distribution and diagnostics
├── test/                        # `pytest` unit tests for core modules
└── translated/                  # Default output folder for translations
```

---

## Testing

1. **Install test dependencies**:
   ```bash
   pip install pytest pytest-cov
   ```
2. **Run tests**:
   ```bash
   pytest test/ --maxfail=1 --disable-warnings -q
   ```
3. **Generate coverage report**:
   ```bash
   pytest --cov=translator --cov-report=html
   open htmlcov/index.html
   ```

All core functions (detection, OCR, overlay, translation) must pass.

---

## Contributing

We welcome contributions! Please adhere to these guidelines:

- **Branch naming**: `feature/<short-description>`, `bugfix/<short-description>`.
- **Commits**: Use imperative mood, e.g., `Add JSON export for translations`.
- **Code style**: Run `black .` and `flake8` before committing.
- **Pull requests**: Include screenshots for UI changes, sample outputs for script updates.
- **Review process**: Automated CI will run tests and linting; maintain 80%+ coverage.

1. Fork the repo.
2. Create a feature branch.
3. Commit changes and push.
4. Open a PR against `main` and request reviews.
5. Address review comments; merge when green.

---

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). You are free to use, modify, and distribute it, provided that you give appropriate credit to Chanakya Bhaskara. See [LICENSE](./LICENSE) for full text.

---

## Contact

For questions, feedback, or contributions, reach out to Chanakya Bhaskara at <Chanakyabhaskara@gmail.com>.

&copy; 2025 ABChanakya. All rights reserved.

