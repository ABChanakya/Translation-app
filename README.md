# Translation-App STILL IN PROGRESS

**Translation-App** is a toolkit for detecting, OCR‑ing, and translating speech bubbles in manga/manhwa pages. It combines a custom YOLOv8 detection model with Manga‑OCR for text recognition and provides both a CLI and web interface for easy use.

---

## 🚀 Features

- **Speech Bubble Detection**: Custom YOLOv8 model to detect speech bubbles and sound effects.
- **Deskew & Crop**: Automatic deskewing and cropping of detected regions for optimal OCR input.
- **Enhanced OCR**: Utilizes [Manga‑OCR](https://github.com/NVlabs/Manga-OCR) for high‑accuracy Japanese text recognition.
- **Dummy Translation**: Quickly preview translated text (prefixes recognized text with `[→<lang>]`).
- **Overlay Translations**: Renders translated text back onto the original page.
- **Interactive Web UI**: Streamlit app for drag‑and‑drop translation workflows.
- **Unit Tested**: Core utilities covered by `unittest` in `translator.py`.

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ABChanakya/Tranlation-app.git
   cd Tranlation-app
   ```

2. **Create & activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\\Scripts\\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

> **Note**: Ensure you have CUDA drivers installed if you plan to use GPU acceleration.

---
#STILL WROK IN PROGRESS
## 🛠 CLI Usage (`translator.py`)

```bash
python translator.py \
  --input /path/to/page.jpg \
  --model /path/to/best.pt \
  --lang en \
  --output translated_page.png
```

**Arguments**:
- `--input` Path to a local image file. Mutually exclusive with `--url`.
- `--url`  URL of an image to process.
- `--model` Path to your YOLOv8 detection model checkpoint (`.pt`).
- `--lang`  Target ISO language code (default: `en`).
- `--output` Output image file (default: `translated.png`).
- `--conf`  Detection confidence threshold (default: `0.25`).
- `--test`  Run built‑in unit tests and exit.

**Example**:
```bash
python translator.py --input page.jpg --model yolov8n.pt --lang ja --output page_ja.png
```

---

## 🌐 Web Interface (`page_upload.py`)

Launch the Streamlit app for an interactive UI:

```bash
streamlit run page_upload.py -- --device cuda --model /path/to/best.pt
```

- **Device**: `cuda` or `cpu` (auto‑detects if not provided).
- **Model**: Path to YOLOv8 `.pt` file.

Once started, visit `http://localhost:8503` in your browser to upload pages and see translations in real time.

---

## 🏋️‍♂️ Model Training (`trainyolo.py`)

Customize and retrain the speech bubble detector on your own dataset:

1. Prepare your dataset directory with `images/` and `labels/` in YOLO format.
2. Update `config.yaml` with your `train`/`val` paths and class names.
3. Run training:
   ```bash
   python trainyolo.py
   ```
4. Trained weights and logs will be saved under `yolo_train_run/train/weights`.

---

## 📁 Project Structure

```
├── config.yaml            # YOLOv8 data configuration
├── data/                  # Sample images & labels
├── page_upload.py         # Streamlit web app
├── translator.py          # CLI OCR & translation pipeline
├── trainyolo.py           # YOLOv8 training script
├── requirements.txt       # Python dependencies
└── yolo_train_run/        # Training outputs (runs, weights, logs)
```

---

## ✉️ Contact & Contribution

For questions or feature requests, open an [issue](https://github.com/ABChanakya/Tranlation-app/issues) or email at **Chanakyabahskara@gmail.com**.

Contributions are welcome! Feel free to fork, fix bugs, and submit pull requests.

---

&copy; 2025 ABChanakya. All rights reserved.
