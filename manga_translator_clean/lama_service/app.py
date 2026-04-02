# lama_service/app.py
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from simple_lama_inpainting import SimpleLama
from PIL import Image
import io
try:
    import torch  # type: ignore[import]
except ImportError:  # pragma: no cover - torch always available inside service venv
    torch = None

# Initialize Flask app
app = Flask(__name__)

# Resolve device and variant at startup (no GPU memory allocated yet)
requested_device = os.environ.get("LAMA_DEVICE", "auto").lower()
MODEL_VARIANT = os.environ.get("LAMA_MODEL_VARIANT", "big-lama")

device = "cpu"
if torch is not None:
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested_device)
    except (TypeError, ValueError) as err:
        print(f"⚠️ Invalid LAMA_DEVICE '{requested_device}', falling back to CPU ({err})")
        device = torch.device("cpu")
else:
    if requested_device not in {"cpu", "auto"}:
        print("⚠️ Torch unavailable; forcing CPU for LaMa service.")
    device = "cpu"

print(f"🖥️  LaMa target device: {device}  (model will load on first inpaint request)")
print(f"🎨 LaMa model variant: {MODEL_VARIANT}")

# Lazy-loaded model — None until first /inpaint request
lama = None
_lama_load_error: str | None = None  # set if model failed to load


def _resolve_lama_model_path():
    """Download dreMaz/AnimeMangaInpainting checkpoint when LAMA_MODEL_VARIANT=anime-manga."""
    if MODEL_VARIANT == "anime-manga":
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="dreMaz/AnimeMangaInpainting",
                filename="big-lama.pt",
            )
            print(f"✅ Loaded dreMaz/AnimeMangaInpainting: {path}")
            return path
        except Exception as e:
            print(f"⚠️  dreMaz download failed ({e}), falling back to default big-lama")
    print("✅ Using default big-lama checkpoint")
    return None


def _ensure_lama_loaded():
    """Load the LaMa model on first call. No-op if already loaded."""
    global lama, _lama_load_error
    if lama is not None:
        return True
    if _lama_load_error is not None:
        return False
    print("🎨 Loading LaMa model on first request...")
    try:
        lama = SimpleLama(device=device if torch is not None else None)
        print("✅ LaMa model loaded successfully.")
        return True
    except Exception as e:
        _lama_load_error = str(e)
        print(f"❌ Error loading LaMa model: {e}")
        return False

@app.route('/', methods=['GET'])
def index():
    model_ok = lama is not None
    model_err = _lama_load_error
    device_label = str(device)
    if model_err:
        status_color = '#e74c3c'
        status_text = f'Load error: {model_err[:80]}'
    elif model_ok:
        status_color = '#2ecc71'
        status_text = 'Model ready'
    else:
        status_color = '#f39c12'
        status_text = 'Waiting for first request (lazy load)'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LaMa Inpainting Service</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee;
           display: flex; align-items: center; justify-content: center;
           min-height: 100vh; margin: 0; }}
    .card {{ background: #16213e; border-radius: 12px; padding: 2.5rem 3rem;
             box-shadow: 0 8px 32px rgba(0,0,0,0.4); max-width: 480px; width: 100%; }}
    h1 {{ margin: 0 0 0.25rem; font-size: 1.5rem; color: #4a90e2; }}
    .subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 2rem; }}
    .badge {{ display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px;
              background: {status_color}22; color: {status_color}; border: 1px solid {status_color}55;
              font-weight: 600; font-size: 0.85rem; margin-bottom: 1.5rem; }}
    .row {{ display: flex; justify-content: space-between; padding: 0.6rem 0;
            border-bottom: 1px solid #ffffff11; font-size: 0.9rem; }}
    .row:last-child {{ border-bottom: none; }}
    .label {{ color: #888; }}
    .value {{ color: #eee; font-weight: 500; }}
    .endpoints {{ margin-top: 1.5rem; }}
    .ep {{ background: #0f3460; border-radius: 6px; padding: 0.4rem 0.75rem;
           font-family: monospace; font-size: 0.85rem; color: #64b5f6;
           display: inline-block; margin: 0.2rem 0.2rem 0 0; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>🎨 LaMa Inpainting Service</h1>
    <p class="subtitle">Manga text removal microservice</p>
    <div class="badge">{status_text}</div>
    <div class="row"><span class="label">Device</span><span class="value">{device_label}</span></div>
    <div class="row"><span class="label">Model</span><span class="value">{MODEL_VARIANT}</span></div>
    <div class="row"><span class="label">Port</span><span class="value">5001</span></div>
    <div class="row"><span class="label">Main app</span>
      <span class="value"><a href="http://127.0.0.1:5000" style="color:#4a90e2">localhost:5000</a></span></div>
    <div class="endpoints">
      <div class="label" style="margin-bottom:0.5rem">Endpoints</div>
      <span class="ep">GET /health</span>
      <span class="ep">POST /inpaint</span>
    </div>
  </div>
</body>
</html>""", 200


@app.route('/inpaint', methods=['POST'])
def inpaint_image():
    """
    Inpaint an image based on a provided mask.
    Expects a multipart/form-data request with 'image' and 'mask' files.

    Optional query params:
      dilate=N  — dilate the mask by N pixels (default 4) to ensure all
                  text ink at mask edges is covered before inpainting.
    """
    if not _ensure_lama_loaded():
        return jsonify({'error': f'LaMa model failed to load: {_lama_load_error}'}), 500

    if 'image' not in request.files or 'mask' not in request.files:
        return jsonify({'error': 'Missing image or mask file.'}), 400

    dilate_px = int(request.args.get('dilate', 4))

    # Read image and mask from the request
    image_file = request.files['image'].read()
    mask_file = request.files['mask'].read()

    # Decode image and mask
    image = cv2.imdecode(np.frombuffer(image_file, np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.frombuffer(mask_file, np.uint8), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        return jsonify({'error': 'Could not decode image or mask'}), 400

    # Ensure mask is binary (0 or 255)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Dilate the mask so ink at box edges is included in the inpainted area.
    # This prevents faint character strokes from remaining visible after
    # inpainting when the YOLO box is slightly too tight.
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    print(f"🎨 Image {image.shape}, mask {mask.shape}, dilate={dilate_px}px")

    # Perform inpainting
    try:
        inpainted_image = lama(image, mask)
        print("✅ Inpainting complete.")
    except Exception as e:
        print(f"❌ Error during inpainting: {e}")
        return jsonify({'error': f'Inpainting failed: {e}'}), 500

    # Normalize the output into a PIL Image
    if isinstance(inpainted_image, Image.Image):
        inpainted_image_pil = inpainted_image
    else:
        inpainted_image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))

    buf = io.BytesIO()
    inpainted_image_pil.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint. Returns model_loaded=True as long as there's no
    load error — the model loads lazily on the first /inpaint request.
    """
    if _lama_load_error:
        return jsonify({'status': 'error', 'model_loaded': False, 'error': _lama_load_error}), 500
    return jsonify({'status': 'ok', 'model_loaded': True}), 200

if __name__ == '__main__':
    # Run the app
    # Use Gunicorn in a real deployment
    app.run(host='0.0.0.0', port=5001, debug=True)
