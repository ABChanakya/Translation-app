"""Flask web application for Auto Manga Translation."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings
from src.batch_processor import BatchProcessor
from src.colorization_service import colorize_image, get_colorization_status
from src.pipeline import MangaTranslationPipeline
from src.progress_tracker import ProcessingStage, ProgressUpdate, get_global_tracker
from src.translators.registry import get_default_engine_status, list_engine_statuses

PRODUCT_NAME = "Auto Manga Translation"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "auto-manga-translation-dev-secret")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(PROJECT_ROOT / "web" / "uploads")
app.config["OUTPUT_FOLDER"] = str(PROJECT_ROOT / "web" / "outputs")
app.config["BATCH_FOLDER"] = str(PROJECT_ROOT / "web" / "batch_outputs")
app.config["ADMIN_JOBS_FOLDER"] = str(PROJECT_ROOT / "web" / "admin_jobs")

for key in ("UPLOAD_FOLDER", "OUTPUT_FOLDER", "BATCH_FOLDER", "ADMIN_JOBS_FOLDER"):
    Path(app.config[key]).mkdir(parents=True, exist_ok=True)

settings = Settings()
pipeline = None
batch_processor = BatchProcessor(output_dir=app.config["BATCH_FOLDER"])
progress_tracker = get_global_tracker()
progress_queues: dict[str, queue.Queue] = {}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
DEFAULT_BATCH_CHUNK_SIZE = 8


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def normalize_training_device(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return "cpu"

    if raw == "cpu":
        return "cpu"

    if raw in {"gpu", "cuda"}:
        return "0" if _cuda_available() else "cpu"

    if raw.startswith("cuda:"):
        suffix = raw.split(":", 1)[1].strip()
        if suffix.isdigit():
            return suffix if _cuda_available() else "cpu"

    if all(part.strip().isdigit() for part in raw.split(",")):
        return raw if _cuda_available() else "cpu"

    return raw


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def admin_credentials_configured() -> bool:
    return bool(os.getenv("ADMIN_USERNAME") and os.getenv("ADMIN_PASSWORD_HASH"))


def is_admin_authenticated() -> bool:
    return bool(session.get("admin_authenticated"))


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not admin_credentials_configured():
            return render_template("admin_login.html", configuration_missing=True), 503
        if not is_admin_authenticated():
            return redirect(url_for("admin_login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def resolve_admin_path(value: str | Path) -> Path:
    if not value:
        raise ValueError("Missing path value")
    return Path(value).expanduser().resolve()


def send_progress(session_id: str, update: ProgressUpdate) -> None:
    if session_id in progress_queues:
        progress_queues[session_id].put(update)


def close_progress(session_id: str) -> None:
    if session_id in progress_queues:
        progress_queues[session_id].put(None)


def collect_models() -> list[dict[str, Any]]:
    search_roots = [
        PROJECT_ROOT / "models" / "checkpoints",
        PROJECT_ROOT / "yolo_train_run",
    ]
    models: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for model_path in sorted(root.rglob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True):
            resolved = model_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            models.append(
                {
                    "name": model_path.name,
                    "path": str(resolved),
                    "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
                    "modified_at": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )
    return models


def collect_datasets() -> list[dict[str, str]]:
    datasets_dir = PROJECT_ROOT / "training" / "datasets"
    if not datasets_dir.exists():
        return []
    return [
        {"name": dataset.name, "path": str(dataset.resolve())}
        for dataset in sorted(datasets_dir.glob("*.yaml"))
    ]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_reports(limit: int = 20) -> list[dict[str, Any]]:
    reports_dir = PROJECT_ROOT / "evaluation" / "results"
    if not reports_dir.exists():
        return []

    reports = []
    for summary_path in reports_dir.rglob("summary.json"):
        try:
            payload = _load_json(summary_path)
        except Exception:
            continue
        if "aggregate_metrics" not in payload:
            payload["aggregate_metrics"] = {
                "mAP50": float(payload.get("mAP50", 0.0)),
                "mAP50_95": float(payload.get("mAP50-95", payload.get("mAP50_95", 0.0))),
                "macro_recall": float(payload.get("macro_recall", payload.get("recall", 0.0))),
                "macro_f1": float(payload.get("macro_f1", payload.get("f1", 0.0))),
                "macro_iou": float(payload.get("macro_iou", payload.get("mIoU", 0.0))),
            }
        reports.append(
            {
                "summary": payload,
                "summary_path": str(summary_path.resolve()),
                "report_html": payload.get("report_html"),
                "metrics_json": payload.get("metrics_json"),
                "fairness_json": payload.get("fairness_json"),
                "timestamp": payload.get("timestamp"),
                "name": summary_path.parent.name,
            }
        )
    reports.sort(key=lambda item: item["timestamp"] or "", reverse=True)
    return reports[:limit]


def collect_admin_jobs(limit: int = 30) -> list[dict[str, Any]]:
    jobs_dir = Path(app.config["ADMIN_JOBS_FOLDER"])
    jobs = []
    for manifest_path in sorted(jobs_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            jobs.append(_load_json(manifest_path))
        except Exception:
            continue
    return jobs[:limit]


def collect_pseudo_label_manifests(limit: int = 20) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for root in (PROJECT_ROOT / "training", PROJECT_ROOT / "data", PROJECT_ROOT / "web"):
        if not root.exists():
            continue
        for manifest_path in root.rglob("review_manifest.json"):
            manifest_resolved = manifest_path.resolve()
            if manifest_resolved in seen_paths:
                continue
            try:
                payload = _load_json(manifest_path)
            except Exception:
                continue
            seen_paths.add(manifest_resolved)
            payload["_manifest_path"] = str(manifest_path.resolve())
            payload["_item_count"] = len(payload.get("items", []))
            candidates.append(payload)

    for job in collect_admin_jobs(limit=100):
        output_dir = job.get("metadata", {}).get("output_dir")
        if not output_dir:
            continue
        manifest_path = Path(output_dir).expanduser().resolve() / "review_manifest.json"
        if not manifest_path.exists() or manifest_path in seen_paths:
            continue
        try:
            payload = _load_json(manifest_path)
        except Exception:
            continue
        seen_paths.add(manifest_path)
        payload["_manifest_path"] = str(manifest_path)
        payload["_item_count"] = len(payload.get("items", []))
        candidates.append(payload)

    candidates.sort(key=lambda item: item.get("completed_at") or item.get("created_at") or "", reverse=True)
    return candidates[:limit]


def load_manifest(manifest_path: str | None) -> dict[str, Any] | None:
    if not manifest_path:
        return None
    path = resolve_admin_path(manifest_path)
    if not path.exists():
        return None
    payload = _load_json(path)
    payload["_manifest_path"] = str(path)
    return payload


def launch_admin_job(job_type: str, label: str, command: list[str], metadata: dict[str, Any]) -> dict[str, Any]:
    jobs_dir = Path(app.config["ADMIN_JOBS_FOLDER"])
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    manifest_path = jobs_dir / f"{job_id}.json"
    log_path = jobs_dir / f"{job_id}.log"

    payload = {
        "job_id": job_id,
        "job_type": job_type,
        "label": label,
        "command": command,
        "metadata": metadata,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "log_path": str(log_path.resolve()),
        "cwd": str(PROJECT_ROOT),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # This is intentionally a lightweight local background-job layer. If the
    # project grows, replace this subprocess launcher with a queue-backed or
    # storage-backed worker system rather than building a second parallel flow.
    runner_path = PROJECT_ROOT / "web" / "admin_job_runner.py"
    subprocess.Popen(
        [sys.executable, str(runner_path), str(manifest_path), "--", *command],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return payload


@app.context_processor
def inject_globals() -> dict[str, Any]:
    return {
        "product_name": PRODUCT_NAME,
        "admin_authenticated": is_admin_authenticated(),
        "colorization_status": get_colorization_status(),
    }


@app.route("/")
def index():
    reports = collect_reports(limit=5)
    return render_template("index.html", reports=reports)


@app.route("/translate")
def translate_page():
    return render_template(
        "translate.html",
        initial_mode=request.args.get("mode", "single"),
        default_engine=get_default_engine_status(),
    )


@app.route("/colorize")
def colorize_page():
    return render_template("colorize.html", colorization_status=get_colorization_status())


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/train")
def train_redirect():
    return redirect(url_for("admin_training"))


@app.route("/annotate")
def annotate_redirect():
    return redirect(url_for("admin_data"))


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if is_admin_authenticated():
        return redirect(url_for("admin_dashboard"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        configured_username = os.getenv("ADMIN_USERNAME", "")
        password_hash = os.getenv("ADMIN_PASSWORD_HASH", "")

        if not admin_credentials_configured():
            error = "Admin credentials are not configured. Set ADMIN_USERNAME and ADMIN_PASSWORD_HASH first."
        elif username == configured_username and check_password_hash(password_hash, password):
            session["admin_authenticated"] = True
            session["admin_username"] = username
            return redirect(request.args.get("next") or url_for("admin_dashboard"))
        else:
            error = "Invalid admin credentials."

    return render_template("admin_login.html", error=error, configuration_missing=not admin_credentials_configured())


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    engine_statuses = list_engine_statuses(scope="admin")
    reports = collect_reports(limit=6)
    jobs = collect_admin_jobs(limit=8)
    manifests = collect_pseudo_label_manifests(limit=6)
    return render_template(
        "admin/dashboard.html",
        engine_statuses=engine_statuses,
        reports=reports,
        jobs=jobs,
        manifests=manifests,
        default_dataset=settings.DEFAULT_DATASET_YAML,
        default_model=settings.YOLO_MODEL_PATH,
    )


@app.route("/admin/training")
@admin_required
def admin_training():
    return render_template(
        "admin/training.html",
        models=collect_models(),
        datasets=collect_datasets(),
        jobs=collect_admin_jobs(limit=12),
        reports=collect_reports(limit=12),
        default_model=settings.YOLO_MODEL_PATH,
        default_dataset=settings.DEFAULT_DATASET_YAML,
    )


@app.route("/admin/evaluation")
@admin_required
def admin_evaluation():
    return render_template(
        "admin/evaluation.html",
        models=collect_models(),
        datasets=collect_datasets(),
        jobs=[job for job in collect_admin_jobs(limit=20) if job.get("job_type") == "evaluation"],
        reports=collect_reports(limit=20),
        default_dataset=settings.DEFAULT_DATASET_YAML,
    )


@app.route("/admin/data")
@admin_required
def admin_data():
    selected_manifest = load_manifest(request.args.get("manifest"))
    return render_template(
        "admin/data.html",
        models=collect_models(),
        manifests=collect_pseudo_label_manifests(limit=30),
        selected_manifest=selected_manifest,
        jobs=[job for job in collect_admin_jobs(limit=20) if job.get("job_type") == "pseudo_labels"],
        classes_path=str((PROJECT_ROOT / "training" / "datasets" / "classes.txt").resolve()),
    )


@app.route("/admin/engines")
@admin_required
def admin_engines():
    return render_template("admin/engines.html", engine_statuses=list_engine_statuses(scope="admin"))


@app.route("/admin/models")
@admin_required
def admin_models():
    return render_template(
        "admin/models.html",
        models=collect_models(),
        datasets=collect_datasets(),
        reports=collect_reports(limit=10),
        yolo_model_path=settings.YOLO_MODEL_PATH,
    )


@app.route("/admin/files")
@admin_required
def admin_file():
    requested_path = request.args.get("path")
    if not requested_path:
        return jsonify({"error": "Missing file path"}), 400
    try:
        path = resolve_admin_path(requested_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(path)


@app.route("/admin/training/launch", methods=["POST"])
@admin_required
def admin_launch_training():
    entrypoint = request.form.get("entrypoint", "train_yolo")
    script_path = PROJECT_ROOT / "training" / f"{entrypoint}.py"
    if not script_path.exists():
        return jsonify({"error": f"Unknown training entrypoint: {entrypoint}"}), 400

    device = normalize_training_device(request.form.get("device", "cpu"))

    command = [
        sys.executable,
        str(script_path),
        "--model",
        request.form.get("model", settings.YOLO_MODEL_PATH),
        "--data",
        request.form.get("data", settings.DEFAULT_DATASET_YAML),
        "--imgsz",
        request.form.get("imgsz", "640"),
        "--batch" if entrypoint == "train_yolo" else "--batch-size",
        request.form.get("batch_size", "8"),
    ]

    if entrypoint == "train_yolo":
        command.extend(
            [
                "--epochs",
                request.form.get("epochs", "30"),
                "--device",
                device,
                "--project",
                request.form.get("project", "yolo_train_run"),
            ]
        )
        if request.form.get("name"):
            command.extend(["--name", request.form["name"]])
    else:
        command.extend(
            [
                "--device",
                device,
                "--project",
                request.form.get("project", "yolo_train_run"),
                "--match-iou",
                request.form.get("match_iou", "0.6"),
            ]
        )
        if entrypoint == "advanced_train_yolo":
            command.extend(
                [
                    "--head-epochs",
                    request.form.get("head_epochs", "10"),
                    "--full-epochs",
                    request.form.get("full_epochs", "30"),
                ]
            )
        else:
            command.extend(
                [
                    "--head-epochs",
                    request.form.get("head_epochs", "8"),
                    "--full-epochs",
                    request.form.get("full_epochs", "24"),
                    "--no-mosaic-epochs",
                    request.form.get("no_mosaic_epochs", "8"),
                ]
            )

    launch_admin_job(
        "training",
        f"Training via {entrypoint}",
        command,
        {
            "entrypoint": entrypoint,
            "model": request.form.get("model", settings.YOLO_MODEL_PATH),
            "data": request.form.get("data", settings.DEFAULT_DATASET_YAML),
            "device": device,
        },
    )
    return redirect(url_for("admin_training"))


@app.route("/admin/evaluation/launch", methods=["POST"])
@admin_required
def admin_launch_evaluation():
    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluation" / "evaluate_model.py"),
        "--model",
        request.form.get("model", settings.YOLO_MODEL_PATH),
        "--data",
        request.form.get("data", settings.DEFAULT_DATASET_YAML),
        "--split",
        request.form.get("split", "val"),
        "--conf",
        request.form.get("conf", "0.001"),
        "--iou",
        request.form.get("iou", "0.6"),
        "--match-iou",
        request.form.get("match_iou", "0.6"),
        "--batch",
        request.form.get("batch", "8"),
        "--imgsz",
        request.form.get("imgsz", "640"),
        "--save-dir",
        request.form.get("save_dir", "evaluation/results"),
    ]
    launch_admin_job(
        "evaluation",
        "Evaluation",
        command,
        {"model": request.form.get("model"), "data": request.form.get("data")},
    )
    return redirect(url_for("admin_evaluation"))


@app.route("/admin/data/launch", methods=["POST"])
@admin_required
def admin_launch_pseudo_labels():
    command = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "create_pseudo_labels.py"),
        "--input",
        request.form.get("input_dir", ""),
        "--output",
        request.form.get("output_dir", ""),
        "--model",
        request.form.get("model", settings.YOLO_MODEL_PATH),
        "--conf",
        request.form.get("conf", "0.25"),
        "--iou",
        request.form.get("iou", "0.55"),
        "--imgsz",
        request.form.get("imgsz", "640"),
        "--batch-size",
        request.form.get("batch_size", "4"),
        "--chunk-size",
        request.form.get("chunk_size", "32"),
        "--names",
        request.form.get("names_path", str((PROJECT_ROOT / "training" / "datasets" / "classes.txt").resolve())),
    ]
    if request.form.get("enable_ocr"):
        command.append("--enable-ocr")

    launch_admin_job(
        "pseudo_labels",
        "Assisted data generation",
        command,
        {
            "input_dir": request.form.get("input_dir"),
            "output_dir": request.form.get("output_dir"),
            "model": request.form.get("model"),
        },
    )
    return redirect(url_for("admin_data"))


@app.route("/admin/data/review/save", methods=["POST"])
@admin_required
def admin_save_review_manifest():
    manifest_path = request.form.get("manifest_path")
    manifest = load_manifest(manifest_path)
    if not manifest:
        return jsonify({"error": "Manifest not found"}), 404

    for item in manifest.get("items", []):
        image_id = item.get("image_id")
        item["review_status"] = request.form.get(f"status_{image_id}", item.get("review_status", "uncertain"))
        item["notes"] = request.form.get(f"notes_{image_id}", item.get("notes", ""))

    manifest["updated_at"] = datetime.now().isoformat()
    Path(manifest["_manifest_path"]).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return redirect(url_for("admin_data", manifest=manifest["_manifest_path"]))


@app.route("/admin/data/review/apply", methods=["POST"])
@admin_required
def admin_apply_review_manifest():
    manifest_path = request.form.get("manifest_path")
    if not manifest_path:
        return jsonify({"error": "Missing manifest path"}), 400

    command = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "apply_pseudo_label_review.py"),
        "--manifest",
        manifest_path,
        "--reject-action",
        request.form.get("reject_action", "remove"),
        "--rejected-dir",
        request.form.get("rejected_dir", "training/rejected_pseudo_labels"),
    ]
    launch_admin_job(
        "pseudo_label_review",
        "Apply pseudo-label review",
        command,
        {"manifest_path": manifest_path},
    )
    return redirect(url_for("admin_data", manifest=manifest_path))


@app.route("/api/progress/<session_id>")
def progress_stream(session_id: str):
    def generate():
        q = queue.Queue()
        progress_queues[session_id] = q
        try:
            while True:
                try:
                    update = q.get(timeout=30)
                    if update is None:
                        break
                    yield update.to_sse()
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            progress_queues.pop(session_id, None)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/engines", methods=["GET"])
def list_engines():
    public_engines = list_engine_statuses(scope="public")
    default_engine = get_default_engine_status()
    return jsonify(
        {
            "engines": [
                {
                    "engine_id": status.engine_id,
                    "label": status.label,
                    "factory_name": status.factory_name,
                }
                for status in public_engines
            ],
            "default_engine": default_engine.engine_id if default_engine else None,
        }
    )


@app.route("/admin/api/engines", methods=["GET"])
@admin_required
def admin_list_engines():
    return jsonify(
        {
            "engines": [
                {
                    "engine_id": status.engine_id,
                    "label": status.label,
                    "factory_name": status.factory_name,
                    "enabled": status.enabled,
                    "implemented": status.implemented,
                    "disable_reason": status.disable_reason,
                    "enable_instructions": status.enable_instructions,
                }
                for status in list_engine_statuses(scope="admin")
            ]
        }
    )


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
        filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
        file.save(filepath)
        return jsonify({"success": True, "filename": filename, "filepath": str(filepath.resolve())})

    return jsonify({"error": "Invalid file type"}), 400


@app.route("/api/translate", methods=["POST"])
def translate_image():
    global pipeline

    try:
        data = request.get_json() or {}
        input_path = data.get("input_path")
        target_lang = data.get("target_lang", "en")
        translator_type = data.get("translator", "gemma3")
        session_id = data.get("session_id", "default")
        confidence = float(data.get("confidence", settings.DEFAULT_CONFIDENCE))
        iou_threshold = float(data.get("iou_threshold", settings.DEFAULT_IOU_THRESHOLD))
        story_context = data.get("story_context", None)
        vlm_context = bool(data.get("vlm_context", False))

        if not input_path or not Path(input_path).exists():
            return jsonify({"error": "Invalid input path"}), 400

        send_progress(
            session_id,
            ProgressUpdate(stage=ProcessingStage.UPLOADING.value, progress=5.0, message="Initializing pipeline..."),
        )

        pipeline = MangaTranslationPipeline(
            source_lang="ja",
            target_lang=target_lang,
            translation_engine=translator_type,
            detection_confidence=confidence,
            nms_iou_threshold=iou_threshold,
            text_color="#000000",
            story_context=story_context,
            vlm_context_enabled=vlm_context,
        )

        output_filename = f"translated_{Path(input_path).name}"
        output_path = Path(app.config["OUTPUT_FOLDER"]) / output_filename
        send_progress(
            session_id,
            ProgressUpdate(stage=ProcessingStage.DETECTING.value, progress=20.0, message="Detecting text bubbles..."),
        )
        result = pipeline.process_image(input_path, str(output_path))
        send_progress(
            session_id,
            ProgressUpdate(stage=ProcessingStage.COMPLETE.value, progress=100.0, message="Translation complete!"),
        )
        close_progress(session_id)

        result["confidence"] = confidence
        result["iou_threshold"] = iou_threshold

        return jsonify(
            {
                "success": True,
                "output_path": str(output_path),
                "output_url": url_for("get_output", filename=output_filename),
                "stats": result,
            }
        )
    except Exception as exc:
        session_id = (request.get_json() or {}).get("session_id", "default")
        send_progress(
            session_id,
            ProgressUpdate(stage=ProcessingStage.ERROR.value, progress=0.0, message=f"Error: {exc}"),
        )
        close_progress(session_id)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/colorize/status", methods=["GET"])
def colorize_status():
    return jsonify(get_colorization_status())


@app.route("/api/colorize", methods=["POST"])
def colorize_image_api():
    try:
        data = request.get_json() or {}
        input_path = data.get("input_path")
        if not input_path or not Path(input_path).exists():
            return jsonify({"error": "Invalid input path"}), 400

        status = get_colorization_status()
        if not status["available"]:
            return jsonify({"error": "Colorization is not configured", "status": status}), 503

        output_filename = f"colorized_{Path(input_path).stem}.png"
        output_path = Path(app.config["OUTPUT_FOLDER"]) / output_filename
        colorize_image(
            input_path,
            output_path,
            size=int(data.get("size", 576)),
            denoiser=bool(data.get("denoiser", True)),
            denoiser_sigma=int(data.get("denoiser_sigma", 25)),
        )

        return jsonify(
            {
                "success": True,
                "output_path": str(output_path),
                "output_url": url_for("get_output", filename=output_filename),
                "status": status,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc), "status": get_colorization_status()}), 500


@app.route("/api/models", methods=["GET"])
def list_models_api():
    return jsonify({"models": collect_models()})


@app.route("/api/datasets", methods=["GET"])
def list_datasets_api():
    return jsonify({"datasets": collect_datasets()})


@app.route("/api/batch/upload", methods=["POST"])
def batch_upload():
    if "files[]" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files[]")
    if not files:
        return jsonify({"error": "No files selected"}), 400

    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
            filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
            file.save(filepath)
            uploaded_files.append({"filename": filename, "filepath": str(filepath.resolve())})
        else:
            return jsonify({"error": f"Invalid file: {file.filename}"}), 400

    return jsonify({"success": True, "files": uploaded_files, "count": len(uploaded_files)})


@app.route("/api/batch/translate", methods=["POST"])
def batch_translate():
    global pipeline

    try:
        data = request.get_json() or {}
        file_paths = data.get("file_paths", [])
        if not file_paths:
            return jsonify({"error": "No files provided"}), 400

        target_lang = data.get("target_lang", "en")
        translator_type = data.get("translator", "gemma3")
        confidence = float(data.get("confidence", settings.DEFAULT_CONFIDENCE))
        iou_threshold = float(data.get("iou_threshold", settings.DEFAULT_IOU_THRESHOLD))
        output_format = data.get("output_format", "zip")
        include_originals = bool(data.get("include_originals", True))
        chunk_size = int(data.get("chunk_size", DEFAULT_BATCH_CHUNK_SIZE))
        story_context = data.get("story_context", None)  # Optional global story context
        vlm_context = bool(data.get("vlm_context", False))

        pipeline = MangaTranslationPipeline(
            source_lang="ja",
            target_lang=target_lang,
            translation_engine=translator_type,
            detection_confidence=confidence,
            nms_iou_threshold=iou_threshold,
            text_color="#000000",
            story_context=story_context,
            vlm_context_enabled=vlm_context,
        )

        def process_single_page(input_path: str, output_path: str, story_context: str = None, **kwargs):
            return pipeline.process_image(input_path, output_path, **kwargs)

        batch_result = batch_processor.process_batch(
            input_paths=file_paths,
            process_func=process_single_page,
            chunk_size=chunk_size,
            story_context=story_context,  # Pass to batch processor
        )

        if batch_result["processed"] == 0:
            return jsonify({
                "error": "All pages failed to process. "
                         "Try raising the confidence threshold or check your images.",
                "page_errors": batch_result["errors"][:5],
            }), 422

        output_files = {}
        if output_format in {"zip", "both"}:
            try:
                zip_path = batch_processor.create_zip(batch_result, include_originals=include_originals)
                output_files["zip"] = {
                    "path": zip_path,
                    "url": url_for("get_batch_output", filename=Path(zip_path).name),
                    "filename": Path(zip_path).name,
                }
            except Exception as zip_err:
                output_files["zip"] = {"error": f"ZIP creation failed: {zip_err}"}

        if output_format in {"pdf", "both"}:
            try:
                pdf_path = batch_processor.create_pdf(batch_result, include_originals=include_originals)
                if pdf_path:
                    output_files["pdf"] = {
                        "path": pdf_path,
                        "url": url_for("get_batch_output", filename=Path(pdf_path).name),
                        "filename": Path(pdf_path).name,
                    }
                else:
                    output_files["pdf"] = {"error": "PDF generation not available (install reportlab)"}
            except Exception as pdf_err:
                output_files["pdf"] = {"error": f"PDF creation failed: {pdf_err}"}

        # Only clean up temp files after outputs are built — temp dir is no
        # longer needed once the ZIP/PDF packages the translated images.
        batch_processor.cleanup_temp_files(batch_result)

        return jsonify(
            {
                "success": True,
                "batch_id": batch_result["batch_id"],
                "processed": batch_result["processed"],
                "failed": batch_result["failed"],
                "total": batch_result["total_pages"],
                "outputs": output_files,
                "errors": batch_result["errors"],
                "chunk_size": chunk_size,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/batch/retranslate", methods=["POST"])
def batch_retranslate():
    """
    Re-translate a failed batch using saved OCR text from batch_info.json.
    Skips YOLO detection and OCR — only redoes translation + inpainting + render.

    Body: { "batch_id": "20260402_091923", "translator": "gemma3",
            "story_context": "...", "target_lang": "en" }
    """
    global pipeline

    try:
        data = request.get_json() or {}
        batch_id = data.get("batch_id")
        if not batch_id:
            return jsonify({"error": "batch_id is required"}), 400

        batch_dir = Path(app.config["BATCH_FOLDER"]) / f"batch_{batch_id}"
        info_path = batch_dir / "batch_info.json"
        if not info_path.exists():
            return jsonify({"error": f"batch_info.json not found for batch {batch_id}"}), 404

        saved = json.loads(info_path.read_text(encoding="utf-8"))

        target_lang = data.get("target_lang", "en")
        translator_type = data.get("translator", "gemma3")
        story_context = data.get("story_context") or saved.get("story_context")
        output_format = data.get("output_format", "zip")
        include_originals = bool(data.get("include_originals", True))

        from PIL import Image as PILImage
        from src.models.inpainter import TextInpainter
        from src.translators.base import TranslatorFactory
        from src.utils.image import find_whitest_pixel
        from src.utils.text import fit_text_to_box, render_text_overlay
        from config.settings import USE_LAMA_FOR_REGIONS
        import numpy as np
        from PIL import ImageDraw, ImageColor

        translator = TranslatorFactory.create(translator_type, "ja", target_lang)
        inpainter = TextInpainter()
        text_rgb = ImageColor.getrgb("#000000")

        new_batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_batch_dir = Path(app.config["BATCH_FOLDER"]) / f"batch_{new_batch_id}"
        new_batch_dir.mkdir(parents=True, exist_ok=True)

        new_pages = []
        errors = []

        for page in saved.get("pages", []):
            input_path = page.get("input")
            filename = page.get("filename", "")
            translations_saved = page.get("stats", {}).get("translations", [])

            if not input_path or not Path(input_path).exists():
                errors.append({"filename": filename, "error": "Input image not found"})
                continue

            try:
                # Collect OCR texts from saved data
                ocr_items = []  # (bbox_or_None, class_id, original_text)
                for t in translations_saved:
                    orig = t.get("original", "")
                    if not orig:
                        continue
                    raw_bbox = t.get("bbox")
                    bbox = tuple(raw_bbox) if raw_bbox else None
                    class_id = t.get("class_id", 0)
                    ocr_items.append((bbox, class_id, orig))

                if not ocr_items:
                    errors.append({"filename": filename, "error": "No saved OCR data"})
                    continue

                has_bboxes = any(item[0] is not None for item in ocr_items)

                # Batch translate all texts for this page
                texts = [item[2] for item in ocr_items]
                context_parts = []
                if story_context:
                    ctx = story_context[:1200] + ("…" if len(story_context) > 1200 else "")
                    context_parts.append(f"[Story Context]\n{ctx}")
                context_prompt = "\n".join(context_parts)

                try:
                    translated_texts = translator.translate_batch(
                        texts, context_prompt=context_prompt, story_context=story_context
                    )
                except Exception as te:
                    errors.append({"filename": filename, "error": f"Translation failed: {te}"})
                    continue

                # Unload translator to free VRAM for inpainting
                translator.unload()

                output_image = PILImage.open(input_path).convert("RGB")

                if has_bboxes:
                    if not inpainter.available:
                        inpainter.try_reconnect()

                    # Pass 2: inpainting
                    for (bbox, class_id, _orig), translated in zip(ocr_items, translated_texts):
                        if not translated or bbox is None:
                            continue
                        x1, y1, x2, y2 = bbox
                        region_pixels = np.array(output_image)[y1:y2, x1:x2]
                        mean_brightness = region_pixels.mean()
                        if mean_brightness >= 240:
                            ImageDraw.Draw(output_image).rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
                        elif class_id in USE_LAMA_FOR_REGIONS and inpainter.available:
                            output_image = inpainter.inpaint_region(output_image, (x1, y1, x2, y2))
                        else:
                            bg = find_whitest_pixel(region_pixels)
                            ImageDraw.Draw(output_image).rectangle([x1, y1, x2, y2], fill=bg)

                    # Pass 3: render text
                    draw = ImageDraw.Draw(output_image)
                    boxes, texts_out, sizes, colors = [], [], [], []
                    for (bbox, _class_id, _orig), translated in zip(ocr_items, translated_texts):
                        if not translated or bbox is None:
                            continue
                        x1, y1, x2, y2 = bbox
                        wrapped, font = fit_text_to_box(draw, translated, (x1, y1, x2, y2))
                        bg = np.array(output_image)[y1:y2, x1:x2].mean()
                        color = (255, 255, 255, 255) if bg < 160 else (*text_rgb, 255)
                        boxes.append((x1, y1, x2, y2))
                        texts_out.append(wrapped)
                        sizes.append(font.size)
                        colors.append(color)

                    if boxes:
                        output_image = render_text_overlay(output_image, boxes, texts_out, sizes, colors)

                output_path = new_batch_dir / f"translated_{filename}"
                output_image.save(str(output_path))

                new_pages.append({
                    "index": page.get("index"),
                    "input": input_path,
                    "output": str(output_path),
                    "filename": filename,
                    "stats": {
                        "bubbles_detected": len(ocr_items),
                        "regions_detected": len(ocr_items),
                        "translations": [
                            {"original": orig, "translated": tgt}
                            for orig, tgt in zip(texts, translated_texts)
                        ],
                    },
                })

            except Exception as page_err:
                errors.append({"filename": filename, "error": str(page_err)})

        if not new_pages:
            return jsonify({"error": "All pages failed to retranslate", "errors": errors}), 422

        batch_result = {
            "batch_id": new_batch_id,
            "temp_dir": str(new_batch_dir),
            "pages": new_pages,
            "processed": len(new_pages),
            "failed": len(errors),
            "total_pages": len(saved.get("pages", [])),
            "errors": errors,
        }

        # Save updated batch_info.json
        (new_batch_dir / "batch_info.json").write_text(
            json.dumps(batch_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        output_files = {}
        if output_format in {"zip", "both"}:
            try:
                zip_path = batch_processor.create_zip(batch_result, include_originals=include_originals)
                output_files["zip"] = {
                    "path": zip_path,
                    "url": url_for("get_batch_output", filename=Path(zip_path).name),
                    "filename": Path(zip_path).name,
                }
            except Exception as ze:
                output_files["zip"] = {"error": str(ze)}

        batch_processor.cleanup_temp_files(batch_result)

        return jsonify({
            "success": True,
            "batch_id": new_batch_id,
            "processed": len(new_pages),
            "failed": len(errors),
            "outputs": output_files,
            "errors": errors,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/batch_outputs/<filename>")
def get_batch_output(filename: str):
    return send_from_directory(app.config["BATCH_FOLDER"], filename)


@app.route("/outputs/<filename>")
def get_output(filename: str):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


@app.route("/uploads/<filename>")
def get_upload(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


def _ensure_manga_font():
    """Download Bangers (OFL manga-style font) on first run. Falls back to DejaVu."""
    font_path = PROJECT_ROOT / "assets" / "fonts" / "Bangers-Regular.ttf"
    if not font_path.exists():
        font_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            # Try multiple CDN URLs in case one is stale
            urls = [
                "https://fonts.gstatic.com/s/bangers/v25/FeVQS0BTqb0h60ACL5la2bxii28wYQ.ttf",
                "https://fonts.gstatic.com/s/bangers/v24/FeVQS0BTqb0h60ACL5la2bxii28wYQ.ttf",
                "https://github.com/google/fonts/raw/main/ofl/bangers/Bangers-Regular.ttf",
            ]
            downloaded = False
            for url in urls:
                try:
                    urllib.request.urlretrieve(url, font_path)
                    downloaded = True
                    break
                except Exception:
                    continue
            if not downloaded:
                raise Exception("All font URLs failed")
            print(f"✅ Downloaded Bangers font → {font_path}")
        except Exception as e:
            print(f"⚠️  Font download failed ({e}), using system font")


if __name__ == "__main__":
    _ensure_manga_font()
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
