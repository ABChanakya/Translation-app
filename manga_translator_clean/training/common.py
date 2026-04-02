from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
DATASET_DIR = TRAINING_DIR / "datasets"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
YOLO_RUNS_DIR = PROJECT_ROOT / "yolo_train_run"
DEFAULT_RESULTS_DIR = EVALUATION_DIR / "results"
DEFAULT_DATASET_YAML = DATASET_DIR / "custom_manga.yaml"
DEFAULT_BASE_MODEL = MODELS_DIR / "yolo11n.pt"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class TrainingPhase:
    """One training phase inside a multi-stage training plan."""

    name: str
    epochs: int
    lr0: float
    lrf: float
    patience: int
    freeze: int | None = 0
    overrides: Dict[str, Any] = field(default_factory=dict)


def resolve_project_path(value: str | Path, base: Path | None = None) -> Path:
    """Resolve a path relative to the project root unless it is already absolute."""
    path = Path(value)
    if path.is_absolute():
        return path
    return ((base or PROJECT_ROOT) / path).resolve()


def resolve_artifact_reference(value: str | Path, base: Path | None = None) -> str:
    """
    Resolve local checkpoints while still allowing remote Ultralytics model names.

    If the path does not exist locally we return the original string so users can still
    pass identifiers such as ``yolo11n.pt`` or Hugging Face model handles explicitly.
    """

    candidate = resolve_project_path(value, base=base)
    return str(candidate) if candidate.exists() else str(value)


def load_yaml(path: str | Path) -> Mapping[str, Any]:
    yaml_path = resolve_project_path(path)
    with open(yaml_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_class_names(names: Any) -> Dict[int, str]:
    """Convert YOLO class name formats into a predictable ``{id: name}`` mapping."""
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def resolve_dataset_root(data_yaml: str | Path, data_config: Mapping[str, Any]) -> Path:
    data_yaml_path = resolve_project_path(data_yaml)
    root_value = data_config.get("path")
    if not root_value:
        return data_yaml_path.parent
    return resolve_project_path(root_value, base=data_yaml_path.parent)


def resolve_split_image_dir(data_yaml: str | Path, split: str) -> Path:
    data_yaml_path = resolve_project_path(data_yaml)
    data_config = load_yaml(data_yaml_path)
    if split not in data_config:
        raise KeyError(f"Split '{split}' is not defined in {data_yaml_path}")
    dataset_root = resolve_dataset_root(data_yaml_path, data_config)
    return resolve_project_path(data_config[split], base=dataset_root)


def prepare_ultralytics_dataset_yaml(data_yaml: str | Path) -> Path:
    """
    Materialize a dataset YAML with absolute paths for Ultralytics.

    Ultralytics may interpret relative ``path`` values relative to the process
    working directory instead of the dataset YAML location in some launch paths.
    To keep training and evaluation stable from the admin UI and CLI, we
    generate a resolved YAML under ``training/datasets/_resolved``.
    """

    data_yaml_path = resolve_project_path(data_yaml)
    data_config = dict(load_yaml(data_yaml_path))
    dataset_root = resolve_dataset_root(data_yaml_path, data_config)

    resolved_config: Dict[str, Any] = dict(data_config)
    resolved_config["path"] = str(dataset_root)

    for split in ("train", "val", "test"):
        if split in data_config:
            resolved_config[split] = str(resolve_project_path(data_config[split], base=dataset_root))

    output_dir = DATASET_DIR / "_resolved"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / data_yaml_path.name
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False, allow_unicode=True)
    return output_path


def resolve_split_label_dir(image_dir: str | Path) -> Path:
    images_path = resolve_project_path(image_dir)
    return images_path.parent.parent / "labels" / images_path.parent.name


def iter_image_files(image_dir: str | Path) -> list[Path]:
    path = resolve_project_path(image_dir)
    files: list[Path] = []
    for suffix in IMAGE_SUFFIXES:
        files.extend(path.glob(f"*{suffix}"))
    return sorted(files)


def run_training_phase(
    model: YOLO,
    *,
    data_yaml: str | Path,
    phase: TrainingPhase,
    device: str,
    batch_size: int,
    imgsz: int,
    project_dir: str | Path = YOLO_RUNS_DIR,
    common_overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Execute one Ultralytics training phase."""
    prepared_data_yaml = prepare_ultralytics_dataset_yaml(data_yaml)
    params: Dict[str, Any] = {
        "data": str(prepared_data_yaml),
        "epochs": phase.epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "lr0": phase.lr0,
        "lrf": phase.lrf,
        "device": device,
        "patience": phase.patience,
        "project": str(resolve_project_path(project_dir)),
        "name": phase.name,
        "exist_ok": True,
        "save": True,
        "val": True,
        "plots": True,
        "verbose": True,
    }
    if phase.freeze is not None:
        params["freeze"] = phase.freeze
    if common_overrides:
        params.update(common_overrides)
    params.update(phase.overrides)

    # Ultralytics 8.4.x can drop ``self.overrides['model']`` after a completed
    # train() call, which breaks subsequent train() calls on the same instance
    # with ``KeyError: 'model'``. Guard it explicitly.
    if not isinstance(getattr(model, "overrides", None), dict):
        model.overrides = {}
    if "model" not in model.overrides:
        model_ref = getattr(model, "ckpt_path", None)
        if not model_ref:
            model_ref = getattr(model, "pt_path", None)
        if not model_ref:
            model_ref = resolve_artifact_reference(DEFAULT_BASE_MODEL)
        model.overrides["model"] = str(model_ref)

    return model.train(**params)


def run_training_plan(
    *,
    model_name: str | Path,
    data_yaml: str | Path,
    phases: Sequence[TrainingPhase],
    device: str,
    batch_size: int,
    imgsz: int,
    project_dir: str | Path = YOLO_RUNS_DIR,
    common_overrides: Mapping[str, Any] | None = None,
) -> tuple[YOLO, list[dict[str, str]]]:
    """Run a list of training phases and return the model plus generated weight paths."""
    current_model_ref = resolve_artifact_reference(model_name)
    model = YOLO(current_model_ref)
    resolved_project_dir = resolve_project_path(project_dir)
    outputs: list[dict[str, str]] = []

    for phase in phases:
        # Re-create the YOLO wrapper per phase to avoid stale internal state
        # across successive train() calls in newer Ultralytics releases.
        model = YOLO(current_model_ref)
        run_training_phase(
            model,
            data_yaml=data_yaml,
            phase=phase,
            device=device,
            batch_size=batch_size,
            imgsz=imgsz,
            project_dir=resolved_project_dir,
            common_overrides=common_overrides,
        )
        outputs.append(
            {
                "phase": phase.name,
                "weights": str(resolved_project_dir / phase.name / "weights" / "best.pt"),
            }
        )
        current_model_ref = outputs[-1]["weights"]

    if outputs:
        model = YOLO(outputs[-1]["weights"])

    return model, outputs


def ensure_directory(path: str | Path) -> Path:
    directory = resolve_project_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"
