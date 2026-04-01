"""Lazy colorization service with setup checks for the public UI."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.settings import DEVICE, PROJECT_ROOT


COLORIZATION_DIR = Path(PROJECT_ROOT) / "colorization"
DEFAULT_GENERATOR_PATH = Path(os.getenv("COLORIZATION_GENERATOR_PATH", COLORIZATION_DIR / "networks" / "generator.zip"))
DEFAULT_EXTRACTOR_PATH = Path(os.getenv("COLORIZATION_EXTRACTOR_PATH", COLORIZATION_DIR / "networks" / "extractor.pth"))


def get_colorization_status() -> dict[str, Any]:
    """Return whether the bundled colorization pipeline is ready to run."""
    missing: list[str] = []
    if not COLORIZATION_DIR.exists():
        missing.append(f"Missing colorization directory: {COLORIZATION_DIR}")
    if not DEFAULT_GENERATOR_PATH.exists():
        missing.append(f"Missing generator weights: {DEFAULT_GENERATOR_PATH}")
    if not DEFAULT_EXTRACTOR_PATH.exists():
        missing.append(f"Missing extractor weights: {DEFAULT_EXTRACTOR_PATH}")

    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as exc:
        missing.append(f"Missing torch/torchvision runtime: {exc}")

    return {
        "available": not missing,
        "device": DEVICE,
        "generator_path": str(DEFAULT_GENERATOR_PATH),
        "extractor_path": str(DEFAULT_EXTRACTOR_PATH),
        "missing_requirements": missing,
        "enable_instructions": (
            "Add the expected generator and extractor weights under colorization/networks "
            "or set COLORIZATION_GENERATOR_PATH and COLORIZATION_EXTRACTOR_PATH."
        ),
    }


@lru_cache(maxsize=1)
def _load_colorizator():
    status = get_colorization_status()
    if not status["available"]:
        raise RuntimeError("; ".join(status["missing_requirements"]))

    if str(COLORIZATION_DIR) not in sys.path:
        sys.path.insert(0, str(COLORIZATION_DIR))

    from colorizator import MangaColorizator

    return MangaColorizator(DEVICE, str(DEFAULT_GENERATOR_PATH), str(DEFAULT_EXTRACTOR_PATH))


def colorize_image(
    input_path: str | Path,
    output_path: str | Path,
    *,
    size: int = 576,
    denoiser: bool = True,
    denoiser_sigma: int = 25,
) -> str:
    """Colorize one manga page and save it to disk."""
    status = get_colorization_status()
    if not status["available"]:
        raise RuntimeError("; ".join(status["missing_requirements"]))

    import matplotlib.pyplot as plt

    colorizator = _load_colorizator()
    image = plt.imread(str(input_path))
    colorizator.set_image(image, size, denoiser, denoiser_sigma)
    colorized = colorizator.colorize()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, colorized)
    return str(output_path)
