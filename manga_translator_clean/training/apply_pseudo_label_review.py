#!/usr/bin/env python3
"""Apply approve/reject decisions from a pseudo-label review manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply pseudo-label review decisions")
    parser.add_argument("--manifest", required=True, help="Path to review_manifest.json")
    parser.add_argument(
        "--reject-action",
        choices=("remove", "move"),
        default="remove",
        help="How to handle rejected files",
    )
    parser.add_argument(
        "--rejected-dir",
        default="training/rejected_pseudo_labels",
        help="Target directory when using --reject-action move",
    )
    return parser.parse_args()


def _safe_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _move_file(source: Path, destination_root: Path) -> None:
    relative_name = source.name
    destination = destination_root / relative_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rejected_dir = Path(args.rejected_dir).resolve()
    processed = {"approved": 0, "rejected": 0, "uncertain": 0}

    for item in manifest.get("items", []):
        status = item.get("review_status", "uncertain")
        processed[status] = processed.get(status, 0) + 1
        if status != "reject":
            continue

        candidate_paths = [
            _safe_path(item.get("source_image")),
            _safe_path(item.get("label_path")),
            _safe_path(item.get("overlay_path")),
        ]

        for path in [candidate for candidate in candidate_paths if candidate]:
            if args.reject_action == "move":
                _move_file(path, rejected_dir)
            else:
                path.unlink(missing_ok=True)

    manifest["applied_at"] = __import__("datetime").datetime.now().isoformat()
    manifest["reject_action"] = args.reject_action
    manifest["review_counts"] = processed
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Applied pseudo-label review decisions.")
    print(json.dumps(processed, indent=2))


if __name__ == "__main__":
    main()
