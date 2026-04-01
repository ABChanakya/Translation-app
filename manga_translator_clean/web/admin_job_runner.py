"""Background runner that updates admin job manifests."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit("Usage: admin_job_runner.py <manifest.json> -- <command...>")

    manifest_path = Path(sys.argv[1]).resolve()
    command = sys.argv[3:]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["status"] = "running"
    payload["started_at"] = datetime.now().isoformat()
    _write_manifest(manifest_path, payload)

    log_path = Path(payload["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"[{datetime.now().isoformat()}] Launching: {' '.join(command)}\n")
        log_handle.flush()
        result = subprocess.run(
            command,
            cwd=payload["cwd"],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    payload["finished_at"] = datetime.now().isoformat()
    payload["return_code"] = result.returncode
    payload["status"] = "completed" if result.returncode == 0 else "failed"
    _write_manifest(manifest_path, payload)


if __name__ == "__main__":
    main()
