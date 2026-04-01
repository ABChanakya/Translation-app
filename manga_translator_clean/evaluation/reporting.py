"""Evaluation and training report generation utilities."""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any


def _lazy_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _safe_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name).strip("_") or "report"


def load_training_history(training_run_dir: str | Path | None) -> list[dict[str, float | int]] | None:
    """Load Ultralytics ``results.csv`` rows if they exist."""
    if not training_run_dir:
        return None

    csv_path = Path(training_run_dir) / "results.csv"
    if not csv_path.exists():
        return None

    history: list[dict[str, float | int]] = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized: dict[str, float | int] = {}
            for key, value in row.items():
                clean_key = (key or "").strip()
                if not clean_key:
                    continue
                try:
                    number = float(value)
                    normalized[clean_key] = int(number) if clean_key == "epoch" else number
                except (TypeError, ValueError):
                    continue
            history.append(normalized)
    return history or None


def _save_bar_chart(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
) -> bool:
    if not labels:
        return False

    plt = _lazy_matplotlib()
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.4), 4.8))
    axis.bar(labels, values, color=color)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, max(max(values) * 1.2, 1.0))
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def _save_grouped_metric_chart(per_class_rows: list[dict[str, Any]], output_path: Path) -> bool:
    if not per_class_rows:
        return False

    plt = _lazy_matplotlib()
    labels = [row["class_name"] for row in per_class_rows]
    precision = [row["precision"] for row in per_class_rows]
    recall = [row["recall"] for row in per_class_rows]
    f1 = [row["f1"] for row in per_class_rows]

    figure, axis = plt.subplots(figsize=(max(9, len(labels) * 1.5), 5.4))
    indices = range(len(labels))
    width = 0.24
    axis.bar([index - width for index in indices], precision, width=width, label="Precision", color="#1f77b4")
    axis.bar(indices, recall, width=width, label="Recall", color="#2ca02c")
    axis.bar([index + width for index in indices], f1, width=width, label="F1", color="#ff7f0e")
    axis.set_xticks(list(indices))
    axis.set_xticklabels(labels, rotation=25)
    axis.set_ylim(0, 1.0)
    axis.set_title("Per-Class Precision / Recall / F1")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def _save_training_curves(history: list[dict[str, float | int]], output_path: Path) -> bool:
    if not history:
        return False

    plt = _lazy_matplotlib()
    epochs = [row.get("epoch", index + 1) for index, row in enumerate(history)]
    metric_columns = [
        ("train/box_loss", "Train Box Loss"),
        ("train/cls_loss", "Train Cls Loss"),
        ("train/dfl_loss", "Train DFL Loss"),
        ("val/box_loss", "Val Box Loss"),
        ("val/cls_loss", "Val Cls Loss"),
        ("val/dfl_loss", "Val DFL Loss"),
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
    ]

    available_columns = [item for item in metric_columns if any(item[0] in row for row in history)]
    if not available_columns:
        return False

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for column, label in available_columns:
        values = [row.get(column) for row in history]
        if any(value is not None for value in values):
            target_axis = axes[0] if "loss" in column else axes[1]
            target_axis.plot(epochs, values, marker="o", linewidth=1.8, label=label)

    axes[0].set_title("Training Loss Curves")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation / Detection Metrics")
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.0)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def _summary_cards_html(summary: dict[str, Any]) -> str:
    metrics = summary["aggregate_metrics"]
    cards = [
        ("mAP50", metrics["mAP50"]),
        ("mAP50-95", metrics["mAP50_95"]),
        ("Macro Recall", metrics["macro_recall"]),
        ("Macro F1", metrics["macro_f1"]),
        ("Macro IoU", metrics["macro_iou"]),
    ]
    return "".join(
        f"<div class='metric-card'><span>{escape(label)}</span><strong>{value:.4f}</strong></div>"
        for label, value in cards
    )


def _per_class_rows_html(summary: dict[str, Any]) -> str:
    rows_html: list[str] = []
    for row in summary["per_class"]:
        rows_html.append(
            "<tr>"
            f"<td>{escape(row['class_name'])}</td>"
            f"<td>{row['support']}</td>"
            f"<td>{row['prediction_count']}</td>"
            f"<td>{row['precision']:.4f}</td>"
            f"<td>{row['recall']:.4f}</td>"
            f"<td>{row['f1']:.4f}</td>"
            f"<td>{row['ap50']:.4f}</td>"
            f"<td>{row['ap50_95']:.4f}</td>"
            f"<td>{row['matched_iou']:.4f}</td>"
            "</tr>"
        )
    return "\n".join(rows_html)


def _class_list_html(title: str, rows: list[dict[str, Any]]) -> str:
    items = "".join(
        "<li>"
        f"<strong>{escape(row['class_name'])}</strong> "
        f"(support={row['support']}, recall={row['recall']:.3f}, F1={row['f1']:.3f}, IoU={row['matched_iou']:.3f})"
        "</li>"
        for row in rows
    )
    return f"<section><h3>{escape(title)}</h3><ul>{items or '<li>No classes available.</li>'}</ul></section>"


def _write_report_html(summary: dict[str, Any], output_path: Path, chart_files: dict[str, str]) -> None:
    metrics = summary["aggregate_metrics"]
    chart_blocks = []
    chart_labels = {
        "training_curves": "Training Curves",
        "class_support": "Class Support",
        "per_class_prf": "Per-Class Precision / Recall / F1",
        "per_class_ap": "Per-Class AP",
        "per_class_iou": "Per-Class Box IoU",
    }
    for key, filename in chart_files.items():
        chart_blocks.append(
            "<section>"
            f"<h3>{escape(chart_labels.get(key, key.replace('_', ' ').title()))}</h3>"
            f"<img src='{escape(filename)}' alt='{escape(chart_labels.get(key, key))}' class='report-image'>"
            "</section>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evaluation Report</title>
  <style>
    body {{
      font-family: "Segoe UI", Tahoma, sans-serif;
      margin: 0;
      background: #f6f8fb;
      color: #1f2937;
    }}
    .shell {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    p {{ margin: 0 0 12px; }}
    .hero {{
      background: linear-gradient(135deg, #143a52, #0f766e);
      color: white;
      border-radius: 18px;
      padding: 28px;
      margin-bottom: 24px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric-card {{
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 14px;
      padding: 14px 16px;
    }}
    .metric-card span {{
      display: block;
      font-size: 0.85rem;
      opacity: 0.9;
    }}
    .metric-card strong {{
      display: block;
      font-size: 1.4rem;
      margin-top: 4px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .panel, section {{
      background: white;
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
      margin-bottom: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
    }}
    th, td {{
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      padding: 10px 8px;
    }}
    th {{
      background: #f8fafc;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
    }}
    .report-image {{
      max-width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid #e5e7eb;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 18px;
      font-size: 0.95rem;
    }}
    .meta div {{
      background: #f8fafc;
      border-radius: 12px;
      padding: 12px 14px;
    }}
    @media (max-width: 860px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <h1>Auto Manga Translation Evaluation Report</h1>
      <p>Fairness-first detection summary with macro and per-class metrics.</p>
      <div class="meta">
        <div><strong>Model</strong><br>{escape(summary['model'])}</div>
        <div><strong>Dataset</strong><br>{escape(summary['data'])} [{escape(summary['split'])}]</div>
        <div><strong>NMS IoU</strong><br>{summary['nms_iou']:.2f}</div>
        <div><strong>Matched Box IoU Threshold</strong><br>{summary['match_iou_threshold']:.2f}</div>
      </div>
      <div class="cards">{_summary_cards_html(summary)}</div>
    </div>

    <div class="grid">
      <div class="panel">
        <h2>Aggregate Metrics</h2>
        <table>
          <tbody>
            <tr><th>Precision</th><td>{metrics['precision']:.4f}</td></tr>
            <tr><th>Recall</th><td>{metrics['recall']:.4f}</td></tr>
            <tr><th>Overall F1</th><td>{metrics['f1']:.4f}</td></tr>
            <tr><th>Macro Precision</th><td>{metrics['macro_precision']:.4f}</td></tr>
            <tr><th>Macro Recall</th><td>{metrics['macro_recall']:.4f}</td></tr>
            <tr><th>Macro F1</th><td>{metrics['macro_f1']:.4f}</td></tr>
            <tr><th>Weighted Precision</th><td>{metrics['weighted_precision']:.4f}</td></tr>
            <tr><th>Weighted Recall</th><td>{metrics['weighted_recall']:.4f}</td></tr>
            <tr><th>Weighted F1</th><td>{metrics['weighted_f1']:.4f}</td></tr>
            <tr><th>Box-Level Macro IoU</th><td>{metrics['macro_iou']:.4f}</td></tr>
            <tr><th>Matched IoU</th><td>{metrics['matched_iou']:.4f}</td></tr>
            <tr><th>Support Total</th><td>{metrics['support_total']}</td></tr>
            <tr><th>Prediction Total</th><td>{metrics['prediction_total']}</td></tr>
          </tbody>
        </table>
      </div>
      <div>
        {_class_list_html("Weakest Classes", summary["weakest_classes"])}
        {_class_list_html("Strongest Classes", summary["strongest_classes"])}
      </div>
    </div>

    <section>
      <h2>Per-Class Performance</h2>
      <p>Support counts are shown next to each class so underrepresented classes stay visible in the summary.</p>
      <table>
        <thead>
          <tr>
            <th>Class</th>
            <th>Support</th>
            <th>Predictions</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>AP50</th>
            <th>AP50-95</th>
            <th>Box IoU</th>
          </tr>
        </thead>
        <tbody>
          {_per_class_rows_html(summary)}
        </tbody>
      </table>
    </section>

    {''.join(chart_blocks)}
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def write_report_artifacts(
    summary: dict[str, Any],
    *,
    save_root: str | Path,
    run_name: str,
    training_run_dir: str | Path | None = None,
) -> dict[str, str]:
    """Persist JSON summaries, charts, and ``report.html`` for one evaluation."""
    report_dir = Path(save_root) / _safe_filename(run_name)
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = report_dir / "metrics.json"
    fairness_path = report_dir / "fairness.json"
    summary_path = report_dir / "summary.json"

    metrics_payload = summary["aggregate_metrics"]
    fairness_payload = {
        "per_class": summary["per_class"],
        "weakest_classes": summary["weakest_classes"],
        "strongest_classes": summary["strongest_classes"],
        "box_iou_label": "Box-level matched IoU for detection analysis",
    }

    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fairness_path.write_text(json.dumps(fairness_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    chart_files: dict[str, str] = {}
    per_class_rows = summary["per_class"]

    if _save_bar_chart(
        [row["class_name"] for row in per_class_rows],
        [row["support"] for row in per_class_rows],
        title="Class Support",
        ylabel="Ground-Truth Instances",
        output_path=report_dir / "class_support.png",
        color="#0f766e",
    ):
        chart_files["class_support"] = "class_support.png"

    if _save_grouped_metric_chart(per_class_rows, report_dir / "per_class_prf.png"):
        chart_files["per_class_prf"] = "per_class_prf.png"

    if _save_bar_chart(
        [row["class_name"] for row in per_class_rows],
        [row["ap50_95"] for row in per_class_rows],
        title="Per-Class AP50-95",
        ylabel="AP50-95",
        output_path=report_dir / "per_class_ap.png",
        color="#2563eb",
    ):
        chart_files["per_class_ap"] = "per_class_ap.png"

    if _save_bar_chart(
        [row["class_name"] for row in per_class_rows],
        [row["matched_iou"] for row in per_class_rows],
        title="Per-Class Box IoU",
        ylabel="Matched IoU",
        output_path=report_dir / "per_class_iou.png",
        color="#ea580c",
    ):
        chart_files["per_class_iou"] = "per_class_iou.png"

    training_history = load_training_history(training_run_dir)
    if _save_training_curves(training_history or [], report_dir / "training_curves.png"):
        chart_files["training_curves"] = "training_curves.png"

    report_html_path = report_dir / "report.html"
    _write_report_html(summary, report_html_path, chart_files)

    return {
        "report_dir": str(report_dir),
        "report_html": str(report_html_path),
        "metrics_json": str(metrics_path),
        "fairness_json": str(fairness_path),
        "summary_json": str(summary_path),
        "charts": {name: str(report_dir / filename) for name, filename in chart_files.items()},
    }
