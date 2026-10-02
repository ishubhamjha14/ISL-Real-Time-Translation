"""
export_tableau_data.py
Phase 7 & 8: Structured Data Exporter for Tableau Desktop and Tableau Public.

Exports:
1. reports/model_evaluation_metrics.csv:
   - Columns: gloss_name, precision, recall, f1_score, support_count, top1_accuracy
   - Per-gloss test classification metrics across all 37 Indian Sign Language classes.
2. reports/confusion_pairs.csv:
   - Columns: actual_gloss, predicted_gloss, count, normalized_percentage
   - Flattened pairwise records of the full confusion matrix for heatmap visualization.
"""

import os
import sys
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Auto-fallback to isl_py310 conda environment
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    conda_py = r"C:\Users\abhis\anaconda3\anaconda\envs\isl_py310\python.exe"
    if os.path.exists(conda_py) and sys.executable.lower() != conda_py.lower():
        import subprocess
        result = subprocess.run([conda_py] + sys.argv, cwd=str(project_root))
        sys.exit(result.returncode)
    raise

from src.dataset import create_dataloaders
from src.model import ISLTransformerClassifier


def load_model(checkpoint_path: Path, device: torch.device):
    """Loads checkpoint weights and instantiates the Transformer Classifier."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    num_classes = checkpoint.get("num_classes", 37)
    feature_dim = checkpoint.get("feature_dim", 226)

    hp = checkpoint.get("hyperparameters", {})
    d_model = hp.get("d_model", 128)
    nhead = hp.get("nhead", 4)
    dim_feedforward = hp.get("dim_feedforward", 256)
    num_layers = hp.get("num_layers", 2)
    dropout = hp.get("dropout", 0.35)

    model = ISLTransformerClassifier(
        input_dim=feature_dim,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        dropout=dropout,
        pooling="mean",
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def export_tableau_csvs(
    checkpoint_path: Path = project_root / "models" / "best_transformer_gloss.pth",
    label_map_path: Path = project_root / "data" / "label_map.json",
    processed_dir: Path = project_root / "data" / "processed_data",
    reports_dir: Path = project_root / "reports",
) -> Tuple[Path, Path]:
    """
    Evaluates test split and writes:
      1. reports/model_evaluation_metrics.csv
      2. reports/confusion_pairs.csv
    Returns:
      (metrics_csv_path, confusion_csv_path)
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Label Mapping
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    class_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    # Load Model
    model, _ = load_model(checkpoint_path, device)

    # Load Test Data
    _, _, test_loader, _ = create_dataloaders(
        processed_dir=processed_dir,
        label_map_path=label_map_path,
        batch_size=16,
    )

    all_preds = []
    all_targets = []

    print("[Tableau Export] Evaluating model on held-out test split...")
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    # 1. Compute Classification Report
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # 2. Export reports/model_evaluation_metrics.csv
    metrics_csv_path = reports_dir / "model_evaluation_metrics.csv"
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gloss_name",
            "precision",
            "recall",
            "f1_score",
            "support_count",
            "top1_accuracy",
        ])

        for cname in class_names:
            c_metrics = report.get(cname, {})
            prec = float(c_metrics.get("precision", 0.0))
            rec = float(c_metrics.get("recall", 0.0))
            f1 = float(c_metrics.get("f1-score", 0.0))
            supp = int(c_metrics.get("support", 0))
            # In multi-class classification, class top-1 accuracy = class recall
            top1_acc = round(rec * 100.0, 2)

            writer.writerow([
                cname,
                f"{prec:.4f}",
                f"{rec:.4f}",
                f"{f1:.4f}",
                supp,
                f"{top1_acc:.2f}",
            ])

    print(f"[Tableau Export] Saved per-gloss metrics -> {metrics_csv_path.resolve()} ({num_classes} classes)")

    # 3. Compute Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))

    # 4. Export reports/confusion_pairs.csv
    confusion_csv_path = reports_dir / "confusion_pairs.csv"
    with open(confusion_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "actual_gloss",
            "predicted_gloss",
            "count",
            "normalized_percentage",
        ])

        pair_count = 0
        for i, actual_class in enumerate(class_names):
            row_sum = int(cm[i].sum())
            for j, pred_class in enumerate(class_names):
                count = int(cm[i, j])
                norm_pct = round((count / row_sum) * 100.0, 2) if row_sum > 0 else 0.0
                writer.writerow([
                    actual_class,
                    pred_class,
                    count,
                    f"{norm_pct:.2f}",
                ])
                pair_count += 1

    print(f"[Tableau Export] Saved confusion matrix pairs -> {confusion_csv_path.resolve()} ({pair_count} pairs)")

    return metrics_csv_path, confusion_csv_path


if __name__ == "__main__":
    print("=== Exporting Tableau Analytics CSV Files ===")
    m_path, c_path = export_tableau_csvs()
    print("\nTableau Files Ready for Import:")
    print(f" 1. Metrics:   {m_path.resolve()}")
    print(f" 2. Confusion: {c_path.resolve()}")
    pilot_csv = project_root / "reports" / "realtime_pilot_logs.csv"
    if pilot_csv.exists():
        print(f" 3. Live Logs: {pilot_csv.resolve()}")
    print("\n=== All Tableau CSVs Exported Successfully! ===")
