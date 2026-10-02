"""
train.py
Phase 4 & 7: Training & Comprehensive Evaluation Pipeline for Indian Sign Language (ISL) Transformer Classifier.

Features:
1. Pipeline Integration:
   - Dynamically loads data via create_dataloaders from src.dataset.
   - Instantiates ISLTransformerClassifier from src.model.
   - Automatic GPU / CPU hardware acceleration.
2. Optimization & Regularization:
   - AdamW optimizer (lr=1e-3, weight_decay=1e-4)
   - CrossEntropyLoss with label smoothing (0.1)
   - CosineAnnealingLR scheduler (T_max=epochs, eta_min=1e-5)
   - Early stopping on validation accuracy (patience=12)
3. Checkpointing:
   - Saves best performing checkpoint to models/best_transformer_gloss.pth
4. Comprehensive Held-Out Test Evaluation:
   - Evaluates best checkpoint on held-out test_loader.
   - Computes Top-1 and Top-5 accuracy.
   - Generates full per-gloss classification report (precision, recall, f1-score).
   - Generates and saves high-resolution normalized confusion matrix to reports/confusion_matrix.png.
   - Generates training loss & accuracy progression curves to reports/training_curves.png.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Auto-fallback to isl_py310 conda environment if torch is not in system python
try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
except ImportError:
    conda_py = r"C:\Users\abhis\anaconda3\anaconda\envs\isl_py310\python.exe"
    if os.path.exists(conda_py) and sys.executable.lower() != conda_py.lower():
        import subprocess
        result = subprocess.run([conda_py] + sys.argv, cwd=str(project_root))
        sys.exit(result.returncode)
    raise

import json
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.dataset import create_dataloaders
from src.model import ISLTransformerClassifier


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Runs a single training epoch and returns (average_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()

        # Gradient clipping for stable transformer training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluates the model on validation data and returns (average_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = criterion(logits, targets)

            running_loss += loss.item() * targets.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    eval_loss = running_loss / total
    eval_acc = correct / total
    return eval_loss, eval_acc


def evaluate_test_set(
    model: nn.Module,
    loader,
    class_names: list[str],
    device: torch.device,
    reports_dir: Path,
) -> dict:
    """
    Performs comprehensive evaluation on the held-out test set:
    - Top-1 & Top-5 accuracy
    - Per-gloss Classification Report
    - Normalized Confusion Matrix plot
    """
    model.eval()
    all_preds = []
    all_targets = []
    top1_correct = 0
    top5_correct = 0
    total = 0
    k_top = min(5, len(class_names))

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)

            # Top-1 predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            top1_correct += (preds == targets).sum().item()

            # Top-5 predictions
            _, top5_indices = torch.topk(logits, k=k_top, dim=1)
            top5_match = (top5_indices == targets.unsqueeze(1)).any(dim=1)
            top5_correct += top5_match.sum().item()

            total += targets.size(0)

    top1_acc = (top1_correct / total) * 100.0
    top5_acc = (top5_correct / total) * 100.0

    print("\n" + "=" * 65)
    print("         HELD-OUT TEST SET EVALUATION RESULTS")
    print("=" * 65)
    print(f"Total Test Samples:   {total}")
    print(f"Top-1 Test Accuracy:  {top1_acc:.2f}% ({top1_correct}/{total})")
    print(f"Top-5 Test Accuracy:  {top5_acc:.2f}% ({top5_correct}/{total})")

    # Classification Report
    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        labels=list(range(len(class_names))),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        labels=list(range(len(class_names))),
        digits=4,
        zero_division=0,
    )

    print("\nPer-Gloss Classification Report:")
    print("-" * 65)
    print(report_text)
    print("=" * 65)

    # Save Classification Report to text file
    report_txt_path = reports_dir / "classification_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Top-1 Test Accuracy: {top1_acc:.2f}%\n")
        f.write(f"Top-5 Test Accuracy: {top5_acc:.2f}%\n\n")
        f.write(report_text)
    print(f"[Info] Saved classification report to '{report_txt_path}'.")

    # Save Test Metrics Summary to JSON
    metrics_json_path = reports_dir / "test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "top1_accuracy": top1_acc,
                "top5_accuracy": top5_acc,
                "macro_avg_f1": report_dict.get("macro avg", {}).get("f1-score", 0.0),
                "weighted_avg_f1": report_dict.get("weighted avg", {}).get("f1-score", 0.0),
            },
            f,
            indent=4,
        )

    # Save per-gloss evaluation metrics to reports/model_evaluation_metrics.csv
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
            c_metrics = report_dict.get(cname, {})
            prec = float(c_metrics.get("precision", 0.0))
            rec = float(c_metrics.get("recall", 0.0))
            f1 = float(c_metrics.get("f1-score", 0.0))
            supp = int(c_metrics.get("support", 0))
            top1_acc_c = round(rec * 100.0, 2)
            writer.writerow([
                cname,
                f"{prec:.4f}",
                f"{rec:.4f}",
                f"{f1:.4f}",
                supp,
                f"{top1_acc_c:.2f}",
            ])
    print(f"[Info] Saved per-gloss evaluation metrics to '{metrics_csv_path}'.")

    # Normalized Confusion Matrix
    cm = confusion_matrix(
        all_targets,
        all_preds,
        labels=list(range(len(class_names))),
        normalize="true",
    )
    raw_cm = confusion_matrix(
        all_targets,
        all_preds,
        labels=list(range(len(class_names))),
    )

    # Save pairwise confusion records to reports/confusion_pairs.csv
    confusion_pairs_path = reports_dir / "confusion_pairs.csv"
    with open(confusion_pairs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "actual_gloss",
            "predicted_gloss",
            "count",
            "normalized_percentage",
        ])
        for i, actual in enumerate(class_names):
            for j, predicted in enumerate(class_names):
                count = int(raw_cm[i, j])
                norm_pct = float(cm[i, j] * 100.0) if not np.isnan(cm[i, j]) else 0.0
                writer.writerow([
                    actual,
                    predicted,
                    count,
                    f"{norm_pct:.2f}",
                ])
    print(f"[Info] Saved confusion pairs to '{confusion_pairs_path}'.")

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 6.5},
        ax=ax,
    )
    ax.set_title("Normalized Confusion Matrix - ISL Transformer Classifier", fontsize=15, pad=12)
    ax.set_xlabel("Predicted Gloss", fontsize=12, labelpad=10)
    ax.set_ylabel("Ground Truth Gloss", fontsize=12, labelpad=10)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    cm_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"[Info] Saved normalized confusion matrix to '{cm_path}'.")

    return {
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "classification_report": report_dict,
    }


def plot_training_curves(history: dict, reports_dir: Path):
    """Plots and saves loss and accuracy learning curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curves
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", markersize=4)
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=13)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy Curves
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Accuracy", markersize=4)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r--s", label="Validation Accuracy", markersize=4)
    ax2.set_title("Top-1 Accuracy vs. Epochs", fontsize=13)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    curves_path = reports_dir / "training_curves.png"
    plt.savefig(curves_path, dpi=300)
    plt.close(fig)
    print(f"[Info] Saved training curves to '{curves_path}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the ISL Transformer Classifier on processed landmark sequences."
    )
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs (default: 80).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and eval (default: 16).")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate for AdamW (default: 3e-4).")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW (default: 1e-4).")
    parser.add_argument("--patience", type=int, default=18, help="Early stopping patience in epochs (default: 18).")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer d_model dimension (default: 128).")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads (default: 4).")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer encoder layers (default: 2).")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="Feedforward dimension (default: 256).")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout rate (default: 0.35).")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "attention"], help="Temporal pooling.")
    parser.add_argument("--processed_dir", type=str, default="data/processed_data", help="Directory of processed .npy files.")
    parser.add_argument("--label_map", type=str, default="data/label_map.json", help="Path to label map JSON.")
    parser.add_argument("--checkpoint_dir", type=str, default="models", help="Directory to save model checkpoints.")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Directory to save evaluation reports.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Directories
    checkpoint_dir = Path(args.checkpoint_dir)
    reports_dir = Path(args.reports_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ISL Transformer Classifier Training Pipeline ===")
    print(f"Device:               {device}")
    print(f"Epochs:               {args.epochs}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Learning Rate:        {args.lr}")
    print(f"Weight Decay:         {args.weight_decay}")
    print(f"Early Stopping:       Patience = {args.patience}")
    print(f"Checkpoint Directory: {checkpoint_dir.resolve()}")
    print(f"Reports Directory:    {reports_dir.resolve()}")

    # Load Label Mapping
    label_map_file = Path(args.label_map)
    if not label_map_file.exists():
        raise FileNotFoundError(f"Label map not found at {label_map_file}")

    with open(label_map_file, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    # Invert mapping: index -> class_name sorted by index
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    num_classes = len(class_names)
    print(f"Total Sign Classes:   {num_classes}")

    # Build DataLoaders
    train_loader, val_loader, test_loader, feature_dim = create_dataloaders(
        processed_dir=args.processed_dir,
        label_map_path=args.label_map,
        batch_size=args.batch_size,
        val_split=0.15,
        test_split=0.15,
        seed=args.seed,
    )

    print(f"Feature Dimension:    {feature_dim}")
    print(f"Train Samples:        {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"Validation Samples:   {len(val_loader.dataset)} ({len(val_loader)} batches)")
    print(f"Test Samples:         {len(test_loader.dataset)} ({len(test_loader)} batches)")

    # Build Model
    model = ISLTransformerClassifier(
        input_dim=feature_dim,
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters:     {total_params:,}")

    # 1. Calculate inverse class frequency weights from training labels
    # class_weights = total_samples / (num_classes * class_sample_counts)
    train_labels_arr = np.array(train_loader.dataset.labels)
    class_sample_counts = np.bincount(train_labels_arr, minlength=num_classes)
    total_train_samples = len(train_labels_arr)
    raw_weights = total_train_samples / (num_classes * np.maximum(class_sample_counts, 1).astype(np.float32))
    normalized_weights = raw_weights / np.mean(raw_weights)
    class_weights = torch.tensor(normalized_weights, dtype=torch.float32).to(device)

    print("\nClass Weight Summary (Top 5 Rare / Highest Weighted):")
    sorted_indices = np.argsort(normalized_weights)[::-1]
    for idx in sorted_indices[:5]:
        print(f"  - {class_names[idx]:<15}: Count={class_sample_counts[idx]:2d} | Weight={normalized_weights[idx]:.3f}")

    # Optimizer, Loss Function, and Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.08)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training Loop with Early Stopping
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = checkpoint_dir / "best_transformer_gloss.pth"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print("\nStarting Training...")
    print("-" * 75)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^10} | {'Val Acc':^10} | {'LR':^9} | {'Status':^8}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = scheduler.get_last_lr()[0]

        # Model Checkpointing based on Validation Accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "val_loss": best_val_loss,
                    "feature_dim": feature_dim,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "label_map": label_map,
                    "hyperparameters": vars(args),
                },
                best_checkpoint_path,
            )
            status_str = "BEST *"
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            status_str = "Loss Impr"
        else:
            if epoch > 10:  # Allow 10 epochs for representation alignment before incrementing early stopping
                patience_counter += 1
                status_str = f"Patience {patience_counter}/{args.patience}"
            else:
                status_str = "Warmup"

        print(
            f"{epoch:^7d} | {train_loss:^12.4f} | {train_acc * 100:^10.2f}% | {val_loss:^10.4f} | {val_acc * 100:^9.2f}% | {current_lr:^9.6f} | {status_str:^8}",
            flush=True,
        )

        if patience_counter >= args.patience:
            print("-" * 75, flush=True)
            print(f"[Early Stopping] Triggered after {epoch} epochs. No val improvement for {args.patience} epochs.", flush=True)
            break

    print("-" * 75)
    print(f"Training Complete. Best Validation Accuracy: {best_val_acc * 100:.2f}% (Epoch {best_epoch})")
    print(f"Best Checkpoint: {best_checkpoint_path.resolve()}")

    # Plot & Save Training Curves
    plot_training_curves(history, reports_dir)

    # Held-out Test Evaluation
    print("\nLoading best model checkpoint for held-out test evaluation...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_results = evaluate_test_set(
        model=model,
        loader=test_loader,
        class_names=class_names,
        device=device,
        reports_dir=reports_dir,
    )

    # Phase 8: Log Training Run Metadata to DatabaseManager (MongoDB / Local Fallback)
    try:
        from datetime import datetime
        import time
        from src.db_manager import DatabaseManager
        db = DatabaseManager()
        db.log_training_metadata(
            vocab_list=class_names,
            model_metrics={
                "best_val_acc": round(float(best_val_acc), 4),
                "best_val_loss": round(float(best_val_loss), 4),
                "test_top1_acc": round(float(test_results.get("top1_acc", 0.0)), 4),
                "test_top5_acc": round(float(test_results.get("top5_acc", 0.0)), 4),
                "epochs_trained": epoch,
                "best_epoch": best_epoch,
                "hyperparameters": vars(args),
            },
            run_id=f"run_transformer_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        )
        time.sleep(0.5)
        db.close()
        print("[DB Info] Training run metadata successfully recorded to DatabaseManager.")
    except Exception as e:
        print(f"[DB Warning] Could not record training metadata to database: {e}")

    print("\n=== ISL Pipeline Step 5 Completed Successfully ===")


if __name__ == "__main__":
    main()

