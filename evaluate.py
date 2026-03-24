"""
models/evaluate.py

Full evaluation report: F1, precision, recall, confusion matrix, confidence distribution.
Generates plots saved to models/eval_outputs/.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LABELS = [
    "maintenance_report",
    "incident_log",
    "technical_manual",
    "safety_bulletin",
    "parts_catalog",
    "inspection_checklist",
]


def evaluate_model(model_dir: str = "saved_model", data_path: str = "processed/chunks.jsonl"):
    """Run full evaluation on the saved model."""
    try:
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error("Install transformers: pip install transformers torch")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(f"{model_dir}/config.json").exists():
        logger.error(f"No saved model found at {model_dir}. Run trainer.py first.")
        return

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    # Load test split (last 10%)
    data = []
    with open(data_path) as f:
        for line in f:
            row = json.loads(line.strip())
            if row.get("label_id") is not None:
                data.append(row)

    test_data = data[int(len(data) * 0.9):]
    logger.info(f"Evaluating on {len(test_data)} test samples")

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for item in test_data:
            enc = tokenizer(
                item["text"],
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            output = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            probs = torch.softmax(output.logits, dim=-1).cpu().numpy()[0]
            pred = probs.argmax()
            all_preds.append(int(pred))
            all_labels.append(int(item["label_id"]))
            all_probs.append(probs.tolist())

    _print_classification_report(all_labels, all_preds)
    _save_confidence_data(all_probs, all_preds, all_labels)
    _plot_results(all_labels, all_preds, all_probs)


def _print_classification_report(labels: List[int], preds: List[int]):
    """Print per-class precision, recall, F1."""
    n_classes = len(LABELS)
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    for true, pred in zip(labels, preds):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(f"{'Class':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)

    macro_f1 = 0.0
    for i, label in enumerate(LABELS):
        precision = tp[i] / (tp[i] + fp[i] + 1e-8)
        recall = tp[i] / (tp[i] + fn[i] + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        support = tp[i] + fn[i]
        macro_f1 += f1
        print(f"{label:<30} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")

    macro_f1 /= n_classes
    accuracy = sum(1 for t, p in zip(labels, preds) if t == p) / len(labels)
    print("=" * 70)
    print(f"{'Macro F1':<30} {macro_f1:>10.3f}")
    print(f"{'Accuracy':<30} {accuracy:>10.3f}")
    print("=" * 70)

    return {"macro_f1": macro_f1, "accuracy": accuracy}


def _save_confidence_data(probs, preds, labels):
    """Save confidence data for analysis."""
    Path("eval_outputs").mkdir(parents=True, exist_ok=True)
    output = {
        "predictions": preds,
        "true_labels": labels,
        "probabilities": probs,
        "labels": LABELS,
    }
    with open("eval_outputs/predictions.json", "w") as f:
        json.dump(output, f)
    logger.info("Predictions saved → models/eval_outputs/predictions.json")


def _plot_results(labels: List[int], preds: List[int], probs: List[List[float]]):
    """Generate confusion matrix and confidence histogram if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        logger.info("matplotlib not available — skipping plots")
        return

    Path("eval_outputs").mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    n = len(LABELS)
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true][pred] += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Confusion matrix heatmap
    ax = axes[0]
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_labels = [l.replace("_", "\n") for l in LABELS]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() * 0.6 else "black", fontsize=9)

    # Plot 2: Confidence distribution
    ax2 = axes[1]
    correct_conf = [max(probs[i]) for i in range(len(preds)) if preds[i] == labels[i]]
    wrong_conf = [max(probs[i]) for i in range(len(preds)) if preds[i] != labels[i]]
    ax2.hist(correct_conf, bins=20, alpha=0.7, label="Correct", color="steelblue")
    ax2.hist(wrong_conf, bins=20, alpha=0.7, label="Incorrect", color="coral")
    ax2.axvline(0.75, color="red", linestyle="--", label="Threshold (0.75)")
    ax2.set_xlabel("Confidence Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence Distribution")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("eval_outputs/evaluation.png", dpi=150, bbox_inches="tight")
    logger.info("Plots saved → models/eval_outputs/evaluation.png")
    plt.close()


if __name__ == "__main__":
    evaluate_model()
