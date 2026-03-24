"""
models/trainer.py

Fine-tunes DistilBERT on aviation document classification.

Key design decisions explained (for interview):
- Why DistilBERT: 40% smaller than BERT, 60% faster, retains 97% of BERT's performance
- Why freeze early layers: lower transformer layers encode general syntax — no need to retrain
- Why AdamW + linear warmup: standard recipe for fine-tuning transformers
- Why class weighting: aviation datasets are often imbalanced; safety bulletins are rarer than manuals
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABELS = [
    "maintenance_report",
    "incident_log",
    "technical_manual",
    "safety_bulletin",
    "parts_catalog",
    "inspection_checklist",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AeroDocDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label_id"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1       # 10% of total steps for LR warmup
    weight_decay: float = 0.01
    freeze_layers: int = 3          # Freeze bottom N transformer layers
    confidence_threshold: float = 0.75
    output_dir: str = "saved_model"
    train_split: float = 0.8
    val_split: float = 0.1
    # test_split is implied: 1 - train - val


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AeroDocTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(config.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        ).to(self.device)

        self._freeze_layers()
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _freeze_layers(self):
        """
        Freeze the first N transformer layers.

        Intuition: The lower layers of DistilBERT learn general linguistic features
        (part-of-speech, grammar). We only want to adapt the higher layers and the
        classification head to our domain.
        """
        for i, layer in enumerate(self.model.distilbert.transformer.layer):
            if i < self.config.freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Frozen params: {frozen:,} | Trainable params: {trainable:,}")

    def load_data(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load JSONL data and split into train/val/test loaders."""
        data = []
        with open(data_path) as f:
            for line in f:
                row = json.loads(line.strip())
                if row.get("label_id") is not None:
                    data.append(row)

        n = len(data)
        n_train = int(n * self.config.train_split)
        n_val = int(n * self.config.val_split)

        import random
        random.shuffle(data)
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]

        logger.info(f"Split: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test")

        def make_loader(split, shuffle):
            dataset = AeroDocDataset(split, self.tokenizer, self.config.max_length)
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

        return make_loader(train_data, True), make_loader(val_data, False), make_loader(test_data, False)

    def _compute_class_weights(self, train_loader: DataLoader) -> torch.Tensor:
        """
        Compute inverse-frequency class weights to handle class imbalance.
        Safety bulletins should carry more weight if they're underrepresented.
        """
        counts = torch.zeros(len(LABELS))
        for batch in train_loader:
            for label_id in batch["labels"]:
                counts[label_id.item()] += 1
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum() * len(LABELS)
        return weights.to(self.device)

    def train(self, data_path: str):
        train_loader, val_loader, test_loader = self.load_data(data_path)

        class_weights = self._compute_class_weights(train_loader)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        logger.info(f"Training: {total_steps} steps, {warmup_steps} warmup steps")

        best_val_loss = float("inf")

        for epoch in range(1, self.config.num_epochs + 1):
            # --- Train ---
            self.model.train()
            total_train_loss = 0.0
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()

                # Gradient clipping — prevents exploding gradients in fine-tuning
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

                if (step + 1) % 20 == 0:
                    logger.info(
                        f"Epoch {epoch} | Step {step+1}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

            avg_train_loss = total_train_loss / len(train_loader)
            self.history["train_loss"].append(avg_train_loss)

            # --- Validate ---
            val_loss, val_acc = self._evaluate(val_loader, loss_fn)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model()
                logger.info(f"  ↳ New best model saved (val_loss={val_loss:.4f})")

        logger.info("Training complete.")
        return test_loader

    def _evaluate(self, loader: DataLoader, loss_fn) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                total_loss += loss.item()

                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(loader), correct / total

    def _save_model(self):
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        with open(f"{self.config.output_dir}/training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Model saved → {self.config.output_dir}")


if __name__ == "__main__":
    config = TrainingConfig(
        num_epochs=3,
        batch_size=16,
        freeze_layers=3,
    )
    trainer = AeroDocTrainer(config)
    test_loader = trainer.train("data/processed/chunks.jsonl")
    logger.info("Run models/evaluate.py for full metrics report.")
