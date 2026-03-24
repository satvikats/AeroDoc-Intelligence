"""
pipeline/preprocessor.py

Pre-processing pipeline for aviation documents.
Handles: cleaning → normalization → chunking → tokenization-ready output.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Document:
    doc_id: str
    raw_text: str
    label: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProcessedChunk:
    chunk_id: str
    doc_id: str
    text: str
    label: Optional[str] = None
    token_count: int = 0
    chunk_index: int = 0


# ---------------------------------------------------------------------------
# Stage 1: Text Cleaner
# ---------------------------------------------------------------------------

class TextCleaner:
    """
    Normalizes raw aviation text:
    - Strips boilerplate headers/footers
    - Normalizes whitespace and Unicode
    - Expands aviation abbreviations to help the tokenizer
    - Removes PII-like patterns (tail numbers → <TAIL_NUM>)
    """

    # Aviation abbreviations expanded so the model sees full words
    ABBREVIATION_MAP = {
        r"\bMEL\b": "minimum equipment list",
        r"\bAOG\b": "aircraft on ground",
        r"\bMTBF\b": "mean time between failures",
        r"\bAMM\b": "aircraft maintenance manual",
        r"\bIPC\b": "illustrated parts catalog",
        r"\bCMM\b": "component maintenance manual",
        r"\bMRO\b": "maintenance repair overhaul",
        r"\bSDR\b": "service difficulty report",
        r"\bAPU\b": "auxiliary power unit",
        r"\bIDG\b": "integrated drive generator",
        r"\bFOD\b": "foreign object damage",
        r"\bNLG\b": "nose landing gear",
        r"\bMLG\b": "main landing gear",
        r"\bEICAS\b": "engine indicating and crew alerting system",
        r"\bBITE\b": "built-in test equipment",
        r"\bFL(\d+)\b": r"flight level \1",
        r"\bPSI\b": "pounds per square inch",
        r"\bFH\b": "flight hours",
    }

    # Pseudo-anonymization: replace tail numbers with token
    TAIL_RE = re.compile(
        r"\b([A-Z]{1,2}-[A-Z]{3,5}|N[0-9]{1,5}[A-Z]{0,2}|[A-Z]{2}[0-9]{3}[A-Z]?)\b"
    )

    def clean(self, text: str, anonymize: bool = False) -> str:
        # Normalize unicode: curly quotes → straight, em-dash → hyphen
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2014", " - ").replace("\u2013", " - ")

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Expand abbreviations
        for pattern, replacement in self.ABBREVIATION_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Optionally anonymize tail numbers for sensitive environments
        if anonymize:
            text = self.TAIL_RE.sub("<TAIL_NUM>", text)

        return text


# ---------------------------------------------------------------------------
# Stage 2: Chunker
# ---------------------------------------------------------------------------

class SlidingWindowChunker:
    """
    Splits documents into overlapping chunks using a sliding window.

    Why sliding window?
    - Aviation documents can be long (maintenance manuals = thousands of words)
    - BERT-family models have a 512-token limit
    - Overlap ensures sentences at chunk boundaries are represented in at least one full chunk
    """

    def __init__(self, max_words: int = 100, stride: int = 50):
        self.max_words = max_words
        self.stride = stride  # how many words to advance each step (overlap = max-stride)

    def chunk(self, doc: Document) -> List[ProcessedChunk]:
        words = doc.raw_text.split()
        if len(words) <= self.max_words:
            return [
                ProcessedChunk(
                    chunk_id=f"{doc.doc_id}_0",
                    doc_id=doc.doc_id,
                    text=doc.raw_text,
                    label=doc.label,
                    chunk_index=0,
                    token_count=len(words),
                )
            ]

        chunks = []
        start = 0
        idx = 0
        while start < len(words):
            end = min(start + self.max_words, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                ProcessedChunk(
                    chunk_id=f"{doc.doc_id}_{idx}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    label=doc.label,
                    chunk_index=idx,
                    token_count=end - start,
                )
            )
            if end == len(words):
                break
            start += self.stride
            idx += 1

        return chunks


# ---------------------------------------------------------------------------
# Stage 3: Label Encoder
# ---------------------------------------------------------------------------

class LabelEncoder:
    LABELS = [
        "maintenance_report",
        "incident_log",
        "technical_manual",
        "safety_bulletin",
        "parts_catalog",
        "inspection_checklist",
    ]

    def __init__(self):
        self.label2id = {label: i for i, label in enumerate(self.LABELS)}
        self.id2label = {i: label for i, label in enumerate(self.LABELS)}

    def encode(self, label: str) -> int:
        if label not in self.label2id:
            raise ValueError(f"Unknown label: {label}. Valid: {self.LABELS}")
        return self.label2id[label]

    def decode(self, idx: int) -> str:
        return self.id2label[idx]


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class AeroDocPreprocessor:
    """
    Orchestrates all preprocessing stages.

    Usage:
        processor = AeroDocPreprocessor()
        chunks = processor.process_file("processed/dataset.jsonl")
        processor.save_chunks(chunks, "processed/chunks.jsonl")
    """

    def __init__(self, max_words: int = 100, stride: int = 50, anonymize: bool = False):
        self.cleaner = TextCleaner()
        self.chunker = SlidingWindowChunker(max_words=max_words, stride=stride)
        self.label_encoder = LabelEncoder()
        self.anonymize = anonymize

    def process_document(self, doc: Document) -> List[ProcessedChunk]:
        cleaned_text = self.cleaner.clean(doc.raw_text, anonymize=self.anonymize)
        doc.raw_text = cleaned_text
        chunks = self.chunker.chunk(doc)
        return chunks

    def process_file(self, input_path: str) -> Tuple[List[ProcessedChunk], Dict]:
        """Process a JSONL file of raw documents."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        chunks = []
        stats = {"total_docs": 0, "total_chunks": 0, "label_counts": {}}

        with open(input_path) as f:
            for line in f:
                row = json.loads(line.strip())
                doc = Document(
                    doc_id=f"doc_{stats['total_docs']}",
                    raw_text=row["text"],
                    label=row.get("label"),
                )
                doc_chunks = self.process_document(doc)
                chunks.extend(doc_chunks)

                stats["total_docs"] += 1
                label = row.get("label", "unknown")
                stats["label_counts"][label] = stats["label_counts"].get(label, 0) + 1

        stats["total_chunks"] = len(chunks)
        logger.info(f"Processed {stats['total_docs']} docs → {stats['total_chunks']} chunks")
        logger.info(f"Label distribution: {stats['label_counts']}")

        return chunks, stats

    def save_chunks(self, chunks: List[ProcessedChunk], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for chunk in chunks:
                record = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "label": chunk.label,
                    "label_id": self.label_encoder.encode(chunk.label) if chunk.label else None,
                    "token_count": chunk.token_count,
                    "chunk_index": chunk.chunk_index,
                }
                f.write(json.dumps(record) + "\n")
        logger.info(f"Saved {len(chunks)} chunks → {output_path}")


# ---------------------------------------------------------------------------
# Post-processing utilities
# ---------------------------------------------------------------------------

class PredictionPostProcessor:
    """
    Post-processes raw model logits into clean, audited predictions.

    Handles:
    - Confidence thresholding (reject low-confidence predictions)
    - Ensemble voting across chunks of the same document
    - Structured output with audit trail
    """

    def __init__(self, confidence_threshold: float = 0.75):
        self.threshold = confidence_threshold
        self.label_encoder = LabelEncoder()

    def process_logits(self, logits: List[float], doc_id: str) -> Dict:
        """Convert raw logits to a structured prediction with confidence."""
        import math

        # Softmax
        exp_logits = [math.exp(l) for l in logits]
        total = sum(exp_logits)
        probs = [e / total for e in exp_logits]

        max_prob = max(probs)
        pred_idx = probs.index(max_prob)
        pred_label = self.label_encoder.decode(pred_idx)

        result = {
            "doc_id": doc_id,
            "predicted_label": pred_label if max_prob >= self.threshold else "uncertain",
            "confidence": round(max_prob, 4),
            "above_threshold": max_prob >= self.threshold,
            "all_probabilities": {
                self.label_encoder.decode(i): round(p, 4) for i, p in enumerate(probs)
            },
            "threshold_used": self.threshold,
        }

        if not result["above_threshold"]:
            logger.warning(
                f"Low confidence prediction for {doc_id}: {max_prob:.3f} < {self.threshold}"
            )

        return result

    def aggregate_chunks(self, chunk_predictions: List[Dict]) -> Dict:
        """
        Aggregate predictions across multiple chunks of the same document.
        Strategy: weighted vote by confidence.
        """
        if not chunk_predictions:
            return {}

        doc_id = chunk_predictions[0]["doc_id"].rsplit("_", 1)[0]
        label_scores: Dict[str, float] = {}

        for pred in chunk_predictions:
            if not pred["above_threshold"]:
                continue
            label = pred["predicted_label"]
            label_scores[label] = label_scores.get(label, 0) + pred["confidence"]

        if not label_scores:
            return {"doc_id": doc_id, "predicted_label": "uncertain", "confidence": 0.0}

        best_label = max(label_scores, key=label_scores.get)
        total_score = sum(label_scores.values())
        confidence = label_scores[best_label] / total_score

        return {
            "doc_id": doc_id,
            "predicted_label": best_label,
            "confidence": round(confidence, 4),
            "chunk_count": len(chunk_predictions),
            "agreeing_chunks": sum(
                1 for p in chunk_predictions if p.get("predicted_label") == best_label
            ),
        }


if __name__ == "__main__":
    # Quick smoke test
    processor = AeroDocPreprocessor(max_words=80, stride=40)

    sample_doc = Document(
        doc_id="test_001",
        raw_text="Aircraft N-XRAY42 underwent A-check maintenance. MEL item 27-10-01A cleared. "
                 "Engine oil consumption within limits. AOG status cleared. "
                 "All deferred items reviewed and resolved. Aircraft returned to service.",
        label="maintenance_report",
    )

    chunks = processor.process_document(sample_doc)
    print(f"\nGenerated {len(chunks)} chunks from sample document:")
    for c in chunks:
        print(f"  [{c.chunk_id}] {c.text[:80]}...")
