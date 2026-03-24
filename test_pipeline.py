"""
tests/test_pipeline.py

Unit tests for preprocessing and post-processing pipeline stages.
Run with: pytest tests/
"""

import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessor import (
    TextCleaner,
    SlidingWindowChunker,
    LabelEncoder,
    Document,
    PredictionPostProcessor,
)


# ---------------------------------------------------------------------------
# TextCleaner tests
# ---------------------------------------------------------------------------

class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_expands_mel_abbreviation(self):
        text = "MEL item 27-10-01A cleared."
        result = self.cleaner.clean(text)
        assert "minimum equipment list" in result.lower()

    def test_expands_aog_abbreviation(self):
        text = "Aircraft is AOG at OMDB."
        result = self.cleaner.clean(text)
        assert "aircraft on ground" in result.lower()

    def test_expands_flight_level(self):
        text = "Flame-out occurred at FL350."
        result = self.cleaner.clean(text)
        assert "flight level" in result.lower()
        assert "350" in result

    def test_collapses_whitespace(self):
        text = "  Lots   of    spaces   here.  "
        result = self.cleaner.clean(text)
        assert "  " not in result
        assert result == result.strip()

    def test_normalizes_curly_quotes(self):
        text = "\u2018Hello\u2019 and \u201cWorld\u201d"
        result = self.cleaner.clean(text)
        assert "\u2018" not in result
        assert "\u2019" not in result

    def test_anonymize_tail_number(self):
        text = "Aircraft N12345 underwent maintenance."
        result = self.cleaner.clean(text, anonymize=True)
        assert "<TAIL_NUM>" in result

    def test_no_anonymize_by_default(self):
        text = "Aircraft G-BAVO underwent maintenance."
        result = self.cleaner.clean(text, anonymize=False)
        assert "G-BAVO" in result


# ---------------------------------------------------------------------------
# SlidingWindowChunker tests
# ---------------------------------------------------------------------------

class TestSlidingWindowChunker:
    def setup_method(self):
        self.chunker = SlidingWindowChunker(max_words=10, stride=5)

    def test_short_doc_returns_single_chunk(self):
        doc = Document(doc_id="d1", raw_text="Short document with few words.", label="incident_log")
        chunks = self.chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "d1_0"

    def test_long_doc_returns_multiple_chunks(self):
        words = ["word"] * 25
        doc = Document(doc_id="d2", raw_text=" ".join(words), label="maintenance_report")
        chunks = self.chunker.chunk(doc)
        assert len(chunks) > 1

    def test_chunk_ids_are_sequential(self):
        words = ["word"] * 30
        doc = Document(doc_id="d3", raw_text=" ".join(words), label="technical_manual")
        chunks = self.chunker.chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"d3_{i}"

    def test_chunks_inherit_label(self):
        doc = Document(doc_id="d4", raw_text=" ".join(["w"] * 25), label="safety_bulletin")
        chunks = self.chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.label == "safety_bulletin"

    def test_overlap_exists(self):
        # With max_words=10, stride=5: chunk 0 words 0-9, chunk 1 words 5-14
        # So words 5-9 appear in both
        words = [str(i) for i in range(20)]
        doc = Document(doc_id="d5", raw_text=" ".join(words), label="parts_catalog")
        chunks = self.chunker.chunk(doc)
        chunk0_words = set(chunks[0].text.split())
        chunk1_words = set(chunks[1].text.split())
        overlap = chunk0_words & chunk1_words
        assert len(overlap) > 0, "Expected overlapping words between adjacent chunks"


# ---------------------------------------------------------------------------
# LabelEncoder tests
# ---------------------------------------------------------------------------

class TestLabelEncoder:
    def setup_method(self):
        self.encoder = LabelEncoder()

    def test_encode_all_labels(self):
        for label in LabelEncoder.LABELS:
            idx = self.encoder.encode(label)
            assert isinstance(idx, int)
            assert 0 <= idx < len(LabelEncoder.LABELS)

    def test_decode_roundtrip(self):
        for label in LabelEncoder.LABELS:
            idx = self.encoder.encode(label)
            decoded = self.encoder.decode(idx)
            assert decoded == label

    def test_unknown_label_raises(self):
        try:
            self.encoder.encode("not_a_real_label")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# PredictionPostProcessor tests
# ---------------------------------------------------------------------------

class TestPredictionPostProcessor:
    def setup_method(self):
        self.processor = PredictionPostProcessor(confidence_threshold=0.75)

    def _make_logits_for(self, class_idx: int, high: float = 3.0):
        """Create logits that will softmax to high probability for class_idx."""
        logits = [-1.0] * 6
        logits[class_idx] = high
        return logits

    def test_high_confidence_above_threshold(self):
        logits = self._make_logits_for(0, high=5.0)
        result = self.processor.process_logits(logits, doc_id="doc_1")
        assert result["above_threshold"] is True
        assert result["confidence"] > 0.75

    def test_low_confidence_below_threshold(self):
        # Uniform logits → uniform probability ≈ 0.167 each
        logits = [1.0] * 6
        result = self.processor.process_logits(logits, doc_id="doc_2")
        assert result["above_threshold"] is False
        assert result["predicted_label"] == "uncertain"

    def test_probabilities_sum_to_one(self):
        logits = self._make_logits_for(2)
        result = self.processor.process_logits(logits, doc_id="doc_3")
        total = sum(result["all_probabilities"].values())
        assert abs(total - 1.0) < 1e-4

    def test_all_classes_present_in_output(self):
        logits = self._make_logits_for(1)
        result = self.processor.process_logits(logits, doc_id="doc_4")
        for label in LabelEncoder.LABELS:
            assert label in result["all_probabilities"]

    def test_aggregate_chunks_majority_wins(self):
        preds = [
            {"doc_id": "doc_5_0", "predicted_label": "maintenance_report", "confidence": 0.9, "above_threshold": True},
            {"doc_id": "doc_5_1", "predicted_label": "maintenance_report", "confidence": 0.85, "above_threshold": True},
            {"doc_id": "doc_5_2", "predicted_label": "incident_log", "confidence": 0.8, "above_threshold": True},
        ]
        result = self.processor.aggregate_chunks(preds)
        assert result["predicted_label"] == "maintenance_report"
        assert result["agreeing_chunks"] == 2

    def test_aggregate_skips_low_confidence_chunks(self):
        preds = [
            {"doc_id": "doc_6_0", "predicted_label": "uncertain", "confidence": 0.3, "above_threshold": False},
            {"doc_id": "doc_6_1", "predicted_label": "safety_bulletin", "confidence": 0.9, "above_threshold": True},
        ]
        result = self.processor.aggregate_chunks(preds)
        assert result["predicted_label"] == "safety_bulletin"


if __name__ == "__main__":
    # Run without pytest
    import traceback
    test_classes = [TestTextCleaner, TestSlidingWindowChunker, TestLabelEncoder, TestPredictionPostProcessor]
    passed = 0
    failed = 0
    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                instance.setup_method()
                try:
                    getattr(instance, method_name)()
                    print(f"  PASS  {cls.__name__}.{method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL  {cls.__name__}.{method_name}: {e}")
                    traceback.print_exc()
                    failed += 1
    print(f"\n{passed} passed, {failed} failed")
