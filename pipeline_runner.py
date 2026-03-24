"""
pipeline_runner.py

Orchestrates the full data engineering pipeline:
  Raw JSONL → Preprocessing → Fine-tuning → Embedding → FAISS Index
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)


def timed_stage(name: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            logger.info(f"\n{'='*60}")
            logger.info(f"STAGE: {name}")
            logger.info(f"{'='*60}")
            start = time.time()
            result = fn(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"STAGE COMPLETE: {name} — {elapsed:.2f}s\n")
            return result
        return wrapper
    return decorator


@timed_stage("1. Generate Synthetic Data")
def stage_generate_data():
    from synthetic_generator import generate_dataset
    Path("processed").mkdir(exist_ok=True)
    generate_dataset(n_samples=900, output_path="processed/dataset.jsonl")


@timed_stage("2. Preprocess Documents")
def stage_preprocess():
    from preprocessor import AeroDocPreprocessor
    processor = AeroDocPreprocessor(max_words=100, stride=50, anonymize=False)
    chunks, stats = processor.process_file("processed/dataset.jsonl")
    processor.save_chunks(chunks, "processed/chunks.jsonl")
    return stats


@timed_stage("3. Fine-tune DistilBERT Classifier")
def stage_train():
    from trainer import AeroDocTrainer, TrainingConfig
    config = TrainingConfig(
        num_epochs=3,
        batch_size=16,
        freeze_layers=3,
        output_dir="saved_model",
    )
    trainer = AeroDocTrainer(config)
    trainer.train("processed/chunks.jsonl")


@timed_stage("4. Build FAISS Vector Index")
def stage_build_index():
    from embedder import IndexBuilder
    builder = IndexBuilder()
    builder.build(
        chunks_path="processed/chunks.jsonl",
        output_dir="vector_store",
    )


def run_pipeline(skip_train: bool = False):
    logger.info("AeroDoc Intelligence — Full Pipeline Run")
    pipeline_start = time.time()

    stage_generate_data()
    stats = stage_preprocess()

    if not skip_train:
        stage_train()
    else:
        logger.info("Skipping training (--skip-train flag set)")

    stage_build_index()

    total_time = time.time() - pipeline_start
    logger.info(f"\nPipeline complete in {total_time:.1f}s")
    logger.info(f"Documents processed: {stats.get('total_docs', 0)}")
    logger.info(f"Chunks generated: {stats.get('total_chunks', 0)}")
    logger.info("Ready: run  uvicorn main:app --reload")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    run_pipeline(skip_train=args.skip_train)
