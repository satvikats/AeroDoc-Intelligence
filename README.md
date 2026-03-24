# AeroDoc Intelligence 🛩️
### A GenAI Pipeline for Aviation Document Understanding

An end-to-end GenAI pipeline that fine-tunes a transformer model on aviation-domain documents, builds a pre/post-processing pipeline, and serves classifications via a REST API with a local RAG (Retrieval-Augmented Generation) layer — all without using any off-the-shelf hosted AI APIs.

---

## Architecture

```
Raw Documents
     │
     ▼
[Preprocessing Pipeline]
  - Clean & normalize text
  - Chunking strategy (sliding window)
  - Tokenization (BPE via HuggingFace)
     │
     ├──────────────────────────────┐
     ▼                              ▼
[Fine-tuned Classifier]      [Embedding Model]
  DistilBERT                  sentence-transformers
  (6 aviation categories)      (MiniLM-L6-v2)
     │                              │
     ▼                              ▼
[Post-processing]            [FAISS Vector Store]
  Confidence filter               k-NN retrieval
  Label mapping                   Chunk ranking
  Audit logging
     │                              │
     └──────────────┬───────────────┘
                    ▼
            [FastAPI REST Layer]
              /classify
              /search
              /health
```

---

## Project Structure

```
aerodoc-intelligence/
├── data/
│   ├── raw/                    # Raw aviation text documents
│   ├── processed/              # Cleaned & chunked data
│   └── synthetic_generator.py  # Generates labelled training data
├── pipeline/
│   ├── preprocessor.py         # Full text preprocessing pipeline
│   ├── embedder.py             # Embedding + FAISS index builder
│   └── pipeline_runner.py      # Orchestrates end-to-end pipeline
├── models/
│   ├── trainer.py              # Fine-tuning loop with eval
│   ├── classifier.py           # Inference wrapper
│   └── evaluate.py             # Metrics, confusion matrix, plots
├── api/
│   ├── main.py                 # FastAPI app
│   └── schemas.py              # Pydantic request/response models
├── notebooks/
│   └── exploration.ipynb       # EDA, loss curves, attention viz
├── tests/
│   └── test_pipeline.py        # Unit tests for pipeline stages
└── requirements.txt
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic training data
python data/synthetic_generator.py

# 3. Run full pipeline (preprocess → embed → train)
python pipeline/pipeline_runner.py

# 4. Evaluate model
python models/evaluate.py

# 5. Serve the API
uvicorn api.main:app --reload
```

---

## Model Details

- **Base Model**: `distilbert-base-uncased` (66M params, 40% smaller than BERT)
- **Fine-tuning Strategy**: Full fine-tune last 3 transformer blocks + classification head
- **Classes**: maintenance_report | incident_log | technical_manual | safety_bulletin | parts_catalog | inspection_checklist
- **Training**: AdamW optimizer, linear LR warmup, 3 epochs, batch size 16
- **Metrics**: F1 (macro), precision, recall, confusion matrix

---

