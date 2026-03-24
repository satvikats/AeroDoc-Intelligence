# AeroDoc Intelligence 🛩️
### A GenAI Pipeline for Aviation Document Understanding

An end-to-end GenAI pipeline that fine-tunes a transformer model on aviation-domain documents, builds a pre/post-processing pipeline, and serves classifications via a REST API with a local RAG (Retrieval-Augmented Generation) layer — all without using any off-the-shelf hosted AI APIs.

---

## What This Demonstrates

| Skill | How It's Shown |
|---|---|
| Fine-tuning LLMs | DistilBERT fine-tuned on custom aviation text classes |
| Pre/Post Processing | Text cleaning → tokenization → model → threshold filtering |
| Data Engineering Pipelines | Modular pipelåine with logging, versioning, error handling |
| GenAI Architecture | Local RAG with FAISS vector store + retrieval |
| NLP Core Concepts | Embeddings, attention, tokenization, BPE |å
| ML Concepts | Loss curves, eval metrics, confusion matrix |
| Frameworks/Tools | PyTorch, HuggingFace Transformers, FAISS, FastAPI |

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

## Interview Talking Points

**"Why DistilBERT and not GPT?"**
For classification tasks, encoder-only models like DistilBERT are more efficient and interpretable. We don't need generation — we need discriminative features. Also, in data-sensitive environments (like aviation), you avoid sending data to external APIs.

**"What does fine-tuning actually change?"**
The pre-trained model learns general language representations. Fine-tuning shifts the final layers' weights to recognize domain-specific patterns in aviation text — abbreviations like MEL, AOG, MTBF, squawk codes — that the base model has never prioritized.

**"How does the RAG layer work?"**
Documents are chunked, passed through an embedding model, and stored in a FAISS index. At query time, the query is embedded and cosine similarity retrieves the top-k relevant chunks. These are injected into context for downstream tasks — no model retraining needed to add new documents.

**"Why FAISS instead of a vector DB?"**
In air-gapped or data-classified environments, you can't use cloud vector DBs. FAISS is open-source, runs fully locally, and scales to millions of vectors on a single machine.
