"""
api/main.py

FastAPI REST layer exposing:
  POST /classify   — Classify a document text
  POST /search     — Semantic search over indexed documents
  GET  /health     — Health check with model status
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("Run: pip install fastapi uvicorn")

app = FastAPI(
    title="AeroDoc Intelligence API",
    description="Aviation document classification and semantic search pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000, description="Document text to classify")
    anonymize: bool = Field(False, description="Anonymize tail numbers before processing")

class ClassifyResponse(BaseModel):
    predicted_label: str
    confidence: float
    above_threshold: bool
    all_probabilities: dict
    processing_time_ms: float
    preprocessed_text_preview: str

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(5, ge=1, le=20)
    min_score: float = Field(0.3, ge=0.0, le=1.0)

class SearchResult(BaseModel):
    chunk_id: str
    text: str
    label: Optional[str]
    similarity_score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    vector_store_loaded: bool
    vector_count: int
    device: str


# ---------------------------------------------------------------------------
# Model loading (on startup)
# ---------------------------------------------------------------------------

classifier = None
retriever = None


@app.on_event("startup")
async def load_models():
    global classifier, retriever

    # Load classifier
    model_dir = "saved_model"
    if Path(f"{model_dir}/config.json").exists():
        try:
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            import sys
            sys.path.insert(0, ".")
            from preprocessor import AeroDocPreprocessor, PredictionPostProcessor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
            model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
            model.eval()

            classifier = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "preprocessor": AeroDocPreprocessor(),
                "post_processor": PredictionPostProcessor(confidence_threshold=0.75),
            }
            logger.info("Classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load classifier: {e}")
    else:
        logger.warning(f"No saved model at {model_dir}. Run pipeline_runner.py first.")

    # Load retriever
    vector_dir = "vector_store"
    if Path(f"{vector_dir}/vectors.faiss").exists():
        try:
            from embedder import RAGRetriever
            retriever = RAGRetriever(vector_dir)
            logger.info("RAG retriever loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load retriever: {e}")
    else:
        logger.warning(f"No vector store at {vector_dir}. Run pipeline_runner.py first.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        classifier_loaded=classifier is not None,
        vector_store_loaded=retriever is not None,
        vector_count=retriever.store.index.ntotal if retriever else 0,
        device=str(classifier["device"]) if classifier else "none",
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify_document(request: ClassifyRequest):
    if classifier is None:
        raise HTTPException(503, detail="Classifier not loaded. Run the pipeline first.")

    start = time.time()
    preprocessor = classifier["preprocessor"]
    post_processor = classifier["post_processor"]

    # Pre-processing
    from preprocessor import Document
    doc = Document(doc_id="api_req", raw_text=request.text)
    cleaned = preprocessor.cleaner.clean(request.text, anonymize=request.anonymize)

    # Tokenize and infer
    enc = classifier["tokenizer"](
        cleaned,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = classifier["model"](
            input_ids=enc["input_ids"].to(classifier["device"]),
            attention_mask=enc["attention_mask"].to(classifier["device"]),
        )

    # Post-processing
    logits = output.logits[0].cpu().tolist()
    result = post_processor.process_logits(logits, doc_id="api_req")

    elapsed_ms = (time.time() - start) * 1000

    return ClassifyResponse(
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        above_threshold=result["above_threshold"],
        all_probabilities=result["all_probabilities"],
        processing_time_ms=round(elapsed_ms, 2),
        preprocessed_text_preview=cleaned[:200] + ("..." if len(cleaned) > 200 else ""),
    )


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    if retriever is None:
        raise HTTPException(503, detail="Vector store not loaded. Run the pipeline first.")

    start = time.time()
    results = retriever.retrieve(request.query, top_k=request.top_k, min_score=request.min_score)
    elapsed_ms = (time.time() - start) * 1000

    return SearchResponse(
        query=request.query,
        results=[
            SearchResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                label=r.get("label"),
                similarity_score=round(r["similarity_score"], 4),
            )
            for r in results
        ],
        total_results=len(results),
        processing_time_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
