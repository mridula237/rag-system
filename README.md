# RAG System

A production-grade Retrieval Augmented Generation system that answers questions over documents with proper citations, hybrid search, and contextual retrieval.

## Architecture

```
PDF/HTML → Chunking (400 tokens, 80 overlap)
                ↓
    Embedding (text-embedding-3-small, 1536 dims)
                ↓
    pgvector (HNSW) + Qdrant (cosine)
                ↓
    BM25 + Vector Search → RRF Fusion → Cohere Rerank (top 50 → top 6)
                ↓
    LLM Generation (GPT-5.4-nano)
                ↓
    {answer, citations: [doc_id, chunk_index]}
```

## Stack
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector stores:** pgvector (HNSW) + Qdrant
- **Keyword search:** BM25 (rank-bm25)
- **Fusion:** Reciprocal Rank Fusion (RRF)
- **Reranker:** Cohere Rerank 3
- **Generation:** GPT-5.4-nano with JSON output + citations
- **Contextual retrieval:** Claude Haiku generates chunk context before embedding
- **Web fallback:** Tavily search when retrieval score < 0.1
- **API:** FastAPI with streaming SSE
- **UI:** Streamlit chat with citations panel

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export COHERE_API_KEY=...
export TAVILY_API_KEY=...

# start databases
docker compose up -d

# download corpus
python download_corpus.py

# ingest (standard)
python ingest.py

# ingest (contextual retrieval)
python contextual_ingest.py
```

## Usage

```bash
# API
uvicorn api:app --reload
# POST /query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is prompt caching?", "use_contextual": true, "stream": false, "tenant_id": "default"}'

# UI
streamlit run app.py

# CLI
python generate.py

# Check for doc updates
python doc_tracker.py
```

## Evaluation Results (30 Q&A pairs)

| Metric | Standard | Contextual | Δ |
|---|---|---|---|
| Recall@5 | 0.525 | 0.570 | +8.6% |
| MRR | 0.820 | 0.868 | +5.9% |
| Faithfulness | 0.709 | 0.864 | +21.9% |
| Avg latency | 1.65s | 1.18s | -28.5% |

Contextual retrieval improved all four metrics. Faithfulness saw the largest gain (+21.9%).

## Corpus
- 12 HTML pages: Anthropic + OpenAI documentation
- 515 total chunks (400 tokens, 80 overlap)
- Stored in both pgvector and Qdrant

## Stretch Goals
- **Web search fallback** — Tavily search triggers when top rerank score < 0.1
- **Document update detection** — MD5 hash tracking; only changed docs are re-embedded
- **Multi-tenant scoping** — `tenant_id` field on all chunks; searches filtered per tenant

## Project Structure
```
ingest.py                Standard ingestion
contextual_ingest.py     Contextual retrieval ingestion
retrieve.py              BM25 + vector + RRF + rerank
generate.py              LLM generation with citations
eval_rag.py              Standard eval harness
eval_contextual.py       Contextual eval + before/after comparison
api.py                   FastAPI POST /query (streaming + tenant scoping)
app.py                   Streamlit chat UI
web_search.py            Tavily web search fallback
doc_tracker.py           Document update detection
tenant.py                Multi-tenant scoping utilities
download_corpus.py       Corpus downloader
config.py                Config and constants
docker-compose.yml       pgvector + Qdrant
```