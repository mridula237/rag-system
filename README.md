# RAG System

A production-grade Retrieval Augmented Generation system that answers questions over documents with citations. Built on Anthropic + OpenAI documentation as the corpus.

## Architecture

```
PDF/HTML → Chunking → Embedding (text-embedding-3-small)
                            ↓
                    pgvector + Qdrant
                            ↓
         BM25 + Vector Search → RRF Fusion → Cohere Rerank
                            ↓
              LLM Generation (GPT-5.4-nano)
                            ↓
              {answer, citations: [doc_id, chunk_id]}
```

## Stack
- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dims)
- **Vector stores:** pgvector (HNSW index) + Qdrant
- **Keyword search:** BM25 (rank-bm25)
- **Fusion:** Reciprocal Rank Fusion (RRF)
- **Reranker:** Cohere Rerank 3 (top 50 → top 6)
- **Generation:** GPT-5.4-nano with JSON output + citations
- **Contextual retrieval:** Claude Haiku generates chunk context before embedding
- **API:** FastAPI with streaming SSE
- **UI:** Streamlit chat with citations panel

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export COHERE_API_KEY=...

# start databases
docker compose up -d

# download corpus (Anthropic + OpenAI docs)
python download_corpus.py

# ingest + embed (standard)
python ingest.py

# ingest + embed (contextual retrieval)
python contextual_ingest.py
```

## Usage

```bash
# CLI
python generate.py

# API
uvicorn api:app --reload
# POST /query with {"query": "...", "use_contextual": true, "stream": false}

# UI
streamlit run app.py
```

## Evaluation Results (30 Q&A pairs)

| Metric | Standard | Contextual | Δ |
|---|---|---|---|
| Recall@5 | 0.525 | 0.570 | +8.6% |
| MRR | 0.820 | 0.868 | +5.9% |
| Faithfulness | 0.709 | 0.864 | +21.9% |
| Avg latency | 1.65s | 1.18s | -28.5% |

Contextual retrieval (Anthropic's pattern) improved all four metrics. Faithfulness saw the largest gain (+21.9%) because contextual chunks give the LLM better grounding.

## Corpus
- 12 HTML pages from Anthropic and OpenAI documentation
- 515 total chunks (400 tokens, 80 overlap)
- Stored in both pgvector and Qdrant for comparison

## Project structure
```
ingest.py              Standard ingestion pipeline
contextual_ingest.py   Contextual retrieval ingestion
retrieve.py            BM25 + vector + RRF + rerank
generate.py            LLM generation with citations
eval_rag.py            Standard retrieval eval
eval_contextual.py     Contextual retrieval eval
api.py                 FastAPI POST /query (streaming)
app.py                 Streamlit chat UI
download_corpus.py     Corpus downloader
config.py              Config and constants
docker-compose.yml     pgvector + Qdrant
```