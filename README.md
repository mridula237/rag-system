# RAG System

A production-grade Retrieval Augmented Generation system that answers questions over documents with proper citations, hybrid search, and enterprise-grade security controls.

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
    [Guardrail] Prompt injection detection (regex → LLM-as-judge)
                ↓
    [PII Redaction] Presidio strips names/emails/phones/SSNs
                ↓
    [Canary Router] Green/Blue model split (configurable %)
                ↓
    LLM Generation (GPT-5.4-nano) with prompt caching + fallback router
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
- **PII redaction:** Microsoft Presidio (analyzer + anonymizer) + regex fallback
- **Observability:** Langfuse tracing on all queries, guardrail decisions, and canary splits
- **API:** FastAPI with streaming SSE
- **UI:** Streamlit chat with citations panel
- **CI:** GitHub Actions — daily eval at 8am UTC with threshold checks

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg

export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export COHERE_API_KEY=...
export TAVILY_API_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_HOST=https://cloud.langfuse.com

docker compose up -d
python download_corpus.py
python ingest.py
python contextual_ingest.py
```

## Usage

```bash
# API
uvicorn api:app --reload
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is prompt caching?", "use_contextual": true, "stream": false, "tenant_id": "default"}'

# UI
streamlit run app.py

# CLI
python generate.py
```

## Evaluation Results

### Contextual Retrieval (30 Q&A pairs)

| Metric | Standard | Contextual | Δ |
|---|---|---|---|
| Recall@5 | 0.525 | 0.570 | +8.6% |
| MRR | 0.820 | 0.868 | +5.9% |
| Faithfulness | 0.709 | 0.864 | +21.9% |
| Avg latency | 1.65s | 1.18s | -28.5% |

### A/B Prompt Test (50 cases)

| Metric | Prompt A | Prompt B | Winner |
|---|---|---|---|
| Semantic similarity | 0.265 | 0.345 | B +30% |
| Output shape | 0.903 | 0.733 | A |
| Adversarial robustness | 0.176 | 0.362 | B +106% |
| Out-of-scope rejection | 0.296 | 0.503 | B +70% |

Prompt B selected as production prompt.

## Security & Safety Layer

### Prompt Injection Guardrail (`guardrails/guardrail.py`)
Two-stage detection before any query reaches the LLM:
- **Stage 1 — Regex (<1ms):** 8 attack families: instruction override, persona injection, prompt leak, jailbreak keywords, base64 obfuscation, delimiter injection, token smuggling, multilingual bypass
- **Stage 2 — LLM-as-judge (borderline only):** GPT-4.1-nano scores ambiguous queries; blocked if score ≥ 0.7
- All decisions logged to Langfuse with `injection_score`

### PII Redaction (`guardrails/pii_redact.py`)
- **Engine:** Microsoft Presidio (NER-based) with regex fallback
- **Entities:** PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN, IP_ADDRESS, IBAN_CODE, URL
- **Replacements:** deterministic placeholders (`[NAME]`, `[EMAIL]`, etc.)
- `sanitize_for_trace(obj)` cleans all metadata before Langfuse logging

### Prompt Caching + Model Fallback (`eval/caching_test.py`)
- System prompt >1024 tokens for OpenAI server-side caching (~50% input token discount on warm calls)
- Fallback router: `gpt-5.4-nano → gpt-4.1-nano → gpt-4.1-mini` on exception or latency breach

## Green/Blue Canary Deployment (`eval/canary.py`)

```bash
python eval/canary.py                  # 90% green / 10% blue
CANARY_PCT=30 python eval/canary.py    # 70% green / 30% blue
```

Routing is deterministic per query index (`random.seed(query_idx * 7919)`).
Promotion verdict: blue safe if `score ≥ green − 0.05 AND latency ≤ green × 1.2`

| Metric | GREEN (gpt-5.4-nano) | BLUE (gpt-4.1-nano) | Delta |
|---|---|---|---|
| Mean score | 0.371 | 0.800 | +0.429 |
| Mean latency | 1.022s | 0.645s | −37% |
| Cost/1k queries | $2.604m | $0.045m | −98% |

Verdict: ✅ BLUE promoted to production.

## Daily Eval CI (`.github/workflows/daily_eval.yml`)

Runs every day at 8am UTC or via `workflow_dispatch`:
1. Installs deps + spaCy model
2. Starts pgvector + Qdrant
3. Runs 50-case eval → `eval/full_eval_results.json`
4. Runs canary test → `eval/canary_results.json`
5. Checks thresholds: `sem_sim > 0.20`, `citation_valid > 0.50`, `output_shape > 0.70`, `success_rate > 0.80`
6. Uploads results artifact (30-day retention)
7. Posts metric table to GitHub Actions job summary

Required secrets: `OPENAI_API_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`, `COHERE_API_KEY`

## Corpus
- 12 HTML pages: Anthropic + OpenAI documentation
- 515 total chunks (400 tokens, 80 overlap)
- Stored in both pgvector and Qdrant

## Project Structure

```
ingest.py                     Standard ingestion
contextual_ingest.py          Contextual retrieval ingestion
retrieve.py                   BM25 + vector + RRF + rerank
generate.py                   LLM generation with guardrails + PII redaction
eval_rag.py                   Standard eval harness
eval_contextual.py            Contextual eval + before/after comparison
api.py                        FastAPI POST /query (streaming + tenant scoping)
app.py                        Streamlit chat UI
web_search.py                 Tavily web search fallback
doc_tracker.py                Document update detection
tenant.py                     Multi-tenant scoping utilities
download_corpus.py            Corpus downloader
config.py                     Config and constants
docker-compose.yml            pgvector + Qdrant
requirements.txt              Python dependencies

guardrails/
  guardrail.py                Two-stage prompt injection detection
  pii_redact.py               Presidio PII redaction + regex fallback

eval/
  run_eval.py                 50-case eval runner
  canary.py                   Green/Blue canary router
  caching_test.py             Prompt caching + model fallback router
  ab_test.py                  A/B prompt comparison
  metrics.py                  Eval metric definitions

.github/workflows/
  daily_eval.yml              Daily CI — eval + canary + threshold checks
```
