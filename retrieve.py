import json
import psycopg2
import cohere
from openai import OpenAI
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from config import *

openai = OpenAI()
co = cohere.Client(COHERE_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL)


# ── Load all chunks for BM25 ──────────────────────────────────────────────────

def load_all_chunks() -> list[dict]:
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()
    cur.execute("SELECT id, doc_id, chunk_index, text, metadata FROM chunks;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {"id": r[0], "doc_id": r[1], "chunk_index": r[2],
         "text": r[3], "metadata": r[4]}
        for r in rows
    ]


# ── Embed query ───────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    r = openai.embeddings.create(model=EMBED_MODEL, input=[query])
    return r.data[0].embedding


# ── Vector search (Qdrant) ────────────────────────────────────────────────────

def vector_search(query_embedding: list[float], top_k: int = VECTOR_TOP_K) -> list[dict]:
    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "id": r.payload.get("chunk_id", str(r.id)),
            "text": r.payload["text"],
            "score": r.score,
            "metadata": {k: v for k, v in r.payload.items() if k != "text"},
        }
        for r in results
    ]


# ── BM25 search ───────────────────────────────────────────────────────────────

def bm25_search(query: str, all_chunks: list[dict], top_k: int = BM25_TOP_K) -> list[dict]:
    tokenized = [c["text"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())

    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    return [
        {**all_chunks[i], "score": float(s)}
        for i, s in ranked if s > 0
    ]


# ── RRF fusion ────────────────────────────────────────────────────────────────

def rrf_fusion(vector_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    scores = {}
    chunks_by_id = {}

    for rank, chunk in enumerate(vector_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunks_by_id[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunks_by_id[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**chunks_by_id[cid], "rrf_score": score}
        for cid, score in ranked
    ]


# ── Cohere rerank ─────────────────────────────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    if not candidates:
        return []

    docs = [c["text"] for c in candidates]
    results = co.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=min(top_n, len(docs)),
    )

    return [
        {**candidates[r.index], "rerank_score": r.relevance_score}
        for r in results.results
    ]


# ── Hybrid search (full pipeline) ────────────────────────────────────────────

def hybrid_search(query: str, use_rerank: bool = True) -> list[dict]:
    all_chunks = load_all_chunks()

    query_embedding = embed_query(query)
    vec_results = vector_search(query_embedding)
    bm25_results = bm25_search(query, all_chunks)

    fused = rrf_fusion(vec_results, bm25_results)
    top_50 = fused[:50]

    if use_rerank:
        return rerank(query, top_50)
    return top_50[:RERANK_TOP_N]


# ── Vector only (for comparison) ─────────────────────────────────────────────

def vector_only_search(query: str) -> list[dict]:
    query_embedding = embed_query(query)
    return vector_search(query_embedding, top_k=RERANK_TOP_N)
def contextual_vector_search(query_embedding: list[float], top_k: int = VECTOR_TOP_K) -> list[dict]:
    """Vector search on the contextual collection."""
    results = qdrant.query_points(
        collection_name="rag_chunks_contextual",
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "id": r.payload.get("chunk_id", str(r.id)),
            "text": r.payload.get("contextual_text", r.payload["text"]),
            "score": r.score,
            "metadata": {k: v for k, v in r.payload.items() if k not in ("text", "contextual_text")},
        }
        for r in results
    ]


def contextual_hybrid_search(query: str, use_rerank: bool = True) -> list[dict]:
    """Full hybrid search using contextual embeddings."""
    all_chunks = load_all_chunks()  # BM25 still uses original chunks

    query_embedding = embed_query(query)
    vec_results = contextual_vector_search(query_embedding)
    bm25_results = bm25_search(query, all_chunks)

    fused = rrf_fusion(vec_results, bm25_results)
    top_50 = fused[:50]

    if use_rerank:
        return rerank(query, top_50)
    return top_50[:RERANK_TOP_N]

if __name__ == "__main__":
    query = "What is prompt caching and how does it reduce costs?"
    print(f"Query: {query}\n")

    print("=== Hybrid search (BM25 + Vector + RRF + Rerank) ===")
    results = hybrid_search(query)
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r['metadata'].get('title', 'unknown')} | score: {r.get('rerank_score', 0):.4f}")
        print(r["text"][:200] + "...")

    print("\n\n=== Vector only ===")
    vec_results = vector_only_search(query)
    for i, r in enumerate(vec_results):
        print(f"\n[{i+1}] {r['metadata'].get('title', 'unknown')} | score: {r['score']:.4f}")
        print(r["text"][:200] + "...")