import os
import json
import hashlib
import re
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import psycopg2
from psycopg2.extras import execute_values

from config import *

openai = OpenAI()
qdrant = QdrantClient(url=QDRANT_URL)
enc = tiktoken.get_encoding("cl100k_base")


# ── DB setup ──────────────────────────────────────────────────────────────────

def setup_postgres():
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            doc_id TEXT,
            chunk_index INTEGER,
            text TEXT,
            embedding vector(1536),
            metadata JSONB
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);")
    conn.commit()
    cur.close()
    return conn


def setup_qdrant():
    existing = [c.name for c in qdrant.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"created qdrant collection: {QDRANT_COLLECTION}")
    else:
        print(f"qdrant collection exists: {QDRANT_COLLECTION}")


# ── Chunking ──────────────────────────────────────────────────────────────────

def extract_text(html_path: str) -> tuple[str, str]:
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    title = soup.title.string if soup.title else Path(html_path).stem

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title.strip(), text


def chunk_text(text: str, doc_id: str, title: str) -> list[dict]:
    tokens = enc.encode(text)
    chunks = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)

        chunk_id = hashlib.md5(f"{doc_id}_{idx}".encode()).hexdigest()
        chunks.append({
            "id": chunk_id,
            "doc_id": doc_id,
            "chunk_index": idx,
            "text": chunk_text_str,
            "metadata": {"title": title, "doc_id": doc_id, "chunk_index": idx}
        })

        start += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_batch(texts: list[str]) -> list[list[float]]:
    r = openai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in r.data]


# ── Storage ───────────────────────────────────────────────────────────────────

def store_postgres(conn, chunks: list[dict], embeddings: list[list[float]]):
    cur = conn.cursor()
    rows = [
        (c["id"], c["doc_id"], c["chunk_index"], c["text"],
         json.dumps(emb), json.dumps(c["metadata"]))
        for c, emb in zip(chunks, embeddings)
    ]
    execute_values(cur, """
        INSERT INTO chunks (id, doc_id, chunk_index, text, embedding, metadata)
        VALUES %s ON CONFLICT (id) DO NOTHING
    """, rows)
    conn.commit()
    cur.close()


def store_qdrant(chunks: list[dict], embeddings: list[list[float]]):
    points = [
        PointStruct(
            id=abs(hash(c["id"])) % (2**63),
            vector=emb,
            payload={**c["metadata"], "text": c["text"], "chunk_id": c["id"]}
        )
        for c, emb in zip(chunks, embeddings)
    ]
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest():
    print("Setting up databases...")
    conn = setup_postgres()
    setup_qdrant()

    docs_dir = Path("docs")
    html_files = list(docs_dir.glob("*.html"))
    print(f"Found {len(html_files)} documents\n")

    # check which docs have changed
    from doc_tracker import check_and_update
    changed_docs = check_and_update(str(docs_dir))

    total_chunks = 0

    for html_path in sorted(html_files):
        doc_id = html_path.stem

        if doc_id not in changed_docs:
            print(f"Skipping {doc_id} (unchanged)")
            continue

        print(f"Processing {doc_id}...")

        title, text = extract_text(str(html_path))
        if len(text) < 100:
            print(f"  skipping — too short ({len(text)} chars)")
            continue

        chunks = chunk_text(text, doc_id, title)
        print(f"  {len(chunks)} chunks from {len(text):,} chars")

        all_embeddings = []
        for i in range(0, len(chunks), 20):
            batch = chunks[i:i+20]
            texts = [c["text"] for c in batch]
            embs = embed_batch(texts)
            all_embeddings.extend(embs)

        store_postgres(conn, chunks, all_embeddings)
        store_qdrant(chunks, all_embeddings)
        total_chunks += len(chunks)
        print(f"  stored in pgvector + qdrant")

    conn.close()
    if total_chunks == 0:
        print("\nAll docs up to date — nothing re-embedded.")
    else:
        print(f"\nDone. {total_chunks} total chunks indexed.")


if __name__ == "__main__":
    ingest()