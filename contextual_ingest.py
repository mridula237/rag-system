import os
import json
import hashlib
import re
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from openai import OpenAI
from anthropic import Anthropic
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import psycopg2
from psycopg2.extras import execute_values

from config import *

openai_client = OpenAI()
anthropic_client = Anthropic()
qdrant = QdrantClient(url=QDRANT_URL)
enc = tiktoken.get_encoding("cl100k_base")

# new collection for contextual chunks
CONTEXTUAL_COLLECTION = "rag_chunks_contextual"
CONTEXTUAL_TABLE = "chunks_contextual"


def setup_postgres_contextual():
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {CONTEXTUAL_TABLE} (
            id TEXT PRIMARY KEY,
            doc_id TEXT,
            chunk_index INTEGER,
            text TEXT,
            context TEXT,
            contextual_text TEXT,
            embedding vector(1536),
            metadata JSONB
        );
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS {CONTEXTUAL_TABLE}_embedding_idx ON {CONTEXTUAL_TABLE} USING hnsw (embedding vector_cosine_ops);")
    conn.commit()
    cur.close()
    return conn


def setup_qdrant_contextual():
    existing = [c.name for c in qdrant.get_collections().collections]
    if CONTEXTUAL_COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=CONTEXTUAL_COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"created qdrant collection: {CONTEXTUAL_COLLECTION}")


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
        chunk_id = hashlib.md5(f"ctx_{doc_id}_{idx}".encode()).hexdigest()
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


def generate_context(doc_text: str, chunk_text: str) -> str:
    """Ask Claude to generate a short context for this chunk."""
    try:
        r = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system="Generate a brief 1-2 sentence context describing where this chunk fits in the document. Be concise.",
            messages=[{
                "role": "user",
                "content": f"Document (first 2000 chars):\n{doc_text[:2000]}\n\nChunk:\n{chunk_text}\n\nContext:"
            }]
        )
        return r.content[0].text.strip()
    except Exception as e:
        return ""


def embed_batch(texts: list[str]) -> list[list[float]]:
    r = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in r.data]


def store_postgres_contextual(conn, chunks: list[dict], embeddings: list[list[float]]):
    cur = conn.cursor()
    rows = [
        (c["id"], c["doc_id"], c["chunk_index"], c["text"],
         c.get("context", ""), c.get("contextual_text", c["text"]),
         json.dumps(emb), json.dumps(c["metadata"]))
        for c, emb in zip(chunks, embeddings)
    ]
    execute_values(cur, f"""
        INSERT INTO {CONTEXTUAL_TABLE}
        (id, doc_id, chunk_index, text, context, contextual_text, embedding, metadata)
        VALUES %s ON CONFLICT (id) DO NOTHING
    """, rows)
    conn.commit()
    cur.close()


def store_qdrant_contextual(chunks: list[dict], embeddings: list[list[float]]):
    points = [
        PointStruct(
            id=abs(hash(c["id"])) % (2**63),
            vector=emb,
            payload={
                **c["metadata"],
                "text": c["text"],
                "context": c.get("context", ""),
                "contextual_text": c.get("contextual_text", c["text"]),
                "chunk_id": c["id"]
            }
        )
        for c, emb in zip(chunks, embeddings)
    ]
    qdrant.upsert(collection_name=CONTEXTUAL_COLLECTION, points=points)


def contextual_ingest():
    print("Setting up contextual databases...")
    conn = setup_postgres_contextual()
    setup_qdrant_contextual()

    docs_dir = Path("docs")
    html_files = list(docs_dir.glob("*.html"))
    print(f"Found {len(html_files)} documents\n")

    total_chunks = 0

    for html_path in sorted(html_files):
        doc_id = html_path.stem
        print(f"Processing {doc_id}...")

        title, text = extract_text(str(html_path))
        if len(text) < 100:
            print(f"  skipping — too short")
            continue

        chunks = chunk_text(text, doc_id, title)
        print(f"  {len(chunks)} chunks — generating context...")

        # add context to each chunk
        for i, chunk in enumerate(chunks):
            context = generate_context(text, chunk["text"])
            chunk["context"] = context
            chunk["contextual_text"] = f"{context}\n\n{chunk['text']}" if context else chunk["text"]
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(chunks)} contexts generated")

        # embed contextual text
        all_embeddings = []
        for i in range(0, len(chunks), 20):
            batch = chunks[i:i+20]
            texts = [c["contextual_text"] for c in batch]
            embs = embed_batch(texts)
            all_embeddings.extend(embs)

        store_postgres_contextual(conn, chunks, all_embeddings)
        store_qdrant_contextual(chunks, all_embeddings)
        total_chunks += len(chunks)
        print(f"  stored {len(chunks)} contextual chunks")

    conn.close()
    print(f"\nDone. {total_chunks} contextual chunks indexed.")


if __name__ == "__main__":
    contextual_ingest()