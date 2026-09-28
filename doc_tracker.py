import hashlib
import json
import os
import psycopg2
from pathlib import Path
from config import PG_DSN


def setup_tracker():
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_hashes (
            doc_id TEXT PRIMARY KEY,
            file_hash TEXT,
            last_updated TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()


def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_stored_hash(doc_id: str) -> str | None:
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()
    cur.execute("SELECT file_hash FROM doc_hashes WHERE doc_id = %s", (doc_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else None


def store_hash(doc_id: str, hash_val: str):
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO doc_hashes (doc_id, file_hash, last_updated)
        VALUES (%s, %s, NOW())
        ON CONFLICT (doc_id) DO UPDATE
        SET file_hash = EXCLUDED.file_hash, last_updated = NOW()
    """, (doc_id, hash_val))
    conn.commit()
    cur.close()
    conn.close()


def delete_chunks(doc_id: str):
    """Delete all chunks for a doc from postgres and qdrant."""
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
    cur.execute("DELETE FROM chunks_contextual WHERE doc_id = %s", (doc_id,))
    conn.commit()
    cur.close()
    conn.close()

    # delete from qdrant
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from config import QDRANT_COLLECTION

    qdrant = QdrantClient(url="http://localhost:6333")
    for collection in [QDRANT_COLLECTION, "rag_chunks_contextual"]:
        qdrant.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        )
    print(f"  deleted old chunks for {doc_id}")


def check_and_update(docs_dir: str = "docs") -> list[str]:
    """Check all docs for changes. Return list of updated doc_ids."""
    setup_tracker()
    updated = []

    for fpath in sorted(Path(docs_dir).iterdir()):
        if not fpath.suffix in (".html", ".pdf", ".txt"):
            continue

        doc_id = fpath.stem
        current_hash = file_hash(str(fpath))
        stored = get_stored_hash(doc_id)

        if stored is None:
            print(f"[NEW] {doc_id}")
            updated.append(doc_id)
            store_hash(doc_id, current_hash)
        elif stored != current_hash:
            print(f"[CHANGED] {doc_id} — re-embedding...")
            delete_chunks(doc_id)
            updated.append(doc_id)
            store_hash(doc_id, current_hash)
        else:
            print(f"[OK] {doc_id}")

    return updated


if __name__ == "__main__":
    updated = check_and_update()
    if updated:
        print(f"\n{len(updated)} docs changed: {updated}")
        print("Run ingest.py to re-embed changed docs.")
    else:
        print("\nAll docs up to date.")