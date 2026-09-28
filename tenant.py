import json
import hashlib
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
from config import PG_DSN, QDRANT_URL, QDRANT_COLLECTION, DEFAULT_TENANT

qdrant = QdrantClient(url=QDRANT_URL)


def add_tenant_to_chunks(chunks: list[dict], tenant_id: str) -> list[dict]:
    """Add tenant_id to each chunk's metadata."""
    for chunk in chunks:
        chunk["metadata"]["tenant_id"] = tenant_id
        chunk["id"] = hashlib.md5(
            f"{tenant_id}_{chunk['doc_id']}_{chunk['chunk_index']}".encode()
        ).hexdigest()
    return chunks


def tenant_vector_search(query_embedding: list[float], tenant_id: str,
                          collection: str = QDRANT_COLLECTION, top_k: int = 50) -> list[dict]:
    """Vector search scoped to a specific tenant."""
    results = qdrant.query_points(
        collection_name=collection,
        query=query_embedding,
        query_filter=Filter(
            must=[FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id)
            )]
        ),
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "id": r.payload.get("chunk_id", str(r.id)),
            "text": r.payload.get("contextual_text", r.payload.get("text", "")),
            "score": r.score,
            "metadata": {k: v for k, v in r.payload.items() if k not in ("text", "contextual_text")},
        }
        for r in results
    ]


def list_tenant_docs(tenant_id: str) -> list[str]:
    """List all doc_ids belonging to a tenant."""
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT doc_id FROM chunks
        WHERE metadata->>'tenant_id' = %s
    """, (tenant_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [r[0] for r in rows]


def delete_tenant_docs(tenant_id: str):
    """Delete all chunks for a tenant."""
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE metadata->>'tenant_id' = %s", (tenant_id,))
    conn.commit()
    cur.close()
    conn.close()

    qdrant.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        )
    )
    print(f"Deleted all docs for tenant: {tenant_id}")


if __name__ == "__main__":
  
    for tenant in ["default", "tenant_a", "tenant_b"]:
        docs = list_tenant_docs(tenant)
        print(f"Tenant '{tenant}': {len(docs)} docs → {docs}")