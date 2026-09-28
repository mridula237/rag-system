import os

# models
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
LLM_MODEL = "gpt-5.4-nano"
RERANK_MODEL = "rerank-english-v3.0"

# chunking
CHUNK_SIZE = 400        # tokens
CHUNK_OVERLAP = 80

# retrieval
BM25_TOP_K = 50
VECTOR_TOP_K = 50
RERANK_TOP_N = 6

# postgres
PG_DSN = os.getenv("PG_DSN", "postgresql://rag:rag@localhost:5433/rag")
# qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "rag_chunks"

# api keys (read from env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")