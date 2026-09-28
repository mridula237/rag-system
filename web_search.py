import os
from tavily import TavilyClient

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

MIN_RERANK_SCORE = 0.1  


def should_fallback(chunks: list[dict]) -> bool:
    """Return True if top chunk has low rerank score."""
    if not chunks:
        return True
    top_score = chunks[0].get("rerank_score", 1.0)
    return top_score < MIN_RERANK_SCORE


def web_search(query: str, max_results: int = 6) -> list[dict]:
    """Search the web and return chunks in same format as retrieval."""
    results = tavily.search(query=query, max_results=max_results)
    chunks = []
    for i, r in enumerate(results.get("results", [])):
        chunks.append({
            "id": f"web_{i}",
            "text": r.get("content", ""),
            "score": r.get("score", 0.0),
            "metadata": {
                "doc_id": "web_search",
                "chunk_index": i,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
            }
        })
    return chunks


if __name__ == "__main__":
    results = web_search("What is prompt caching in Claude?")
    for r in results:
        print(f"\n[{r['metadata']['title']}]")
        print(r["text"][:200])