import json
from openai import OpenAI
from retrieve import hybrid_search, vector_only_search
from config import LLM_MODEL

openai = OpenAI()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context chunks.

Rules:
- Answer ONLY from the context. Never use outside knowledge.
- If the context does not contain enough information, say "I don't have enough information in the provided context to answer this."
- Always cite which chunks you used by including their doc_id and chunk_index.
- Be concise and accurate.

Output format (strict JSON):
{
  "answer": "your answer here",
  "citations": [
    {"doc_id": "...", "chunk_index": 0}
  ]
}"""


def generate(query: str, use_hybrid: bool = True) -> dict:
    if use_hybrid:
        chunks = hybrid_search(query)
    else:
        chunks = vector_only_search(query)

    if not chunks:
        return {"answer": "No relevant context found.", "citations": []}

    # build context block
    context = ""
    for i, chunk in enumerate(chunks):
        doc_id = chunk["metadata"].get("doc_id", "unknown")
        chunk_index = chunk["metadata"].get("chunk_index", i)
        context += f"\n[{i+1}] doc_id={doc_id} chunk_index={chunk_index}\n{chunk['text']}\n"

    r = openai.chat.completions.create(
        model=LLM_MODEL,
        max_completion_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        response_format={"type": "json_object"},
    )

    raw = r.choices[0].message.content
    result = json.loads(raw)

    # validate citations exist in chunks
    valid_citations = []
    chunk_ids = {(c["metadata"].get("doc_id"), c["metadata"].get("chunk_index")) for c in chunks}
    for cit in result.get("citations", []):
        if (cit.get("doc_id"), cit.get("chunk_index")) in chunk_ids:
            valid_citations.append(cit)

    result["citations"] = valid_citations
    result["chunks_used"] = len(chunks)
    result["model"] = LLM_MODEL
    return result


if __name__ == "__main__":
    queries = [
        "What is prompt caching and how does it reduce costs?",
        "What are the differences between Claude and GPT embeddings?",
        "How does tool use work in the Anthropic API?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        result = generate(query)
        print(f"\nA: {result['answer']}")
        print(f"\nCitations: {json.dumps(result['citations'], indent=2)}")
        print(f"Chunks used: {result['chunks_used']}")