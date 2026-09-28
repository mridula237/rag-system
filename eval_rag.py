import json
import time
from openai import OpenAI
from generate import generate
from retrieve import hybrid_search, vector_only_search, embed_query, vector_search
from config import LLM_MODEL

openai = OpenAI()

# 30 Q&A pairs grounded in the corpus
QA_PAIRS = [
    {"q": "What is prompt caching?", "a": "Prompt caching allows reusing a cached prefix of the prompt to reduce costs and latency."},
    {"q": "How long does a prompt cache last by default?", "a": "5 minutes"},
    {"q": "What is the minimum number of tokens required for prompt caching?", "a": "1024 tokens for Claude models"},
    {"q": "What embedding model does Anthropic recommend?", "a": "Voyage AI embeddings"},
    {"q": "What is text-embedding-3-large?", "a": "OpenAI's large embedding model with 3072 dimensions"},
    {"q": "What does RRF stand for in retrieval?", "a": "Reciprocal Rank Fusion"},
    {"q": "What is tool use in Claude?", "a": "Tool use allows Claude to call functions defined by the developer"},
    {"q": "What is structured output in OpenAI?", "a": "Structured outputs guarantee the model response matches a JSON schema"},
    {"q": "What is RAG?", "a": "Retrieval Augmented Generation combines document retrieval with LLM generation"},
    {"q": "What is the Claude API messages endpoint?", "a": "The messages endpoint is the primary way to interact with Claude models"},
    {"q": "What is function calling?", "a": "Function calling lets the model request execution of developer-defined functions"},
    {"q": "What is a system prompt?", "a": "A system prompt sets the behavior and context for the model before the conversation"},
    {"q": "What are stop sequences?", "a": "Stop sequences are strings that cause the model to stop generating when encountered"},
    {"q": "What is max_tokens?", "a": "max_tokens controls the maximum number of tokens the model can generate in a response"},
    {"q": "What is temperature in LLMs?", "a": "Temperature controls randomness in generation; 0 is deterministic, higher values increase randomness"},
    {"q": "What is top_p sampling?", "a": "Top-p sampling restricts generation to tokens whose cumulative probability reaches p"},
    {"q": "What is streaming in the API?", "a": "Streaming returns tokens incrementally as they are generated rather than waiting for the full response"},
    {"q": "What is a context window?", "a": "The context window is the maximum number of tokens the model can process in one request"},
    {"q": "What are embeddings used for?", "a": "Embeddings represent text as vectors for semantic search and similarity comparison"},
    {"q": "What is cosine similarity?", "a": "Cosine similarity measures the angle between two vectors to determine their semantic similarity"},
    {"q": "What is chunking in RAG?", "a": "Chunking splits documents into smaller pieces for embedding and retrieval"},
    {"q": "What is a vector database?", "a": "A vector database stores embeddings and enables fast similarity search"},
    {"q": "What is BM25?", "a": "BM25 is a keyword-based ranking algorithm used for text retrieval"},
    {"q": "What is reranking?", "a": "Reranking scores retrieved candidates with a cross-encoder model to improve relevance"},
    {"q": "What is the Anthropic API key used for?", "a": "The API key authenticates requests to the Anthropic API"},
    {"q": "What is Claude's training approach?", "a": "Claude is trained using Constitutional AI and RLHF"},
    {"q": "What is a cache breakpoint?", "a": "A cache breakpoint marks where the cached prefix ends in the prompt"},
    {"q": "What are tool results?", "a": "Tool results are the outputs returned after executing a tool call"},
    {"q": "What is pgvector?", "a": "pgvector is a PostgreSQL extension for storing and querying vector embeddings"},
    {"q": "What is Qdrant?", "a": "Qdrant is a vector database optimized for high-performance similarity search"},
]


def recall_at_k(query: str, expected_answer: str, k: int = 5) -> float:
    """Check if any of the top-k chunks contain relevant content."""
    try:
        emb = embed_query(query)
        results = vector_search(emb, top_k=k)
        combined = " ".join(r["text"].lower() for r in results)
        key_terms = [w.lower() for w in expected_answer.split() if len(w) > 4]
        if not key_terms:
            return 0.0
        found = sum(1 for term in key_terms if term in combined)
        return found / len(key_terms)
    except Exception:
        return 0.0


def mrr(query: str, expected_answer: str, k: int = 10) -> float:
    """Mean Reciprocal Rank — position of first relevant chunk."""
    try:
        emb = embed_query(query)
        results = vector_search(emb, top_k=k)
        key_terms = [w.lower() for w in expected_answer.split() if len(w) > 4]
        if not key_terms:
            return 0.0
        for rank, r in enumerate(results, 1):
            text_lower = r["text"].lower()
            if any(term in text_lower for term in key_terms):
                return 1.0 / rank
        return 0.0
    except Exception:
        return 0.0


def llm_judge(query: str, answer: str, expected: str) -> float:
    """Use GPT as judge: is the answer faithful to the expected answer?"""
    try:
        r = openai.chat.completions.create(
            model=LLM_MODEL,
            max_completion_tokens=10,
            messages=[
                {"role": "system", "content": "You are an evaluator. Reply with only a number 0.0 to 1.0 indicating how well the answer matches the expected answer. 1.0 = perfect match, 0.0 = completely wrong."},
                {"role": "user", "content": f"Question: {query}\nExpected: {expected}\nActual: {answer}\nScore:"}
            ],
        )
        return float(r.choices[0].message.content.strip())
    except Exception:
        return 0.0


def run_eval():
    print("Running RAG evaluation on 30 Q&A pairs...\n")
    results = []

    for i, pair in enumerate(QA_PAIRS):
        q, expected = pair["q"], pair["a"]
        print(f"[{i+1}/30] {q[:60]}...")

        start = time.perf_counter()
        try:
            result = generate(q)
            answer = result.get("answer", "")
            latency = time.perf_counter() - start
        except Exception as e:
            answer = ""
            latency = 0.0

        rec = recall_at_k(q, expected)
        mrr_score = mrr(q, expected)
        faithfulness = llm_judge(q, answer, expected)

        results.append({
            "question": q,
            "expected": expected,
            "answer": answer,
            "recall@5": rec,
            "mrr": mrr_score,
            "faithfulness": faithfulness,
            "latency": latency,
        })

    # summary
    avg_recall = sum(r["recall@5"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_lat = sum(r["latency"] for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"Recall@5:     {avg_recall:.3f}")
    print(f"MRR:          {avg_mrr:.3f}")
    print(f"Faithfulness: {avg_faith:.3f}")
    print(f"Avg latency:  {avg_lat:.2f}s")

    with open("eval_results.json", "w") as f:
        json.dump({"summary": {
            "recall@5": avg_recall, "mrr": avg_mrr,
            "faithfulness": avg_faith, "avg_latency": avg_lat
        }, "results": results}, f, indent=2)

    print("\nSaved to eval_results.json")
    return avg_recall, avg_mrr, avg_faith


if __name__ == "__main__":
    run_eval()