import json
from eval_rag import QA_PAIRS, recall_at_k, mrr, llm_judge
from retrieve import contextual_hybrid_search, embed_query, contextual_vector_search
from generate import generate
from config import LLM_MODEL
import time
from openai import OpenAI

openai = OpenAI()


def contextual_recall_at_k(query: str, expected: str, k: int = 5) -> float:
    try:
        emb = embed_query(query)
        results = contextual_vector_search(emb, top_k=k)
        combined = " ".join(r["text"].lower() for r in results)
        key_terms = [w.lower() for w in expected.split() if len(w) > 4]
        if not key_terms:
            return 0.0
        found = sum(1 for term in key_terms if term in combined)
        return found / len(key_terms)
    except Exception:
        return 0.0


def contextual_mrr(query: str, expected: str, k: int = 10) -> float:
    try:
        emb = embed_query(query)
        results = contextual_vector_search(emb, top_k=k)
        key_terms = [w.lower() for w in expected.split() if len(w) > 4]
        if not key_terms:
            return 0.0
        for rank, r in enumerate(results, 1):
            if any(term in r["text"].lower() for term in key_terms):
                return 1.0 / rank
        return 0.0
    except Exception:
        return 0.0


def run_contextual_eval():
    print("Running contextual retrieval evaluation...\n")
    results = []

    for i, pair in enumerate(QA_PAIRS):
        q, expected = pair["q"], pair["a"]
        print(f"[{i+1}/30] {q[:60]}...")

        start = time.perf_counter()
        try:
            chunks = contextual_hybrid_search(q)
            context = "\n\n".join(c["text"] for c in chunks)
            r = openai.chat.completions.create(
                model=LLM_MODEL,
                max_completion_tokens=500,
                messages=[
                    {"role": "system", "content": "Answer using only the provided context. Be concise."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"}
                ],
            )
            answer = r.choices[0].message.content
            latency = time.perf_counter() - start
        except Exception as e:
            answer = ""
            latency = 0.0

        rec = contextual_recall_at_k(q, expected)
        mrr_score = contextual_mrr(q, expected)
        faithfulness = llm_judge(q, answer, expected)

        results.append({
            "question": q,
            "recall@5": rec,
            "mrr": mrr_score,
            "faithfulness": faithfulness,
            "latency": latency,
        })

    avg_recall = sum(r["recall@5"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_lat = sum(r["latency"] for r in results) / len(results)

    print(f"\n{'='*50}")
    print("CONTEXTUAL RETRIEVAL RESULTS")
    print(f"Recall@5:     {avg_recall:.3f}")
    print(f"MRR:          {avg_mrr:.3f}")
    print(f"Faithfulness: {avg_faith:.3f}")
    print(f"Avg latency:  {avg_lat:.2f}s")

    print(f"\n{'='*50}")
    print("BEFORE (standard) vs AFTER (contextual)")
    print(f"Recall@5:     0.525 → {avg_recall:.3f}")
    print(f"MRR:          0.820 → {avg_mrr:.3f}")
    print(f"Faithfulness: 0.709 → {avg_faith:.3f}")

    with open("eval_contextual_results.json", "w") as f:
        json.dump({"summary": {
            "recall@5": avg_recall, "mrr": avg_mrr,
            "faithfulness": avg_faith, "avg_latency": avg_lat
        }, "results": results}, f, indent=2)

    print("\nSaved to eval_contextual_results.json")


if __name__ == "__main__":
    run_contextual_eval()