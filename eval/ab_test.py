import json
import time
import sys
import os
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI
from retrieve import hybrid_search
from eval.metrics import check_output_shape, check_citation_present, semantic_similarity, llm_judge

openai_client = OpenAI()

PROMPT_A = """You are a helpful assistant that answers questions using ONLY the provided context chunks.

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

PROMPT_B = """You are a precise technical assistant for Anthropic and OpenAI documentation.

Your job:
1. Read the context chunks carefully.
2. Answer the question directly and specifically using ONLY information from the context.
3. If the question asks about something not in the context, say exactly: "This topic is not covered in the provided documentation."
4. For adversarial or harmful requests, respond: "I can only answer questions about AI/ML documentation."
5. Cite every chunk you drew from.

Context quality note: Some chunks may be navigation menus or headers — ignore those and focus on substantive content.

Output format (strict JSON):
{
  "answer": "your answer here",
  "citations": [
    {"doc_id": "...", "chunk_index": 0}
  ]
}"""


def run_with_prompt(query: str, system_prompt: str) -> dict:
    chunks = hybrid_search(query)
    if not chunks:
        return {"answer": "No relevant context found.", "citations": [], "chunks": []}

    context = ""
    for i, chunk in enumerate(chunks):
        doc_id = chunk["metadata"].get("doc_id", "unknown")
        chunk_index = chunk["metadata"].get("chunk_index", i)
        context += f"\n[{i+1}] doc_id={doc_id} chunk_index={chunk_index}\n{chunk['text']}\n"

    r = openai_client.chat.completions.create(
        model="gpt-5.4-nano",
        max_completion_tokens=1000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        response_format={"type": "json_object"},
    )
    raw = r.choices[0].message.content
    result = json.loads(raw)
    result["chunks"] = chunks
    return result


def run_ab_test(cases_path: str = "eval/eval_set.json",
                output_path: str = "eval/ab_test_results.json"):
    with open(cases_path) as f:
        cases = json.load(f)

    print(f"Running A/B test on {len(cases)} cases...\n")
    results_a, results_b = [], []

    for case in cases:
        qid = case["id"]
        question = case["question"]
        gold = case["gold"]
        category = case["category"]
        print(f"[{qid}/50] {question[:55]}...")

        for label, prompt, results_list in [("A", PROMPT_A, results_a), ("B", PROMPT_B, results_b)]:
            try:
                start = time.perf_counter()
                result = run_with_prompt(question, prompt)
                latency = time.perf_counter() - start

                answer = result.get("answer", "")
                citations = result.get("citations", [])
                chunks = result.get("chunks", [])

                shape = check_output_shape(answer, citations)
                sem_sim = semantic_similarity(answer, gold)
                context_str = " ".join(c.get("text", "") for c in chunks[:3])
                judge = llm_judge(question, answer, gold, context_str)

                scores = {
                    "output_shape": shape["score"],
                    "semantic_similarity": sem_sim["score"],
                    "llm_judge_overall": judge.get("overall", 0) / 4,
                    "llm_judge_helpfulness": judge.get("helpfulness", 0),
                    "llm_judge_faithfulness": judge.get("faithfulness", 0),
                }
                results_list.append({
                    "id": qid, "category": category, "question": question,
                    "gold": gold, "answer": answer, "scores": scores,
                    "latency_s": round(latency, 2), "error": None
                })
                print(f"  [{label}] sim={sem_sim['score']:.2f} judge={judge.get('overall',0)}/4 lat={latency:.1f}s")
            except Exception as e:
                print(f"  [{label}] ❌ {e}")
                results_list.append({
                    "id": qid, "category": category, "question": question,
                    "gold": gold, "answer": "", "scores": {}, "latency_s": 0, "error": str(e)
                })

    # compute summary stats
    def summarize(results):
        ok = [r for r in results if not r["error"]]
        cats = {}
        for r in ok:
            cats.setdefault(r["category"], []).append(r["scores"].get("semantic_similarity", 0))
        return {
            "n_ok": len(ok),
            "semantic_similarity": round(sum(r["scores"].get("semantic_similarity", 0) for r in ok) / max(len(ok), 1), 3),
            "output_shape": round(sum(r["scores"].get("output_shape", 0) for r in ok) / max(len(ok), 1), 3),
            "llm_judge_overall": round(sum(r["scores"].get("llm_judge_overall", 0) for r in ok) / max(len(ok), 1), 3),
            "by_category": {cat: round(sum(v)/len(v), 3) for cat, v in cats.items()},
        }

    summary_a = summarize(results_a)
    summary_b = summarize(results_b)

    print(f"\n{'='*60}")
    print(f"A/B RESULTS")
    print(f"{'Metric':<25} {'Prompt A':>10} {'Prompt B':>10} {'Winner':>8}")
    print(f"{'-'*55}")
    for metric in ["semantic_similarity", "output_shape", "llm_judge_overall"]:
        va, vb = summary_a[metric], summary_b[metric]
        winner = "B" if vb > va else "A" if va > vb else "tie"
        print(f"  {metric:<23} {va:>10.3f} {vb:>10.3f} {winner:>8}")

    print(f"\nBy category (semantic_similarity):")
    all_cats = sorted(set(list(summary_a["by_category"]) + list(summary_b["by_category"])))
    for cat in all_cats:
        va = summary_a["by_category"].get(cat, 0)
        vb = summary_b["by_category"].get(cat, 0)
        print(f"  {cat:<15} A={va:.3f}  B={vb:.3f}")

    output = {
        "prompt_a": {"name": "baseline", "system_prompt": PROMPT_A, "summary": summary_a, "results": results_a},
        "prompt_b": {"name": "improved", "system_prompt": PROMPT_B, "summary": summary_b, "results": results_b},
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")
    return summary_a, summary_b


if __name__ == "__main__":
    run_ab_test()