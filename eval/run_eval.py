import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from generate import generate
from eval.metrics import (
    check_output_shape, check_citation_present,
    semantic_similarity, llm_judge, trace_query
)

PROMPT_VERSION = "v1"
MODEL = "gpt-5.4-nano"


def run_full_eval(cases_path: str = "eval/eval_set.json",
                  output_path: str = "eval/full_eval_results.json",
                  session_id: str = "eval_v1"):
    with open(cases_path) as f:
        cases = json.load(f)

    print(f"Running eval on {len(cases)} cases...\n")
    results = []

    for case in cases:
        qid = case["id"]
        question = case["question"]
        gold = case["gold"]
        category = case["category"]
        print(f"[{qid}/50] [{category}] {question[:60]}...")

        start = time.perf_counter()
        try:
            result = generate(question, use_hybrid=True)
            latency = time.perf_counter() - start

            answer = result.get("answer", "")
            citations = result.get("citations", [])
            chunks = result.get("chunks", [])  # we'll add this to generate()

            # compute all 4 metric types
            shape = check_output_shape(answer, citations)
            citation_check = check_citation_present(answer, citations, chunks)
            sem_sim = semantic_similarity(answer, gold)
            context_str = " ".join(c.get("text", "") for c in chunks[:3])
            judge = llm_judge(question, answer, gold, context_str)

            scores = {
                "output_shape": shape["score"],
                "citation_valid": citation_check["score"],
                "semantic_similarity": sem_sim["score"],
                "llm_judge_overall": judge.get("overall", 0) / 4,  # normalize to 0-1
                "llm_judge_helpfulness": judge.get("helpfulness", 0),
                "llm_judge_faithfulness": judge.get("faithfulness", 0),
                "llm_judge_harm": judge.get("harm", 4),
            }

            # trace to Langfuse
            trace_id = trace_query(
                question=question,
                answer=answer,
                citations=citations,
                chunks=chunks,
                gold=gold,
                category=category,
                model=MODEL,
                prompt_version=PROMPT_VERSION,
                input_tokens=result.get("in", 0),
                output_tokens=result.get("out", 0),
                latency_s=latency,
                scores=scores,
                session_id=session_id,
            )

            record = {
                "id": qid,
                "category": category,
                "question": question,
                "gold": gold,
                "answer": answer,
                "citations": citations,
                "scores": scores,
                "latency_s": round(latency, 2),
                "trace_id": trace_id,
                "error": None,
            }
            print(f"  sim={sem_sim['score']:.2f} judge={judge.get('overall',0)}/4 "
                  f"shape={shape['score']:.1f} lat={latency:.1f}s")

        except Exception as e:
            latency = time.perf_counter() - start
            print(f"  ❌ {e}")
            record = {"id": qid, "category": category, "question": question,
                      "gold": gold, "answer": "", "citations": [],
                      "scores": {}, "latency_s": round(latency, 2),
                      "trace_id": None, "error": str(e)}

        results.append(record)

    # summary
    ok = [r for r in results if not r["error"]]
    print(f"\n{'='*60}")
    print(f"EVAL SUMMARY ({len(ok)}/50 successful)")

    for metric in ["semantic_similarity", "output_shape", "citation_valid",
                   "llm_judge_overall"]:
        avg = sum(r["scores"].get(metric, 0) for r in ok) / max(len(ok), 1)
        print(f"  {metric}: {avg:.3f}")

    by_cat = {}
    for r in ok:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(r["scores"].get("semantic_similarity", 0))

    print("\nSemantic similarity by category:")
    for cat, scores in sorted(by_cat.items()):
        print(f"  {cat}: {sum(scores)/len(scores):.3f}")

    with open(output_path, "w") as f:
        json.dump({"prompt_version": PROMPT_VERSION, "model": MODEL,
                   "results": results}, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    run_full_eval()