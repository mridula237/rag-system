import os, sys, time, json, random, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
from langfuse import Langfuse
from config import LLM_MODEL

client = OpenAI()
langfuse = Langfuse()
GREEN_MODEL = LLM_MODEL
BLUE_MODEL  = "gpt-4.1-nano"
CANARY_PCT  = int(os.environ.get("CANARY_PCT", "10"))

SYSTEM_PROMPT = """You are an expert document retrieval assistant.
Answer using ONLY the retrieved context. Cite as [doc_id:chunk_index].
If context is insufficient say: "I don't have enough information in the retrieved context."
Never fabricate facts or citations."""

SAMPLE_CONTEXT = """
[api-docs:3] The API enforces rate limits of 100 requests per minute per API key.
Enterprise plans have a 1000 req/min limit. HTTP 429 is returned when exceeded.
[upload-guide:1] File uploads are limited to 50MB per file and 500MB per day.
Supported formats: PDF, DOCX, TXT, MD, CSV.
[security-policy:7] API keys can be rotated from Settings > API Keys.
Old keys are invalidated immediately. Keys expire after 365 days of inactivity.
"""

TEST_CASES = [
    {"query": "What is the API rate limit?",          "gold": "100 requests per minute per API key"},
    {"query": "What file formats are supported?",     "gold": "PDF, DOCX, TXT, MD, CSV"},
    {"query": "How do I rotate my API key?",          "gold": "Settings > API Keys page"},
    {"query": "What is the max upload size?",         "gold": "50MB per file, 500MB per day"},
    {"query": "What happens if I exceed rate limit?", "gold": "HTTP 429 is returned"},
    {"query": "How long until API keys expire?",      "gold": "365 days of inactivity"},
    {"query": "What is the enterprise rate limit?",   "gold": "1000 requests per minute"},
    {"query": "What is the capital of France?",       "gold": "I don't have enough information"},
    {"query": "Are uploads processed synchronously?", "gold": "asynchronously"},
    {"query": "What HTTP code for rate limit?",       "gold": "429"},
]

def route_request(idx):
    random.seed(idx * 7919)
    return "blue" if random.random() < (CANARY_PCT / 100) else "green"

def run_query(query, channel):
    model = BLUE_MODEL if channel == "blue" else GREEN_MODEL
    start = time.perf_counter()
    try:
        r = client.chat.completions.create(model=model, max_completion_tokens=200,
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":f"Context:\n{SAMPLE_CONTEXT}\n\nQuestion: {query}"}])
        return {"success":True,"channel":channel,"model":model,
                "answer":r.choices[0].message.content.strip(),
                "latency_s":round(time.perf_counter()-start,3),
                "input_tokens":r.usage.prompt_tokens,"output_tokens":r.usage.completion_tokens,"error":None}
    except Exception as e:
        return {"success":False,"channel":channel,"model":model,"answer":"",
                "latency_s":round(time.perf_counter()-start,3),"input_tokens":0,"output_tokens":0,"error":str(e)}

def score_answer(answer, gold):
    if not answer: return 0.0
    gw = set(gold.lower().split()); aw = set(answer.lower().split())
    return round(len(gw & aw) / max(len(gw), 1), 3)

def trace_to_langfuse(query, result, score, idx):
    try:
        t = langfuse.trace(name="canary_query", input={"question":query},
            output={"answer":result["answer"]},
            metadata={"model":result["model"],"channel":result["channel"],"latency_s":result["latency_s"]},
            tags=["canary", result["channel"], result["model"]])
        langfuse.score(trace_id=t.id, name="keyword_overlap", value=score)
        langfuse.score(trace_id=t.id, name="latency_s", value=result["latency_s"])
        price = {"gpt-5.4-nano":(1.0,5.0),"gpt-4.1-nano":(0.10,0.40)}
        p_in,p_out = price.get(result["model"],(1.0,5.0))
        langfuse.score(trace_id=t.id, name="cost_usd",
            value=round(result["input_tokens"]/1e6*p_in + result["output_tokens"]/1e6*p_out, 8))
        langfuse.flush()
    except Exception as e:
        print(f"  [langfuse] {e}")

def run_canary():
    print(f"\n{'='*60}\nCANARY: {100-CANARY_PCT}% green ({GREEN_MODEL}) / {CANARY_PCT}% blue ({BLUE_MODEL})\n{'='*60}")
    results = []
    for i, case in enumerate(TEST_CASES):
        channel = route_request(i)
        r = run_query(case["query"], channel)
        score = score_answer(r["answer"], case["gold"]) if r["success"] else 0.0
        trace_to_langfuse(case["query"], r, score, i)
        results.append({**r, "query":case["query"], "gold":case["gold"], "score":score})
        print(f"  {'✓' if r['success'] else '✗'} [{channel:5}] [{r['latency_s']:5.2f}s] score={score:.2f}  {case['query'][:45]}")

    for ch in ["green","blue"]:
        rows = [r for r in results if r["channel"]==ch and r["success"]]
        if not rows: continue
        price = {"gpt-5.4-nano":(1.0,5.0),"gpt-4.1-nano":(0.10,0.40)}
        costs = [r["input_tokens"]/1e6*price.get(r["model"],(1.0,5.0))[0] +
                 r["output_tokens"]/1e6*price.get(r["model"],(1.0,5.0))[1] for r in rows]
        print(f"\n  {ch.upper():5}  n={len(rows)}  score={statistics.mean(r['score'] for r in rows):.3f}"
              f"  lat={statistics.mean(r['latency_s'] for r in rows):.3f}s"
              f"  cost=${sum(costs)*1000:.4f}m")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canary_results.json")
    with open(out,"w") as f: json.dump({"green_model":GREEN_MODEL,"blue_model":BLUE_MODEL,
        "canary_pct":CANARY_PCT,"results":results}, f, indent=2)
    print(f"\nResults → {out}")
    langfuse.flush()

if __name__ == "__main__":
    run_canary()
