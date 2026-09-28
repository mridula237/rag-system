import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

from retrieve import contextual_hybrid_search, hybrid_search
from config import LLM_MODEL

app = FastAPI(title="RAG System")
openai = OpenAI()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context chunks.
- Answer ONLY from the context. Never use outside knowledge.
- If the context doesn't contain enough information, say so.
- Always cite which chunks you used by including doc_id and chunk_index.
- Be concise and accurate."""


class QueryRequest(BaseModel):
    query: str
    use_contextual: bool = True
    stream: bool = False


def build_context(chunks: list[dict]) -> str:
    context = ""
    for i, chunk in enumerate(chunks):
        doc_id = chunk["metadata"].get("doc_id", "unknown")
        chunk_index = chunk["metadata"].get("chunk_index", i)
        context += f"\n[{i+1}] doc_id={doc_id} chunk_index={chunk_index}\n{chunk['text']}\n"
    return context


@app.post("/query")
async def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    chunks = contextual_hybrid_search(req.query) if req.use_contextual else hybrid_search(req.query)
    context = build_context(chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"}
    ]

    if req.stream:
        async def stream_response():
            citations = [
                {"doc_id": c["metadata"].get("doc_id"), "chunk_index": c["metadata"].get("chunk_index")}
                for c in chunks
            ]
            yield f"data: {json.dumps({'citations': citations})}\n\n"

            stream = openai.chat.completions.create(
                model=LLM_MODEL,
                max_completion_tokens=1000,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'token': content})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    # non-streaming
    r = openai.chat.completions.create(
        model=LLM_MODEL,
        max_completion_tokens=1000,
        messages=messages,
        response_format={"type": "json_object"},
    )
    result = json.loads(r.choices[0].message.content)
    result["chunks_used"] = len(chunks)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}