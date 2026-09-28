import streamlit as st
import requests
import json

st.set_page_config(page_title="RAG System", layout="wide")
st.title("RAG System")
st.caption("Ask questions over Anthropic and OpenAI documentation.")

with st.sidebar:
    st.header("Settings")
    use_contextual = st.toggle("Use contextual retrieval", value=True)
    use_streaming = st.toggle("Stream response", value=True)
    st.divider()
    st.caption("Corpus: Anthropic + OpenAI docs")
    st.caption("Retrieval: BM25 + Vector + RRF + Cohere Rerank")
    if use_contextual:
        st.success("Contextual retrieval ON")
    else:
        st.warning("Standard retrieval")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Citations"):
                for c in msg["citations"]:
                    st.write(f"- {c['doc_id']} (chunk {c['chunk_index']})")

query = st.chat_input("Ask something about Claude or OpenAI APIs...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        citations = []

        if use_streaming:
            full_text = ""
            try:
                with requests.post(
                    "http://localhost:8000/query",
                    json={"query": query, "use_contextual": use_contextual, "stream": True},
                    stream=True,
                    timeout=30,
                ) as r:
                    for line in r.iter_lines():
                        if not line:
                            continue
                        line = line.decode("utf-8")
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if "citations" in chunk:
                            citations = chunk["citations"]
                        if "token" in chunk:
                            full_text += chunk["token"]
                            placeholder.markdown(full_text + "▌")
                placeholder.markdown(full_text)
            except Exception as e:
                placeholder.error(f"Error: {e}")
                full_text = ""
        else:
            try:
                r = requests.post(
                    "http://localhost:8000/query",
                    json={"query": query, "use_contextual": use_contextual, "stream": False},
                    timeout=30,
                )
                result = r.json()
                full_text = result.get("answer", "No answer returned.")
                citations = result.get("citations", [])
                placeholder.markdown(full_text)
            except Exception as e:
                placeholder.error(f"Error: {e}")
                full_text = ""

        if citations:
            with st.expander(f"Citations ({len(citations)})"):
                for c in citations:
                    st.write(f"- **{c['doc_id']}** chunk {c['chunk_index']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "citations": citations
    })