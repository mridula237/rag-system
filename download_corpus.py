import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

os.makedirs("docs", exist_ok=True)

SOURCES = [
    # Anthropic docs
    ("https://docs.anthropic.com/en/docs/intro-to-claude", "anthropic_intro.html"),
    ("https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview", "anthropic_prompting.html"),
    ("https://docs.anthropic.com/en/docs/build-with-claude/embeddings", "anthropic_embeddings.html"),
    ("https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview", "anthropic_tool_use.html"),
    ("https://docs.anthropic.com/en/docs/build-with-claude/RAG", "anthropic_rag.html"),
    ("https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching", "anthropic_caching.html"),
    ("https://docs.anthropic.com/en/docs/about-claude/models/overview", "anthropic_models.html"),
    # OpenAI docs
    ("https://platform.openai.com/docs/introduction", "openai_intro.html"),
    ("https://platform.openai.com/docs/guides/embeddings", "openai_embeddings.html"),
    ("https://platform.openai.com/docs/guides/function-calling", "openai_function_calling.html"),
    ("https://platform.openai.com/docs/guides/structured-outputs", "openai_structured_outputs.html"),
    ("https://platform.openai.com/docs/guides/retrieval-augmented-generation", "openai_rag.html"),
]

headers = {"User-Agent": "Mozilla/5.0 (research bot)"}

for url, fname in SOURCES:
    out = f"docs/{fname}"
    if os.path.exists(out):
        print(f"skip {fname}")
        continue
    print(f"downloading {fname}...")
    try:
        r = requests.get(url, headers=headers, timeout=15)
        with open(out, "w", encoding="utf-8") as f:
            f.write(r.text)
        print(f"  saved {len(r.text):,} chars")
        time.sleep(1)
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nDone. {len(os.listdir('docs'))} files in docs/")