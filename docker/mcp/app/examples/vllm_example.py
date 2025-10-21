"""Example MCP tool usage that calls the vLLM OpenAI-compatible endpoint."""

import os
import requests

OPENAI_BASE = os.getenv("OPENAI_API_BASE", "http://host.docker.internal:8005/v1")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")


def call_vllm(prompt: str, model: str = "qwen3-30b-awq") -> str:
    url = f"{OPENAI_BASE}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if OPENAI_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_KEY}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(call_vllm("테스트 문장을 하나 작성해 줘."))
