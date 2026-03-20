"""
OpenAI-compatible client for testing inference engines.

Usage:
    python -m inference.tasks.llm.client

Assumes a server is already running at the configured host:port.
Auto-detects which model the server is serving.
"""

from __future__ import annotations

from openai import OpenAI


BASE_URL = "http://0.0.0.0:8000/v1"
API_KEY = "dummy"


def main():
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    models = client.models.list()
    model_name = models.data[0].id

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_tokens=256,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    print(f"\n{'='*60}")
    print(f"  Model:  {response.model}")
    print(f"  Tokens: {response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion")
    print(f"{'='*60}")
    print(f"\n{response.choices[0].message.content}\n")


if __name__ == "__main__":
    main()
