"""
llm_client.py
Thin wrapper around the OpenAI API used by every other module.
"""

import os
from openai import OpenAI


class LLMClient:
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    def stream(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.2):
        stream = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def embed(self, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
        """Batch-embed a list of texts using OpenAI embeddings."""
        resp = self.client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
