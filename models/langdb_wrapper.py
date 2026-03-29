from __future__ import annotations

import os

from openai import OpenAI


OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_MAP = {
    "gpt-5": "openai/gpt-4o",
    "claude-opus-4.5": "anthropic/claude-sonnet-4",
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview",
    "grok-4": "x-ai/grok-3-beta",
    "deepseek-r1-0528": "deepseek/deepseek-r1",
}


class LangDBGenerator:
    def __init__(self, model_name: str, api_key: str = OPENROUTER_API_KEY):
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )
        self.model = MODEL_MAP.get(model_name, model_name)
        self.display_name = model_name

    def sample(self, prompt: str, n_samples: int = 1, max_tokens: int = 150) -> list[str]:
        responses = []
        for _ in range(n_samples):
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            responses.append(res.choices[0].message.content or "")
        return responses

    def sample_with_system(
        self, system: str, prompt: str, n_samples: int = 1, max_tokens: int = 150
    ) -> list[str]:
        responses = []
        for _ in range(n_samples):
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            responses.append(res.choices[0].message.content or "")
        return responses
