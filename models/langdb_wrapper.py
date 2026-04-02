from __future__ import annotations

import os
import time

from openai import OpenAI


OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MAX_RETRIES = 6
RETRY_BACKOFF = 10.0
CALL_INTERVAL = 5.0


class LangDBGenerator:
    def __init__(self, model_name: str, api_key: str | None = None):
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key or OPENROUTER_API_KEY,
        )
        self.model = model_name
        self.display_name = model_name
        self._last_call: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < CALL_INTERVAL:
            time.sleep(CALL_INTERVAL - elapsed)

    def _call_with_retry(self, messages: list[dict], max_tokens: int) -> str:
        backoff = RETRY_BACKOFF
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                res = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                self._last_call = time.time()
                return res.choices[0].message.content or ""
            except Exception as e:
                self._last_call = time.time()
                err = str(e)
                is_retryable = any(code in err for code in ("429", "502", "503", "504", "529"))
                if not is_retryable or attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 60.0)
        return ""

    def sample(self, prompt: str, n_samples: int = 1, max_tokens: int = 256) -> list[str]:
        responses: list[str] = []
        for _ in range(n_samples):
            text = self._call_with_retry(
                [{"role": "user", "content": prompt}],
                max_tokens,
            )
            responses.append(text)
        return responses
