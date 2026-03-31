"""OpenRouter HTTP client (OpenAI-compatible) with retries and backoff."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

from .config import PipelineConfig

logger = logging.getLogger(__name__)

_session = requests.Session()


def call_openrouter(
    cfg: PipelineConfig,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    role: str = "extractor",
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Add it to .env or export it."
        )

    url = f"{cfg.openrouter_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/skills-extraction",
        "X-OpenRouter-Title": "skills_extraction",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": cfg.openrouter_max_tokens,
    }

    last_err: Optional[Exception] = None
    for attempt in range(cfg.ollama_max_retries):
        try:
            t0 = time.perf_counter()
            r = _session.post(
                url,
                json=payload,
                headers=headers,
                timeout=cfg.ollama_timeout_sec,
            )
            r.raise_for_status()
            data = r.json()
            out = data["choices"][0]["message"]["content"]
            if not out:
                raise ValueError("empty response")
            elapsed = time.perf_counter() - t0
            if getattr(cfg, "llm_timing_callback", None):
                cfg.llm_timing_callback(model, elapsed, role)
            if cfg.per_call_delay_sec:
                time.sleep(cfg.per_call_delay_sec)
            return out
        except Exception as e:
            last_err = e
            logger.warning("OpenRouter attempt %s failed: %s", attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"OpenRouter failed after {cfg.ollama_max_retries} attempts: {last_err}"
    )
