"""Ollama HTTP client with retries and JSON repair."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Optional

import requests

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def repair_json_text(text: str) -> str:
    t = text.strip()
    if "```" in t:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, re.I)
        if m:
            t = m.group(1).strip()
    m = re.search(r"[\[{][\s\S]*[\]}]", t)
    if m:
        t = m.group(0)
    return t


def parse_json_loose(text: str) -> Any:
    t = repair_json_text(text)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        t2 = re.sub(r",\s*([\]}])", r"\1", t)
        return json.loads(t2)


def call_ollama(
    cfg: PipelineConfig,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    role: str = "extractor",
) -> str:
    payload = {
        "model": model,
        "prompt": f"{system}\n\n{user}",
        "stream": False,
        "keep_alive": cfg.ollama_keep_alive,
        "options": {
            "temperature": temperature,
            "num_ctx": cfg.ollama_context_size,
        },
    }

    last_err: Optional[Exception] = None
    for attempt in range(cfg.ollama_max_retries):
        try:
            t0 = time.perf_counter()
            r = requests.post(
                cfg.generate_url(),
                json=payload,
                headers=cfg.ollama_headers(),
                timeout=cfg.ollama_timeout_sec,
            )
            r.raise_for_status()
            data = r.json()
            out = data.get("response", "")
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
            logger.warning("Ollama attempt %s failed: %s", attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Ollama failed after {cfg.ollama_max_retries} attempts: {last_err}")


def call_llm(
    cfg: PipelineConfig,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    role: str = "extractor",
) -> str:
    """Unified dispatcher: routes to Ollama or vLLM based on cfg.llm_backend."""
    if cfg.llm_backend == "vllm":
        from .llm_vllm import call_vllm

        return call_vllm(cfg, model, system, user, temperature, role)
    return call_ollama(cfg, model, system, user, temperature, role)
