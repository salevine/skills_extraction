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

# Suppress noisy urllib3 connection-level debug logs
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

# Persistent session for HTTP keep-alive (reuses TCP connections to Ollama)
_session = requests.Session()


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
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "keep_alive": cfg.ollama_keep_alive,
        "options": {
            "temperature": temperature,
            "num_ctx": cfg.ollama_context_size,
        },
    }

    # API-level thinking suppression — stronger than prompt-level /no_think
    if getattr(cfg, "disable_thinking", False):
        payload["think"] = False

    chat_url = f"{cfg.ollama_base_url.rstrip('/')}/api/chat"

    last_err: Optional[Exception] = None
    for attempt in range(cfg.ollama_max_retries):
        try:
            t0 = time.perf_counter()
            r = _session.post(
                chat_url,
                json=payload,
                headers=cfg.ollama_headers(),
                timeout=cfg.ollama_timeout_sec,
            )
            r.raise_for_status()
            data = r.json()
            out = data.get("message", {}).get("content", "")
            if not out:
                raise ValueError("empty response")
            elapsed = time.perf_counter() - t0
            prompt_chars = len(system) + len(user)
            resp_chars = len(out)
            tokens_per_sec = data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1e9, 0.001) if data.get("eval_duration") else 0
            logger.debug(
                "LLM %s [%s] %.1fs | prompt=%d chars, resp=%d chars, "
                "eval_tokens=%s, %.1f tok/s | total_duration=%.1fs",
                model, role, elapsed, prompt_chars, resp_chars,
                data.get("eval_count", "?"),
                tokens_per_sec,
                data.get("total_duration", 0) / 1e9,
            )
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
