"""vLLM OpenAI-compatible HTTP client with round-robin endpoint selection."""

from __future__ import annotations

import itertools
import logging
import time
from typing import Optional

import requests

from .config import PipelineConfig

logger = logging.getLogger(__name__)

# Module-level round-robin counter shared across calls
_endpoint_cycle: Optional[itertools.cycle] = None
_endpoint_list: Optional[list] = None


def _get_next_endpoint(cfg: PipelineConfig) -> str:
    """Return the next vLLM endpoint URL via round-robin."""
    global _endpoint_cycle, _endpoint_list
    endpoints = cfg.vllm_endpoints()
    if _endpoint_list != endpoints:
        _endpoint_list = endpoints
        _endpoint_cycle = itertools.cycle(endpoints)
    return next(_endpoint_cycle)


def call_vllm(
    cfg: PipelineConfig,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    role: str = "extractor",
) -> str:
    """Call a vLLM endpoint using the OpenAI-compatible chat/completions API."""
    last_err: Optional[Exception] = None
    for attempt in range(cfg.vllm_max_retries):
        endpoint = _get_next_endpoint(cfg)
        url = f"{endpoint}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": False,
        }
        try:
            t0 = time.perf_counter()
            r = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=cfg.vllm_timeout_sec,
            )
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("empty choices in vLLM response")
            out = choices[0].get("message", {}).get("content", "")
            if not out:
                raise ValueError("empty content in vLLM response")
            elapsed = time.perf_counter() - t0
            if getattr(cfg, "llm_timing_callback", None):
                cfg.llm_timing_callback(model, elapsed, role)
            if cfg.per_call_delay_sec:
                time.sleep(cfg.per_call_delay_sec)
            return out
        except Exception as e:
            last_err = e
            logger.warning("vLLM attempt %s (%s) failed: %s", attempt + 1, endpoint, e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"vLLM failed after {cfg.vllm_max_retries} attempts: {last_err}")
