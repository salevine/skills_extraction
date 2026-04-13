"""vLLM OpenAI-compatible HTTP client with endpoint pool."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import requests

from .config import PipelineConfig

logger = logging.getLogger(__name__)

# Module-level endpoint pool — thread-safe checkout/return
_endpoint_pool: Optional[queue.Queue] = None
_pool_lock: threading.Lock = threading.Lock()
_pool_endpoints: Optional[list] = None

# Thread-local HTTP sessions — each thread reuses its own TCP connections
_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Return a per-thread persistent session (created once per thread)."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})
        _thread_local.session = session
    return session


def _get_endpoint(cfg: PipelineConfig) -> str:
    """Block until an endpoint is available and return it (checkout)."""
    global _endpoint_pool, _pool_endpoints
    endpoints = cfg.vllm_endpoints()
    with _pool_lock:
        if _pool_endpoints != endpoints:
            _pool_endpoints = endpoints
            _endpoint_pool = queue.Queue()
            for ep in endpoints:
                _endpoint_pool.put(ep)
    return _endpoint_pool.get()  # blocks if all checked out


def _return_endpoint(endpoint: str) -> None:
    """Return an endpoint to the pool after use."""
    if _endpoint_pool is not None:
        _endpoint_pool.put(endpoint)


def call_vllm_direct(
    cfg: PipelineConfig,
    endpoint: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    role: str = "extractor",
) -> str:
    """Call a specific vLLM endpoint directly, bypassing the endpoint pool."""
    last_err: Optional[Exception] = None
    for attempt in range(cfg.vllm_max_retries):
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
        # Disable Qwen3 thinking mode to avoid wasting tokens on reasoning.
        # Only applied for Qwen models — other models (e.g. Mistral-Nemo)
        # don't support this chat_template kwarg.
        if cfg.disable_thinking and "qwen" in model.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            session = _get_session()
            t0 = time.perf_counter()
            r = session.post(url, json=payload, timeout=cfg.vllm_timeout_sec)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("empty choices in vLLM response")
            out = choices[0].get("message", {}).get("content", "")
            if not out:
                raise ValueError("empty content in vLLM response")
            elapsed = time.perf_counter() - t0
            logger.debug("vLLM %s (%s) %.1fs", role, endpoint, elapsed)
            if getattr(cfg, "llm_timing_callback", None):
                cfg.llm_timing_callback(model, elapsed, role)
            if cfg.vllm_per_call_delay_sec:
                time.sleep(cfg.vllm_per_call_delay_sec)
            return out
        except Exception as e:
            last_err = e
            logger.warning("vLLM attempt %s (%s) failed: %s", attempt + 1, endpoint, e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"vLLM failed after {cfg.vllm_max_retries} attempts: {last_err}")


def call_vllm_direct_with_failover(
    cfg: PipelineConfig,
    preferred_endpoint: str,
    all_endpoints: list,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    role: str = "extractor",
) -> str:
    """Try *preferred_endpoint* first, then fail over to other endpoints.

    Each endpoint gets ``cfg.vllm_max_retries`` attempts before the next
    endpoint is tried.  Raises after all endpoints are exhausted.
    """
    # Build ordered list: preferred first, then the rest
    ordered = [preferred_endpoint] + [ep for ep in all_endpoints if ep != preferred_endpoint]
    last_err: Optional[Exception] = None
    for endpoint in ordered:
        try:
            return call_vllm_direct(cfg, endpoint, model, system, user, temperature, role)
        except Exception as e:
            last_err = e
            logger.warning("vLLM failover: %s exhausted, trying next endpoint", endpoint)
    raise RuntimeError(f"vLLM failed on all {len(ordered)} endpoints: {last_err}")


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
        endpoint = _get_endpoint(cfg)
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
        if cfg.disable_thinking and "qwen" in model.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            session = _get_session()
            t0 = time.perf_counter()
            r = session.post(url, json=payload, timeout=cfg.vllm_timeout_sec)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("empty choices in vLLM response")
            out = choices[0].get("message", {}).get("content", "")
            if not out:
                raise ValueError("empty content in vLLM response")
            elapsed = time.perf_counter() - t0
            _return_endpoint(endpoint)
            logger.debug("vLLM %s (%s) %.1fs", role, endpoint, elapsed)
            if getattr(cfg, "llm_timing_callback", None):
                cfg.llm_timing_callback(model, elapsed, role)
            if cfg.vllm_per_call_delay_sec:
                time.sleep(cfg.vllm_per_call_delay_sec)
            return out
        except Exception as e:
            last_err = e
            logger.warning("vLLM attempt %s (%s) failed: %s", attempt + 1, endpoint, e)
            _return_endpoint(endpoint)  # return before retry sleep
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"vLLM failed after {cfg.vllm_max_retries} attempts: {last_err}")
