"""LLM dispatcher: routes calls to Ollama, OpenRouter, or vLLM based on cfg.backend."""

from __future__ import annotations

from .config import PipelineConfig


def call_llm(
    cfg: PipelineConfig,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    role: str = "extractor",
) -> str:
    if cfg.backend == "openrouter":
        from .llm_openrouter import call_openrouter

        return call_openrouter(cfg, model, system, user, temperature, role)
    if cfg.backend == "vllm":
        from .llm_vllm import call_vllm

        return call_vllm(cfg, model, system, user, temperature, role)
    from .llm_ollama import call_ollama

    return call_ollama(cfg, model, system, user, temperature, role)
