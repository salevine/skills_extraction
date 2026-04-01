"""
Pipeline configuration: models, thresholds, Ollama URL (aligned with Run_ollama patterns).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv

# Load .env when package is imported
_env_local = Path(__file__).resolve().parent.parent / ".env.local"
_env = Path(__file__).resolve().parent.parent / ".env"
if _env_local.exists():
    load_dotenv(_env_local)
elif _env.exists():
    load_dotenv(_env)


def resolve_ollama_base_url(use_local: bool = False) -> str:
    env_base = os.getenv("OLLAMA_BASE_URL", "").strip()
    if env_base:
        return env_base.rstrip("/")
    env_url = os.getenv("OLLAMA_URL", "").strip()
    if env_url:
        if "/api/generate" in env_url:
            return env_url.rsplit("/api/generate", 1)[0].rstrip("/")
        return env_url.rstrip("/")
    if use_local:
        return "http://localhost:11434"
    return "http://ollama.rs.gsu.edu"


@dataclass
class PipelineConfig:
    """Configurable knobs for the extraction pipeline."""

    # Models (Ollama tags)
    extractor_model: str = "qwen3:14b"
    verifier_model: str = "mistral-nemo:12b"
    requirement_model: str = "mistral-nemo:12b"
    hardsoft_model: str = "mistral-nemo:12b"
    fallback_model: str = "llama3.1:8b"

    ollama_base_url: str = field(default_factory=lambda: resolve_ollama_base_url(False))
    ollama_context_size: int = 32768
    ollama_keep_alive: str = "5m"
    ollama_timeout_sec: int = 300
    ollama_max_retries: int = 3
    per_call_delay_sec: float = 0.25

    # LLM backend: "ollama" (default) or "vllm"
    llm_backend: str = "ollama"

    # vLLM settings (only used when llm_backend="vllm")
    vllm_host: str = "localhost"
    vllm_base_port: int = 8001
    vllm_num_endpoints: int = 8
    vllm_max_retries: int = 3
    vllm_timeout_sec: int = 300

    # Line batching for LLM
    extractor_batch_max_lines: int = 5
    verifier_enabled: bool = True
    requirement_classifier_enabled: bool = True
    hardsoft_classifier_enabled: bool = True

    # Thresholds
    verifier_confidence_threshold: float = 0.72
    verify_on_unknown_hard_soft: bool = True
    verify_on_unclear_requirement: bool = True
    verify_on_uncertain_boilerplate: bool = True
    verify_on_low_extractor_confidence: float = 0.78

    # Quality
    quality_complete_min_score: float = 0.45

    # Pipeline
    pipeline_version: str = "3.0.0"
    skip_llm: bool = False  # for tests: candidates only

    # Runtime timing: (model: str, elapsed_sec: float, role: "extractor"|"verifier") -> None
    llm_timing_callback: Optional[Callable[[str, float, str], None]] = None

    def ollama_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "User-Agent": "skills_extraction/2.0",
        }

    def generate_url(self) -> str:
        return f"{self.ollama_base_url.rstrip('/')}/api/generate"


    def vllm_endpoints(self) -> list:
        """Generate vLLM endpoint URLs from host, base_port, and num_endpoints."""
        return [
            f"http://{self.vllm_host}:{self.vllm_base_port + i}/v1"
            for i in range(self.vllm_num_endpoints)
        ]

def load_config_from_env(overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    cfg = PipelineConfig()
    if os.getenv("SKILLS_EXTRACTOR_MODEL"):
        cfg.extractor_model = os.getenv("SKILLS_EXTRACTOR_MODEL", cfg.extractor_model)
    if os.getenv("SKILLS_VERIFIER_MODEL"):
        cfg.verifier_model = os.getenv("SKILLS_VERIFIER_MODEL", cfg.verifier_model)
    if os.getenv("SKILLS_REQUIREMENT_MODEL"):
        cfg.requirement_model = os.getenv("SKILLS_REQUIREMENT_MODEL", cfg.requirement_model)
    if os.getenv("SKILLS_HARDSOFT_MODEL"):
        cfg.hardsoft_model = os.getenv("SKILLS_HARDSOFT_MODEL", cfg.hardsoft_model)
    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg
