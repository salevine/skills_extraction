"""
Skills extraction pipeline v2 — open-vocabulary, line-level, audit-first.

See `pipeline.run_pipeline` and `cli.main` (run: ``python -m skills_extraction``).
"""

__version__ = "2.0.0"

from .config import PipelineConfig, load_config_from_env
from .pipeline import run_pipeline

__all__ = ["run_pipeline", "PipelineConfig", "load_config_from_env", "__version__"]
