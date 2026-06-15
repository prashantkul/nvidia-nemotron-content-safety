"""Configuration: load credentials and expose the model id."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

MODEL_ID = "nvidia/nemotron-3.5-content-safety:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Config:
  """Resolved runtime configuration."""

  api_key: str | None
  base_url: str
  model_id: str = MODEL_ID

  @property
  def has_key(self) -> bool:
    """True when a non-empty API key is available."""
    return bool(self.api_key)


def load_config() -> Config:
  """Load config from environment / .env.

  Missing key is not an error here; callers check ``has_key`` and degrade
  gracefully. This keeps offline use (e.g. unit tests) friction-free.
  """
  load_dotenv()
  api_key = os.getenv("OPENROUTER_API_KEY") or None
  base_url = os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL
  return Config(api_key=api_key, base_url=base_url)
