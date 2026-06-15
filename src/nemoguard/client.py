"""Thin OpenRouter client wrapper around the openai SDK for content-safety calls."""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path

from openai import APITimeoutError, OpenAI, RateLimitError

from .config import Config, load_config
from .parser import SafetyVerdict, parse_output

_DEFAULT_TIMEOUT = 60.0
_MAX_RETRIES = 3


class MissingKeyError(RuntimeError):
  """Raised when a live call is attempted without an API key."""


def _to_data_uri(path: str | Path) -> str:
  """Encode a local image file as a base64 data URI."""
  p = Path(path)
  mime = mimetypes.guess_type(p.name)[0] or "image/png"
  data = base64.b64encode(p.read_bytes()).decode("ascii")
  return f"data:{mime};base64,{data}"


def _resolve_image(image: str) -> str:
  """Return a usable image_url value: pass through URLs/data URIs, encode local paths."""
  if image.startswith(("http://", "https://", "data:")):
    return image
  return _to_data_uri(image)


def _apply_policy(user_prompt: str, custom_policy: str | None) -> str:
  """Embed a custom policy into the user turn.

  On the OpenRouter path the documented ``chat_template_kwargs.custom_policy`` knob
  is silently ignored, and a system message is ignored too -- the policy only
  reaches the model when it is part of the user turn. (A self-hosted NIM/vLLM that
  applies the official chat template can use ``custom_policy`` directly instead.)
  """
  if not custom_policy:
    return user_prompt
  return f"{custom_policy.strip()}\n\n### User Prompt\n{user_prompt}"


def _build_messages(
  user_prompt: str,
  image: str | None,
  assistant_response: str | None,
  custom_policy: str | None = None,
) -> list[dict]:
  content: list[dict] = []
  if image:
    content.append({"type": "image_url", "image_url": {"url": _resolve_image(image)}})
  content.append({"type": "text", "text": _apply_policy(user_prompt, custom_policy)})

  messages: list[dict] = [{"role": "user", "content": content}]
  if assistant_response is not None:
    messages.append(
      {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}
    )
  return messages


def _build_extra_body(
  enable_thinking: bool,
  request_categories: bool,
) -> dict:
  kwargs: dict = {"enable_thinking": enable_thinking}
  if request_categories:
    kwargs["request_categories"] = "/categories"
  return {"chat_template_kwargs": kwargs}


def _client(config: Config) -> OpenAI:
  return OpenAI(
    api_key=config.api_key,
    base_url=config.base_url,
    timeout=_DEFAULT_TIMEOUT,
    max_retries=0,  # we handle 429 backoff ourselves
  )


def moderate(
  user_prompt: str,
  *,
  image: str | None = None,
  assistant_response: str | None = None,
  enable_thinking: bool = False,
  request_categories: bool = False,
  custom_policy: str | None = None,
  config: Config | None = None,
) -> SafetyVerdict:
  """Run one moderation request and return a parsed :class:`SafetyVerdict`.

  Raises :class:`MissingKeyError` if no API key is configured.
  """
  config = config or load_config()
  if not config.has_key:
    raise MissingKeyError("OPENROUTER_API_KEY is not set; cannot make a live call.")

  client = _client(config)
  messages = _build_messages(user_prompt, image, assistant_response, custom_policy)
  extra_body = _build_extra_body(enable_thinking, request_categories)

  start = time.perf_counter()
  text, reasoning = _call_with_backoff(client, config.model_id, messages, extra_body)
  latency_ms = (time.perf_counter() - start) * 1000.0
  return parse_output(text, latency_ms=latency_ms, reasoning=reasoning)


def _call_with_backoff(
  client: OpenAI,
  model: str,
  messages: list[dict],
  extra_body: dict,
) -> tuple[str, str | None]:
  """Return (content, reasoning). OpenRouter exposes the THINK trace separately
  in ``message.reasoning`` rather than inline as <think> tags in content."""
  delay = 1.0
  last_exc: Exception | None = None
  for attempt in range(_MAX_RETRIES):
    try:
      resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        extra_body=extra_body,
      )
      message = resp.choices[0].message
      reasoning = getattr(message, "reasoning", None) or None
      return message.content or "", reasoning
    except (RateLimitError, APITimeoutError) as exc:
      last_exc = exc
      if attempt == _MAX_RETRIES - 1:
        break
      time.sleep(delay)
      delay *= 2
  raise RuntimeError(f"Moderation call failed after {_MAX_RETRIES} attempts: {last_exc}")
