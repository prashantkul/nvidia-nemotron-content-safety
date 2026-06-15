"""Parse the plain-text Nemotron content-safety output into a structured verdict."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_USER_RE = re.compile(r"^\s*User Safety:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_RESP_RE = re.compile(r"^\s*Response Safety:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_CAT_RE = re.compile(r"^\s*Safety Categories:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass
class SafetyVerdict:
  """Structured result of a moderation call.

  ``parse_ok`` is False when no recognizable safety lines were found; the raw
  text is always preserved so nothing is lost on malformed output.
  """

  user_safety: str | None = None
  response_safety: str | None = None
  safety_categories: list[str] = field(default_factory=list)
  think_trace: str | None = None
  raw_text: str = ""
  latency_ms: float | None = None
  parse_ok: bool = False

  @property
  def user_unsafe(self) -> bool:
    return (self.user_safety or "").lower() == "unsafe"

  @property
  def response_unsafe(self) -> bool:
    return (self.response_safety or "").lower() == "unsafe"


def _norm_safety(value: str | None) -> str | None:
  """Normalize a safety token to 'safe'/'unsafe' when recognizable."""
  if value is None:
    return None
  token = value.strip().lower()
  if token.startswith("unsafe"):
    return "unsafe"
  if token.startswith("safe"):
    return "safe"
  return value.strip() or None


def _parse_categories(raw: str | None) -> list[str]:
  if not raw:
    return []
  cleaned = raw.strip()
  if cleaned.lower() in {"none", "n/a", "-"}:
    return []
  return [c.strip() for c in cleaned.split(",") if c.strip()]


def parse_output(
  text: str,
  latency_ms: float | None = None,
  reasoning: str | None = None,
) -> SafetyVerdict:
  """Parse model output into a :class:`SafetyVerdict`. Never raises.

  ``reasoning`` is an out-of-band trace (OpenRouter surfaces THINK output in
  ``message.reasoning`` rather than inline <think> tags); it is used as the
  think_trace when no inline block is present.
  """
  raw = text or ""
  verdict = SafetyVerdict(raw_text=raw, latency_ms=latency_ms)

  think_match = _THINK_RE.search(raw)
  if think_match:
    verdict.think_trace = think_match.group(1).strip() or None
  elif reasoning and reasoning.strip():
    verdict.think_trace = reasoning.strip()

  body = _THINK_RE.sub("", raw)

  # Use the LAST match of each label: when the model rambles before emitting the
  # final verdict block, prose may contain stray "Response Safety:" substrings.
  user_match = _last(_USER_RE, body)
  if user_match:
    verdict.user_safety = _norm_safety(user_match.group(1))

  resp_match = _last(_RESP_RE, body)
  if resp_match:
    # Only accept a recognizable safety token; ignore prose that merely mentions
    # the label (e.g. quoting the instructions inside a reasoning paragraph).
    resp = _norm_safety(resp_match.group(1))
    if resp in {"safe", "unsafe"}:
      verdict.response_safety = resp

  cat_match = _last(_CAT_RE, body)
  if cat_match:
    verdict.safety_categories = _parse_categories(cat_match.group(1))

  verdict.parse_ok = verdict.user_safety is not None
  return verdict


def _last(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
  matches = list(pattern.finditer(text))
  return matches[-1] if matches else None
