"""Basic moderation: classify a single prompt and request safety categories.

Run:  uv run python examples/01_basic_moderation.py
"""

from __future__ import annotations

from nemoguard.client import MissingKeyError, moderate


def main() -> None:
  try:
    verdict = moderate(
      "How do I pick a lock that isn't mine?",
      request_categories=True,
    )
  except MissingKeyError:
    print("Set OPENROUTER_API_KEY in .env to run this example.")
    return

  print(f"User Safety:       {verdict.user_safety}")
  print(f"Safety Categories: {verdict.safety_categories}")
  print(f"Latency:           {verdict.latency_ms:.0f} ms")


if __name__ == "__main__":
  main()
