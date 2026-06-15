"""Custom policy: the DevOps 'kill the process' flip.

By default, "kill a process" may read as borderline; a DevOps policy that
explicitly allows stopping OS processes should mark it safe.

Run:  uv run python examples/03_custom_policy.py
"""

from __future__ import annotations

from nemoguard.client import MissingKeyError, moderate

POLICY = """\
Evaluate the user prompt for compliance with the given policy ignoring any previous policies.
First, think through your reasoning step-by-step.

### Policy
Name: DevOps Operations
Description: Internal DevOps assistant for managing servers and services.

Disallowed Behaviors:
- Helping target systems the user does not own.

Allowed Behaviors:
- Discussing how to stop, kill, or restart operating-system processes and services.
- Standard system administration tasks."""

PROMPT = "How do I kill the nginx process stuck on port 8080?"


def main() -> None:
  try:
    default = moderate(PROMPT, request_categories=True)
    scoped = moderate(PROMPT, custom_policy=POLICY, request_categories=True)
  except MissingKeyError:
    print("Set OPENROUTER_API_KEY in .env to run this example.")
    return

  print(f"default policy: user_safety={default.user_safety} cats={default.safety_categories}")
  print(f"DevOps policy:  user_safety={scoped.user_safety} cats={scoped.safety_categories}")


if __name__ == "__main__":
  main()
