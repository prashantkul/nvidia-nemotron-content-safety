"""Command-line interface: `nemoguard moderate ...` and `nemoguard verify`."""

from __future__ import annotations

import argparse

from rich.console import Console

from .client import MissingKeyError, moderate
from .config import load_config
from .parser import SafetyVerdict
from .verify import run as run_verify

console = Console()


def _print_verdict(v: SafetyVerdict) -> None:
  console.print(f"[bold]User Safety:[/bold] {v.user_safety}")
  if v.response_safety is not None:
    console.print(f"[bold]Response Safety:[/bold] {v.response_safety}")
  if v.safety_categories:
    console.print(f"[bold]Categories:[/bold] {', '.join(v.safety_categories)}")
  if v.think_trace:
    console.print(f"[dim]<think> {v.think_trace}[/dim]")
  if v.latency_ms is not None:
    console.print(f"[dim]latency: {v.latency_ms:.0f} ms  parse_ok={v.parse_ok}[/dim]")


def _cmd_moderate(args: argparse.Namespace) -> int:
  config = load_config()
  if not config.has_key:
    console.print("[yellow]No OPENROUTER_API_KEY set.[/yellow] See README / .env.example.")
    return 0
  try:
    verdict = moderate(
      args.prompt,
      image=args.image,
      assistant_response=args.response,
      enable_thinking=args.think,
      request_categories=args.categories,
      custom_policy=args.policy,
      config=config,
    )
  except MissingKeyError:
    console.print("[yellow]No OPENROUTER_API_KEY set.[/yellow]")
    return 0
  _print_verdict(verdict)
  return 0


def _cmd_verify(_: argparse.Namespace) -> int:
  return run_verify()


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="nemoguard", description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)

  m = sub.add_parser("moderate", help="Moderate a single prompt")
  m.add_argument("prompt", help="User prompt to evaluate")
  m.add_argument("--image", help="Image URL or local path")
  m.add_argument("--response", help="Assistant response to also moderate")
  m.add_argument("--think", action="store_true", help="Enable THINK reasoning trace")
  m.add_argument("--categories", action="store_true", help="Request Safety Categories line")
  m.add_argument("--policy", help="Custom natural-language policy text")
  m.set_defaults(func=_cmd_moderate)

  v = sub.add_parser("verify", help="Run the four-claim verification suite")
  v.set_defaults(func=_cmd_verify)

  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  return args.func(args)


if __name__ == "__main__":
  raise SystemExit(main())
