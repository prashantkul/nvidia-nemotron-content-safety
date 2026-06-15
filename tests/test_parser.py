"""Offline unit tests for the plain-text output parser. No API key needed."""

from __future__ import annotations

from nemoguard.parser import SafetyVerdict, parse_output


def test_basic_safe():
  v = parse_output("User Safety: safe")
  assert v.parse_ok
  assert v.user_safety == "safe"
  assert v.response_safety is None
  assert v.safety_categories == []
  assert not v.user_unsafe


def test_user_and_response():
  text = "User Safety: unsafe\nResponse Safety: safe"
  v = parse_output(text)
  assert v.user_safety == "unsafe"
  assert v.response_safety == "safe"
  assert v.user_unsafe
  assert not v.response_unsafe


def test_categories_parsed():
  text = (
    "User Safety: unsafe\n"
    "Response Safety: unsafe\n"
    "Safety Categories: Criminal Planning/Confessions, Controlled Substances"
  )
  v = parse_output(text)
  assert v.safety_categories == ["Criminal Planning/Confessions", "Controlled Substances"]


def test_think_trace_extracted():
  text = (
    "<think>\nThe user asks how to synthesize a drug. This is unsafe.\n</think>\n"
    "User Safety: unsafe\n"
    "Response Safety: unsafe\n"
    "Safety Categories: Controlled Substances"
  )
  v = parse_output(text)
  assert v.think_trace is not None
  assert "synthesize" in v.think_trace
  assert v.user_safety == "unsafe"
  assert v.response_safety == "unsafe"
  assert v.safety_categories == ["Controlled Substances"]


def test_think_block_not_leaked_into_safety_lines():
  text = "<think>User Safety: this is just reasoning text</think>\nUser Safety: safe"
  v = parse_output(text)
  assert v.user_safety == "safe"
  assert v.parse_ok


def test_missing_response_line():
  v = parse_output("User Safety: safe\nSafety Categories: None")
  assert v.user_safety == "safe"
  assert v.response_safety is None
  assert v.safety_categories == []


def test_categories_none_token():
  v = parse_output("User Safety: safe\nSafety Categories: none")
  assert v.safety_categories == []


def test_malformed_output_never_throws():
  v = parse_output("the model said something weird and unstructured")
  assert isinstance(v, SafetyVerdict)
  assert not v.parse_ok
  assert v.user_safety is None
  assert v.raw_text == "the model said something weird and unstructured"


def test_empty_output():
  v = parse_output("")
  assert not v.parse_ok
  assert v.raw_text == ""


def test_none_output():
  v = parse_output(None)  # type: ignore[arg-type]
  assert not v.parse_ok
  assert v.raw_text == ""


def test_latency_passthrough():
  v = parse_output("User Safety: safe", latency_ms=123.4)
  assert v.latency_ms == 123.4


def test_case_insensitive_and_whitespace():
  text = "  user safety:   UNSAFE  \n  response safety: Safe "
  v = parse_output(text)
  assert v.user_safety == "unsafe"
  assert v.response_safety == "safe"


def test_reasoning_fallback_used_as_think_trace():
  # OpenRouter surfaces THINK output via message.reasoning, not inline <think>.
  v = parse_output("User Safety: unsafe", reasoning="The user requests harmful info.")
  assert v.think_trace == "The user requests harmful info."
  assert v.user_safety == "unsafe"


def test_inline_think_wins_over_reasoning_param():
  v = parse_output("<think>inline</think>\nUser Safety: safe", reasoning="external")
  assert v.think_trace == "inline"


def test_empty_reasoning_param_ignored():
  v = parse_output("User Safety: safe", reasoning="   ")
  assert v.think_trace is None


def test_rambling_response_safety_prose_ignored():
  # The model sometimes echoes the instructions in prose. A "Response Safety:"
  # line that is not a safe/unsafe token must not be captured as a verdict.
  text = (
    'Response Safety: (since no assistant response present, we omit? '
    'The instruction says omit if none). So just output user safety.\n'
    "User Safety: unsafe"
  )
  v = parse_output(text)
  assert v.user_safety == "unsafe"
  assert v.response_safety is None


def test_last_user_safety_wins_after_rambling():
  text = (
    "We need to decide. User Safety: could be unsafe, let me think.\n"
    "User Safety: safe"
  )
  v = parse_output(text)
  assert v.user_safety == "safe"


def test_empty_think_block():
  v = parse_output("<think></think>\nUser Safety: safe")
  assert v.think_trace is None
  assert v.user_safety == "safe"
