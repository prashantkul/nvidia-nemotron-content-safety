"""Client tests. Pure helpers run offline; live calls skip without a key."""

from __future__ import annotations

import base64

import pytest

from nemoguard.client import (
  MissingKeyError,
  _build_extra_body,
  _build_messages,
  _resolve_image,
  moderate,
)
from nemoguard.config import Config, load_config


def test_build_messages_text_only():
  msgs = _build_messages("hello", None, None)
  assert msgs == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_build_messages_with_image_url():
  msgs = _build_messages("desc", "https://x/y.png", None)
  content = msgs[0]["content"]
  assert content[0] == {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
  assert content[1]["type"] == "text"


def test_build_messages_with_assistant_response():
  msgs = _build_messages("p", None, "the answer")
  assert msgs[1]["role"] == "assistant"
  assert msgs[1]["content"][0]["text"] == "the answer"


def test_build_extra_body_flags():
  body = _build_extra_body(True, True, "POLICY")
  kw = body["chat_template_kwargs"]
  assert kw["enable_thinking"] is True
  assert kw["request_categories"] == "/categories"
  assert kw["custom_policy"] == "POLICY"


def test_build_extra_body_minimal():
  body = _build_extra_body(False, False, None)
  kw = body["chat_template_kwargs"]
  assert kw == {"enable_thinking": False}


def test_resolve_image_local_file_to_data_uri(tmp_path):
  img = tmp_path / "pic.png"
  img.write_bytes(b"\x89PNG\r\n")
  uri = _resolve_image(str(img))
  assert uri.startswith("data:image/png;base64,")
  decoded = base64.b64decode(uri.split(",", 1)[1])
  assert decoded == b"\x89PNG\r\n"


def test_moderate_raises_without_key():
  cfg = Config(api_key=None, base_url="https://openrouter.ai/api/v1")
  with pytest.raises(MissingKeyError):
    moderate("hi", config=cfg)


@pytest.mark.skipif(not load_config().has_key, reason="no OPENROUTER_API_KEY set")
def test_live_moderate_smoke():
  v = moderate("Hello, how are you today?", request_categories=True)
  assert v.user_safety in {"safe", "unsafe"}
  assert v.parse_ok
