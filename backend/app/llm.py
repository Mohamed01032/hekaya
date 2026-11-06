"""LLM providers.

`AnthropicProvider` calls Claude (vision + text). `OfflineProvider` is a
deterministic fallback used when no API key is configured — the app keeps
working in the museum even if the network or the AI service is down: it tells
the curated story and answers from the retrieved facts.
"""
from __future__ import annotations

import base64
import json
import re
from typing import Iterator, Protocol


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], *, images: list[tuple[bytes, str]] | None = None,
                 fast: bool = False, max_tokens: int = 700) -> str: ...

    def stream(self, system: str, messages: list[dict], *, fast: bool = False,
               max_tokens: int = 700) -> Iterator[str]: ...


def _with_images(messages: list[dict], images: list[tuple[bytes, str]] | None) -> list[dict]:
    if not images:
        return messages
    msgs = [dict(m) for m in messages]
    last = msgs[-1]
    blocks = [
        {"type": "image", "source": {"type": "base64", "media_type": mt,
                                     "data": base64.b64encode(data).decode()}}
        for data, mt in images
    ]
    blocks.append({"type": "text", "text": last["content"]})
    last["content"] = blocks
    return msgs


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, fast_model: str):
        import anthropic  # imported lazily so offline mode needs no SDK

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.fast_model = fast_model

    def complete(self, system, messages, *, images=None, fast=False, max_tokens=700) -> str:
        resp = self.client.messages.create(
            model=self.fast_model if fast else self.model,
            system=system,
            max_tokens=max_tokens,
            messages=_with_images(messages, images),
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    def stream(self, system, messages, *, fast=False, max_tokens=700) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.fast_model if fast else self.model,
            system=system,
            max_tokens=max_tokens,
            messages=messages,
        ) as s:
            yield from s.text_stream


class GeminiProvider:
    """Google Gemini (free key from Google AI Studio). Same interface as AnthropicProvider."""

    name = "gemini"

    def __init__(self, api_key: str, model: str, fast_model: str):
        from google import genai  # imported lazily so other modes need no SDK
        from google.genai import types

        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model = model
        self.fast_model = fast_model

    def _contents(self, messages: list[dict], images: list[tuple[bytes, str]] | None = None):
        t = self.types
        out = []
        for i, m in enumerate(messages):
            parts = [t.Part.from_text(text=m["content"])]
            if images and i == len(messages) - 1:
                parts = [t.Part.from_bytes(data=d, mime_type=mt) for d, mt in images] + parts
            out.append(t.Content(role="model" if m["role"] == "assistant" else "user", parts=parts))
        return out

    def _config(self, system: str, max_tokens: int):
        return self.types.GenerateContentConfig(
            system_instruction=system or None, max_output_tokens=max(max_tokens * 6, 4096))

    # Google retires model names over time (e.g. "gemini-flash-latest" can start returning 404).
    # When the configured name is not found, pick the newest available Flash model and keep using it.
    @staticmethod
    def _model_rank(name: str) -> tuple:
        n = name.rsplit("/", 1)[-1]
        if "flash" not in n or any(x in n for x in ("image", "tts", "live", "audio", "embedding", "thinking", "exp")):
            return ()
        nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", n.split("flash")[0])] or [0.0]
        return (0 if "lite" in n else 1, 0 if "preview" in n else 1, nums[0], -len(n))

    def _fallback_model(self) -> str | None:
        try:
            names = []
            for m in self.client.models.list():
                actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
                if actions and "generateContent" not in actions:
                    continue
                if self._model_rank(m.name):
                    names.append(m.name.rsplit("/", 1)[-1])
            return max(names, key=self._model_rank) if names else None
        except Exception:
            return None

    def _with_fallback(self, fast: bool, call):
        model = self.fast_model if fast else self.model
        try:
            return call(model)
        except Exception as exc:
            if "404" not in str(exc) and "NOT_FOUND" not in str(exc):
                raise
            better = self._fallback_model()
            if not better or better == model:
                raise
            self.model = self.fast_model = better
            return call(better)

    def complete(self, system, messages, *, images=None, fast=False, max_tokens=700) -> str:
        resp = self._with_fallback(fast, lambda model: self.client.models.generate_content(
            model=model, contents=self._contents(messages, images), config=self._config(system, max_tokens)))
        return (resp.text or "").strip()

    def stream(self, system, messages, *, fast=False, max_tokens=700) -> Iterator[str]:
        def first(model):
            it = iter(self.client.models.generate_content_stream(
                model=model, contents=self._contents(messages), config=self._config(system, max_tokens)))
            try:
                head = next(it)
            except StopIteration:
                head = None
            return head, it
        head, rest = self._with_fallback(fast, first)
        for chunk in ([head] if head is not None else []) + [None]:
            if chunk is not None and chunk.text:
                yield chunk.text
        for chunk in rest:
            if chunk.text:
                yield chunk.text


class OfflineProvider:
    """Template answers built from the facts embedded in the system prompt."""

    name = "offline"

    _FACT_LINE = re.compile(r"^- (.+)$", re.M)

    def complete(self, system, messages, *, images=None, fast=False, max_tokens=700) -> str:
        if images:
            # Cannot see without a vision model: ask the client to fall back to the display code.
            return json.dumps({"id": None, "confidence": 0.0,
                               "reason": "offline mode — enter the display code instead"})
        facts = self._FACT_LINE.findall(system)
        name_match = re.search(r"Name: (.+?) /", system)
        name = name_match.group(1) if name_match else "this piece"
        arabic = "Egyptian Arabic" in system or "Standard Arabic" in system
        question = messages[-1]["content"] if messages else ""
        if not facts:
            return question[:400]
        chosen = self._pick(facts, question)
        if arabic:
            return (f"أهلاً بيك! أنا {name}. " + " ".join(chosen) +
                    " (الدليل شغال دلوقتي من غير إنترنت، فبحكيلك من المعلومات الموثقة بس.)")
        return f"Hello! I am {name}. " + " ".join(chosen)

    def stream(self, system, messages, *, fast=False, max_tokens=700) -> Iterator[str]:
        text = self.complete(system, messages)
        for word in text.split(" "):
            yield word + " "

    @staticmethod
    def _pick(facts: list[str], question: str) -> list[str]:
        q = set(re.findall(r"\w+", question.lower()))
        ranked = sorted(facts, key=lambda f: -len(q & set(re.findall(r"\w+", f.lower()))))
        return ranked[:2] if q else facts[:2]


def build_provider(settings) -> LLMProvider:
    """Claude if its key is set, else Gemini if its key is set, else offline.
    HEKAYA_PROVIDER=anthropic|gemini|offline forces one."""
    order = {"anthropic": ["anthropic"], "gemini": ["gemini"], "offline": []}.get(
        settings.provider, ["anthropic", "gemini"])
    for name in order:
        try:
            if name == "anthropic" and settings.anthropic_api_key:
                return AnthropicProvider(settings.anthropic_api_key, settings.model, settings.fast_model)
            if name == "gemini" and settings.gemini_api_key:
                return GeminiProvider(settings.gemini_api_key, settings.gemini_model, settings.gemini_fast_model)
        except Exception:  # SDK missing or misconfigured: try the next one
            continue
    return OfflineProvider()


def parse_json(text: str) -> dict | None:
    """Tolerant JSON extraction from a model reply."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None
