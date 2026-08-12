"""
ai_service.py
-------------
Provider-agnostic AI service layer.

Design goal: Flask routes must NEVER talk to Gemini (or any other provider)
directly. They talk to `AIService`, which depends on an `AIProvider`
abstraction (dependency inversion). Today the only concrete implementation
is `GeminiProvider`. Adding OpenAI, Anthropic, Azure OpenAI, or a local LLM
later means writing a new class that satisfies `AIProvider` and wiring it up
in `get_ai_service()` — nothing else in the application changes.

Classes
-------
AIProvider          Abstract base class every provider implementation must satisfy.
GeminiProvider       Concrete implementation using the official Google GenAI SDK.
AIService            Facade used by the rest of the app; wraps a provider and
                      adds cross-cutting concerns (timing, validation, logging).
AnalysisResult        Typed result object returned by AIService.analyze_ticket().
"""

from __future__ import annotations

import abc
import html as html_module
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from config import config
from prompts import build_system_prompt, build_ticket_analysis_prompt

logger = logging.getLogger(__name__)

REQUIRED_ANALYSIS_KEYS = {
    "summary",
    "category",
    "sub_category",
    "suggested_priority",
    "confidence",
    "possible_causes",
    "troubleshooting_steps",
    "customer_reply",
    "technician_notes",
}


class AIProviderError(Exception):
    """Raised when a provider fails to produce a usable response."""


@dataclass
class AnalysisResult:
    """Result of a single ticket analysis, including diagnostics."""

    data: dict[str, Any]
    model: str
    processing_time: float
    raw_response: str
    prompt: str
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True


class AIProvider(abc.ABC):
    """
    Abstract interface every AI provider implementation must satisfy.

    Only two methods are required for the MVP. Future providers (OpenAI,
    Anthropic, Azure OpenAI, local LLMs) implement this same interface.
    """

    name: str = "abstract"

    @abc.abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, model: Optional[str] = None) -> str:
        """
        Send a prompt to the provider and return the raw text response.
        `model` optionally overrides the provider's configured default for
        this single call (used for the widget's runtime model selector).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def chat(
        self, system_prompt: str, history: list[dict[str, str]], model: Optional[str] = None
    ) -> str:
        """
        Multi-turn conversational call. `history` is a list of
        {"role": "user"|"model", "text": "..."} dicts, most recent last —
        the caller (AIService.chat) has already appended the new user
        message before calling this. Returns the assistant's reply text.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Return True if the provider is reachable and configured correctly."""
        raise NotImplementedError


class GeminiProvider(AIProvider):
    """AIProvider implementation backed by Google Gemini (google-genai SDK)."""

    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            logger.warning("GEMINI_API_KEY is not set; Gemini calls will fail.")
        self._model = model
        self._api_key = api_key
        self._client = None  # Lazily initialized to keep import-time side effects minimal.

    def _get_client(self):
        if self._client is None:
            from google import genai  # Imported lazily so the module is optional at import time.

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str, model: Optional[str] = None) -> str:
        from google.genai import types  # Local import; keeps provider swap-out clean.

        client = self._get_client()
        effective_model = model or self._model
        try:
            response = client.models.generate_content(
                model=effective_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.2,
                    max_output_tokens=1024,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - convert any SDK error into a domain error
            logger.exception("Gemini generate_content call failed")
            message = str(exc)
            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                raise AIProviderError(
                    "Gemini API quota exceeded. Check your plan/billing at "
                    "aistudio.google.com, or wait for the rate limit to reset."
                ) from exc
            raise AIProviderError(f"Gemini request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise AIProviderError("Gemini returned an empty response.")
        return text

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            # A minimal, cheap call to confirm credentials/network are OK.
            client.models.generate_content(model=self._model, contents="ping")
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Gemini health check failed")
            return False

    def chat(
        self, system_prompt: str, history: list[dict[str, str]], model: Optional[str] = None
    ) -> str:
        from google.genai import types

        client = self._get_client()
        effective_model = model or self._model
        contents = [{"role": h["role"], "parts": [{"text": h["text"]}]} for h in history]

        try:
            response = client.models.generate_content(
                model=effective_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=512,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gemini chat call failed")
            message = str(exc)
            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                raise AIProviderError(
                    "Gemini API quota exceeded. Check your plan/billing at "
                    "aistudio.google.com, or wait for the rate limit to reset."
                ) from exc
            raise AIProviderError(f"Gemini chat request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise AIProviderError("Gemini returned an empty chat response.")
        return text


def _build_provider() -> AIProvider:
    """Factory that selects the configured provider implementation."""
    provider_name = config.ai_provider.lower()
    if provider_name == "gemini":
        return GeminiProvider(api_key=config.gemini_api_key, model=config.gemini_model)

    # Future providers get added here, e.g.:
    # if provider_name == "openai":
    #     return OpenAIProvider(api_key=config.openai_api_key, model=config.openai_model)

    raise ValueError(f"Unsupported AI_PROVIDER: {provider_name}")


def _clean_description(text: str) -> str:
    """
    Defensively strip any HTML that made it into a ticket's description,
    regardless of source (the widget's own ZOHODESK.get('ticket') call, or
    our own thread-fetch endpoint). Raw HTML in the prompt both looks bad
    to the technician and can throw off the model's JSON output.
    """
    if not text or "<" not in text:
        return text or ""
    stripped = re.sub(r"(?i)<(br|/p|/div|/li)\s*/?>", "\n", text)
    stripped = re.sub(r"<[^>]+>", "", stripped)
    stripped = html_module.unescape(stripped)
    stripped = re.sub(r"[ \t]+", " ", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


class AIService:
    """
    Facade used by the rest of the application.

    Wraps whatever `AIProvider` is configured and adds provider-agnostic
    concerns: prompt construction, timing, JSON validation/parsing, and
    logging. Flask routes should only ever import and use `AIService`.
    """

    def __init__(self, provider: AIProvider) -> None:
        self._provider = provider

    @property
    def model_name(self) -> str:
        return getattr(self._provider, "_model", self._provider.name)

    def health_check(self) -> bool:
        return self._provider.health_check()

    def chat(
        self,
        ticket: dict[str, Any],
        history: list[dict[str, str]],
        message: str,
        analysis: Optional[dict[str, Any]] = None,
        model_override: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        """
        Send a chat message about the given ticket, with prior conversation
        history. `analysis`, when provided, is the structured result from a
        prior /api/analyze call for this same ticket — folded into the
        system prompt so the chat model stays consistent with conclusions
        already shown to the technician, instead of independently
        re-deriving (and potentially diverging from) them on every turn.
        Returns (reply_text, warnings) — warnings is empty on success, or
        contains one entry describing the failure (e.g. quota exceeded) so
        the route can surface a specific error to the widget.
        """
        from prompts import build_chat_system_prompt

        system_prompt = build_chat_system_prompt(
            {**ticket, "description": _clean_description(ticket.get("description", ""))},
            analysis=analysis,
        )
        full_history = history + [{"role": "user", "text": message}]

        try:
            reply = self._provider.chat(system_prompt, full_history, model=model_override)
            return reply, []
        except AIProviderError as exc:
            logger.error("Chat failed for ticket_id=%s: %s", ticket.get("ticket_id"), exc)
            return "", [str(exc)]

    def analyze_ticket(self, ticket: dict[str, Any], model_override: Optional[str] = None) -> AnalysisResult:
        """
        Analyze a ticket and return a structured, validated result.

        `model_override`, when provided, is used for this call instead of
        the configured default model (used by the widget's runtime model
        selector). Caller is responsible for validating it against an
        allowlist — see app.py.

        Never raises for "expected" failure modes (invalid JSON, provider
        errors) — those are captured in the returned AnalysisResult so the
        caller can decide how to present them. Only truly unexpected errors
        propagate.
        """
        prompt = build_ticket_analysis_prompt(
            {**ticket, "description": _clean_description(ticket.get("description", ""))}
        )
        system_prompt = build_system_prompt(config.ticket_categories, config.ticket_sub_categories)
        effective_model = model_override or self.model_name
        warnings: list[str] = []

        start = time.monotonic()
        try:
            raw_response = self._provider.generate(system_prompt, prompt, model=model_override)
        except AIProviderError as exc:
            elapsed = time.monotonic() - start
            logger.error("AI provider error for ticket %s: %s", ticket.get("ticket_id"), exc)
            return AnalysisResult(
                data={},
                model=effective_model,
                processing_time=elapsed,
                raw_response="",
                prompt=prompt,
                warnings=[str(exc)],
                is_valid=False,
            )
        elapsed = time.monotonic() - start

        parsed, parse_warnings = self._parse_and_validate(raw_response)
        warnings.extend(parse_warnings)

        return AnalysisResult(
            data=parsed or {},
            model=effective_model,
            processing_time=elapsed,
            raw_response=raw_response,
            prompt=prompt,
            warnings=warnings,
            is_valid=parsed is not None,
        )

    @staticmethod
    def _parse_and_validate(raw_response: str) -> tuple[Optional[dict[str, Any]], list[str]]:
        """Parse the raw model output as JSON and validate required keys."""
        warnings: list[str] = []
        cleaned = raw_response.strip()

        # Defensive cleanup in case the model wraps output in markdown fences
        # despite instructions not to.
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            warnings.append(f"Response was not valid JSON: {exc}")
            return None, warnings

        if not isinstance(parsed, dict):
            warnings.append("Response JSON was not an object.")
            return None, warnings

        missing = REQUIRED_ANALYSIS_KEYS - parsed.keys()
        if missing:
            warnings.append(f"Response missing expected keys: {sorted(missing)}")

        return parsed, warnings


_ai_service_singleton: Optional[AIService] = None


def get_ai_service() -> AIService:
    """Return a process-wide singleton AIService instance."""
    global _ai_service_singleton
    if _ai_service_singleton is None:
        _ai_service_singleton = AIService(provider=_build_provider())
    return _ai_service_singleton