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
    "kb_sources",
    "used_general_knowledge",
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


REQUIRED_KB_EXTRACTION_KEYS = {
    "extractable",
    "title",
    "symptoms",
    "cause",
    "resolution",
    "keywords",
    "related_systems",
}


@dataclass
class KBExtractionResult:
    """Result of extracting KB fields from a resolved ticket's conversation."""

    data: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True

    @property
    def extractable(self) -> bool:
        return bool(self.is_valid and self.data.get("extractable"))


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
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        model: Optional[str] = None,
        enable_web_grounding: bool = False,
    ) -> tuple[str, list[dict[str, str]]]:
        """
        Multi-turn conversational call. `history` is a list of
        {"role": "user"|"model", "text": "..."} dicts, most recent last —
        the caller (AIService.chat) has already appended the new user
        message before calling this.

        `enable_web_grounding`, when True, lets the provider decide
        per-message whether an actual web search is warranted (not forced
        every call) — see GeminiProvider.chat for the concrete mechanism.
        Providers that don't support grounding simply ignore this flag and
        return an empty source list; callers should not assume sources are
        always present even when the flag is True.

        Returns (reply_text, web_sources) — web_sources is a list of
        {"title": ..., "url": ...} dicts for any real citations the
        provider's search actually used this turn, or [] if none/not
        supported. This is intentionally separate from KB-article
        attribution, which the model reports inline in the reply text
        itself per the chat system prompt's instructions — web_sources
        here are ONLY ever real, provider-verified URLs, never invented.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Return True if the provider is reachable and configured correctly."""
        raise NotImplementedError

    @abc.abstractmethod
    def embed(self, text: str, task_type: Optional[str] = None) -> list[float]:
        """
        Return an embedding vector for `text`, used by the KB Builder's
        similarity/dedup check (see kb_service.py). Separate from
        generate()/chat() since embedding is a distinct model family, not
        a text-generation call.

        `task_type` (e.g. "SEMANTIC_SIMILARITY", "RETRIEVAL_QUERY",
        "RETRIEVAL_DOCUMENT") materially changes what the embedding
        optimizes for — leaving it unset degrades retrieval quality
        significantly (documented Gemini API behavior, not a minor
        tuning knob). Callers should always pass one; see kb_service for
        which type applies to which comparison.
        """
        raise NotImplementedError


class GeminiProvider(AIProvider):
    """AIProvider implementation backed by Google Gemini (google-genai SDK)."""

    name = "gemini"

    def __init__(self, api_key: str, model: str, embedding_model: Optional[str] = None) -> None:
        if not api_key:
            logger.warning("GEMINI_API_KEY is not set; Gemini calls will fail.")
        self._model = model
        self._api_key = api_key
        self._embedding_model = embedding_model or "gemini-embedding-001"
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
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        model: Optional[str] = None,
        enable_web_grounding: bool = False,
    ) -> tuple[str, list[dict[str, str]]]:
        from google.genai import types

        client = self._get_client()
        effective_model = model or self._model
        contents = [{"role": h["role"], "parts": [{"text": h["text"]}]} for h in history]

        # Google Search grounding is mutually exclusive with structured
        # JSON output at the API level (a real, current Gemini API
        # constraint — "Search Grounding can't be used with JSON/YAML/XML
        # mode"), which is exactly why this is only ever wired up for
        # chat (plain text), never for analyze_ticket (strict JSON
        # schema). The tool is offered, not forced — Gemini decides per
        # message whether a search is actually warranted.
        tools = [types.Tool(google_search=types.GoogleSearch())] if enable_web_grounding else None

        try:
            response = client.models.generate_content(
                model=effective_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=512,
                    tools=tools,
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

        web_sources = self._extract_grounding_sources(response) if enable_web_grounding else []
        return text, web_sources

    @staticmethod
    def _extract_grounding_sources(response: Any) -> list[dict[str, str]]:
        """
        Best-effort extraction of real source URLs from Gemini's grounding
        metadata, when the model actually used a search this turn (it
        often won't, even with the tool available). Deliberately
        defensive — every attribute access is guarded, because a shape
        mismatch here must never break the chat reply itself; degrading
        to "no web sources shown" is always safe, inventing a source
        would not be.
        """
        sources: list[dict[str, str]] = []
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                return sources
            grounding_metadata = getattr(candidates[0], "grounding_metadata", None)
            if grounding_metadata is None:
                return sources
            chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
            seen_urls: set[str] = set()
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web is None:
                    continue
                url = getattr(web, "uri", None)
                title = getattr(web, "title", None)
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({"title": title or url, "url": url})
        except Exception:  # noqa: BLE001 - grounding metadata shape is best-effort, never fatal
            logger.warning("Could not parse Gemini grounding metadata — showing reply without web sources.")
            return []
        return sources

    def embed(self, text: str, task_type: Optional[str] = None) -> list[float]:
        from google.genai import types

        client = self._get_client()
        try:
            config = types.EmbedContentConfig(task_type=task_type) if task_type else None
            response = client.models.embed_content(model=self._embedding_model, contents=text, config=config)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gemini embed_content call failed")
            raise AIProviderError(f"Gemini embedding request failed: {exc}") from exc

        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise AIProviderError("Gemini returned no embedding.")
        values = getattr(embeddings[0], "values", None)
        if not values:
            raise AIProviderError("Gemini embedding response had no values.")
        return list(values)


def _build_provider() -> AIProvider:
    """Factory that selects the configured provider implementation."""
    provider_name = config.ai_provider.lower()
    if provider_name == "gemini":
        return GeminiProvider(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
            embedding_model=config.gemini_embedding_model,
        )

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
    ) -> tuple[str, list[str], list[dict[str, str]]]:
        """
        Send a chat message about the given ticket, with prior conversation
        history. `analysis`, when provided, is the structured result from a
        prior /api/analyze call for this same ticket — folded into the
        system prompt so the chat model stays consistent with conclusions
        already shown to the technician, instead of independently
        re-deriving (and potentially diverging from) them on every turn.

        Returns (reply_text, warnings, web_sources). `warnings` is empty on
        success, or contains one entry describing the failure (e.g. quota
        exceeded). `web_sources` is real, provider-verified URLs the model
        actually cited via Google Search grounding this turn (see
        GeminiProvider.chat) — empty if grounding is disabled, wasn't
        needed for this message, or isn't supported by the provider. KB
        article attribution is separate: the model reports that inline in
        reply_text itself, per build_chat_system_prompt's instructions.
        """
        from prompts import build_chat_system_prompt

        system_prompt = build_chat_system_prompt(
            {**ticket, "description": _clean_description(ticket.get("description", ""))},
            analysis=analysis,
        )
        full_history = history + [{"role": "user", "text": message}]

        try:
            reply, web_sources = self._provider.chat(
                system_prompt, full_history, model=model_override,
                enable_web_grounding=config.kb_enable_web_grounding_in_chat,
            )
            return reply, [], web_sources
        except AIProviderError as exc:
            # Google Search grounding is billed/quota'd SEPARATELY from
            # plain text generation (a real, documented Gemini API
            # behavior — grounding has its own, often much smaller,
            # quota bucket). That means Chat can hit RESOURCE_EXHAUSTED
            # from grounding alone while normal generation (and Analyze,
            # which never uses grounding) is unaffected. Rather than
            # surface that as a dead end, retry once with grounding off —
            # a plain, ungrounded reply is a better outcome for the
            # technician than no reply at all, and this only fires when
            # grounding was actually the thing that failed.
            message_text = str(exc)
            is_quota_error = "quota" in message_text.lower() or "RESOURCE_EXHAUSTED" in message_text
            if config.kb_enable_web_grounding_in_chat and is_quota_error:
                logger.warning(
                    "Chat grounding call hit quota for ticket_id=%s — retrying without web grounding.",
                    ticket.get("ticket_id"),
                )
                try:
                    reply, _ = self._provider.chat(
                        system_prompt, full_history, model=model_override, enable_web_grounding=False
                    )
                    return reply, [], []
                except AIProviderError as retry_exc:
                    logger.error(
                        "Chat retry without grounding also failed for ticket_id=%s: %s",
                        ticket.get("ticket_id"), retry_exc,
                    )
                    return "", [str(retry_exc)], []

            logger.error("Chat failed for ticket_id=%s: %s", ticket.get("ticket_id"), exc)
            return "", [str(exc)], []

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

    def embed_text(self, text: str, task_type: Optional[str] = None) -> Optional[list[float]]:
        """
        Return an embedding vector for `text`, or None on failure. Never
        raises — the KB scan job treats an embedding failure the same as
        "no match found" and falls through to creating a new draft, rather
        than blocking the whole scan pass on one bad embedding call.

        `task_type` matters — see AIProvider.embed. Callers in kb_service
        pass "SEMANTIC_SIMILARITY" for dedup/reinforcement comparisons
        (two similar-shaped ticket summaries being compared to each
        other) and "RETRIEVAL_QUERY"/"RETRIEVAL_DOCUMENT" for the
        Analyze/Chat relevance search (an asymmetric short-query vs.
        stored-article comparison) — mixing these up is what caused
        search results to look unrelated even for legitimate matches.
        """
        try:
            return self._provider.embed(text, task_type=task_type)
        except AIProviderError as exc:
            logger.error("Embedding failed: %s", exc)
            return None

    def extract_kb_fields(self, ticket_id: str, subject: str, conversation: str) -> KBExtractionResult:
        """
        Extract Symptoms/Cause/Resolution/Keywords/Related systems from a
        resolved ticket's full conversation. Returns is_valid=False (never
        raises) on a provider error so the scan job can skip this ticket
        and continue with the rest of the batch.
        """
        from prompts import KB_EXTRACTION_SYSTEM_PROMPT, build_kb_extraction_prompt

        prompt = build_kb_extraction_prompt(ticket_id, subject, conversation)
        try:
            raw_response = self._provider.generate(KB_EXTRACTION_SYSTEM_PROMPT, prompt)
        except AIProviderError as exc:
            logger.error("KB extraction failed for ticket_id=%s: %s", ticket_id, exc)
            return KBExtractionResult(data={}, warnings=[str(exc)], is_valid=False)

        parsed, warnings = self._parse_and_validate(raw_response, required_keys=REQUIRED_KB_EXTRACTION_KEYS)
        return KBExtractionResult(data=parsed or {}, warnings=warnings, is_valid=parsed is not None)

    def check_new_detail(self, article: dict[str, Any], new_ticket_summary: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Ask whether a new occurrence already matched to `article` adds any
        genuinely new detail worth folding in. Returns None on failure
        (caller should treat that as "no new detail, just bump the
        counter") rather than raising, since this is a nice-to-have
        refinement on top of the core reinforcement logic, not something
        that should block a scan pass.
        """
        from prompts import KB_MERGE_SYSTEM_PROMPT, build_kb_merge_prompt

        prompt = build_kb_merge_prompt(article, new_ticket_summary)
        try:
            raw_response = self._provider.generate(KB_MERGE_SYSTEM_PROMPT, prompt)
        except AIProviderError as exc:
            logger.warning("KB merge-detail check failed: %s", exc)
            return None

        parsed, warnings = self._parse_and_validate(
            raw_response, required_keys={"has_new_detail", "updated_symptoms", "updated_related_systems", "note"}
        )
        if warnings:
            logger.warning("KB merge-detail check response issues: %s", warnings)
        return parsed

    @staticmethod
    def _parse_and_validate(
        raw_response: str, required_keys: set[str] = REQUIRED_ANALYSIS_KEYS
    ) -> tuple[Optional[dict[str, Any]], list[str]]:
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

        missing = required_keys - parsed.keys()
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