"""
app.py
------
Flask application entrypoint.

Routes are intentionally thin: validate input, delegate to services
(AIService, database), and shape the HTTP response. No AI provider logic
and no SQL lives here — see ai_service.py and database.py.

Endpoints
---------
GET  /api/health    -> liveness + AI provider health
POST /api/analyze   -> analyze a ticket, persist the result, return JSON
"""

from __future__ import annotations

import datetime as _dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from ai_service import get_ai_service
from config import config
from database import get_session, init_db
from models import TicketAnalysis
import zoho_client

# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("service_desk_copilot")

REQUIRED_TICKET_FIELDS = ("ticket_id", "subject")

# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
# Cheap insurance for an internal tool: the widget's backend URL is visible
# to anyone who inspects network traffic from an installed extension, even
# though the Gemini/Zoho secrets behind it never are. Without a limit,
# anyone who finds that URL could call the AI endpoints directly (bypassing
# Desk entirely) and spend our Gemini quota. Keyed by IP, in-memory —
# that's fine for a single-instance deploy; if this ever runs behind a
# load balancer with multiple instances, swap storage_uri to a shared
# Redis instance (see Flask-Limiter docs) or the limit becomes per-instance
# rather than global.
limiter = Limiter(key_func=get_remote_address, default_limits=["120 per hour"])


def create_app() -> Flask:
    """Application factory. Keeps test setup and WSGI config clean."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.secret_key

    allowed_origins = (
        "*" if config.cors_allowed_origins.strip() == "*" else config.cors_allowed_origins.split(",")
    )
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
    limiter.init_app(app)

    init_db()
    register_routes(app)
    register_error_handlers(app)
    return app


def register_routes(app: Flask) -> None:
    @app.get("/api/health")
    def health() -> Any:
        ai_service = get_ai_service()
        ai_ok = ai_service.health_check()
        return jsonify(
            {
                "status": "ok" if ai_ok else "degraded",
                "ai_provider": config.ai_provider,
                "ai_model": ai_service.model_name,
                "ai_reachable": ai_ok,
                "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
        ), (200 if ai_ok else 503)

    @app.post("/api/chat")
    @limiter.limit("30 per hour")
    def chat_about_ticket() -> Any:
        """
        Conversational Q&A about a specific ticket. Stateless — the widget
        resends the full conversation history each turn; nothing is
        persisted server-side for this feature (unlike /api/analyze, which
        writes to ticket_analysis).
        """
        payload = request.get_json(silent=True) or {}
        ticket = payload.get("ticket") or {}
        analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else None
        message = (payload.get("message") or "").strip()
        history = payload.get("history") or []
        requested_model = payload.get("model")
        model_override = requested_model if requested_model in config.allowed_gemini_models else None

        if not ticket.get("ticket_id"):
            return jsonify({"error": "Request must include a ticket with a ticket_id."}), 400
        if not message:
            return jsonify({"error": "Message cannot be empty."}), 400
        if not isinstance(history, list):
            return jsonify({"error": "history must be a list."}), 400

        # Defensive shape-check on each history entry rather than trusting
        # the client fully — malformed entries would otherwise reach the
        # Gemini SDK and produce a confusing error deep in provider code.
        clean_history = [
            {"role": h.get("role"), "text": h.get("text")}
            for h in history
            if isinstance(h, dict) and h.get("role") in ("user", "model") and h.get("text")
        ]

        ai_service = get_ai_service()
        try:
            reply, warnings = ai_service.chat(
                ticket, clean_history, message, analysis=analysis, model_override=model_override
            )
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error in chat for ticket_id=%s", ticket.get("ticket_id"))
            return jsonify({"error": "The chat request encountered an unexpected error."}), 502

        if warnings:
            return jsonify({"error": warnings[0]}), 502

        return jsonify({"reply": reply}), 200

    @app.get("/api/models")
    def list_models() -> Any:
        return jsonify(
            {
                "default_model": config.gemini_model,
                "available_models": list(config.allowed_gemini_models),
            }
        ), 200

    @app.get("/api/ticket/<ticket_id>/related")
    def get_related(ticket_id: str) -> Any:
        """
        Deterministic Location+Sub-Category match — identity-independent
        (see zoho_client.get_related_tickets docstring). Fetches the
        current ticket's own location/sub-category first, then finds
        others matching. For the top few results, also pulls a short
        "what was done" snippet — the last few agent messages, not just the
        final one, since the final message is often a canned closing note
        rather than the actual resolution detail — from each. Capped to
        RESOLUTION_SNIPPET_LIMIT so this doesn't scale API calls linearly
        with however many related tickets exist, only the ones actually
        shown as most relevant.
        """
        RESOLUTION_SNIPPET_LIMIT = 5

        try:
            details = zoho_client.get_ticket_details(ticket_id)
            related = zoho_client.get_related_tickets(
                details["location"], details["sub_category"], exclude_ticket_id=ticket_id
            )
        except zoho_client.ZohoAuthError as exc:
            logger.warning("Zoho auth error fetching related tickets for %s: %s", ticket_id, exc)
            return jsonify({"related_tickets": [], "count": 0, "error": str(exc)}), 200

        if related:
            snippet_targets = related[:RESOLUTION_SNIPPET_LIMIT]
            with ThreadPoolExecutor(max_workers=RESOLUTION_SNIPPET_LIMIT) as executor:
                snippets = list(
                    executor.map(lambda t: zoho_client.get_agent_resolution_context(t["ticket_id"]), snippet_targets)
                )
            for t, snippet in zip(snippet_targets, snippets):
                t["resolution_snippet"] = snippet or ""

        return jsonify(
            {
                "related_tickets": related,
                "count": len(related),
                "location": details["location"],
                "sub_category": details["sub_category"],
            }
        ), 200

    @app.get("/api/ticket/<ticket_id>/requester-history")
    def get_requester_history(ticket_id: str) -> Any:
        """
        Deterministic lookup of a requester's recent ticket history — no AI
        involved in the fetch itself. `email` is passed as a query param
        since the widget already has it from ZOHODESK.get('ticket').
        """
        email = request.args.get("email", "").strip()
        if not email:
            return jsonify({"error": "email query parameter is required."}), 400

        try:
            tickets = zoho_client.get_requester_ticket_history(email, exclude_ticket_id=ticket_id)
        except zoho_client.ZohoAuthError as exc:
            logger.warning("Zoho auth error fetching requester history for %s: %s", email, exc)
            return jsonify({"tickets": [], "count": 0, "error": str(exc)}), 200

        return jsonify({"tickets": tickets, "count": len(tickets)}), 200

    @app.get("/api/ticket/<ticket_id>/description")
    def get_ticket_description(ticket_id: str) -> Any:
        """
        Fetch the latest thread's content for a ticket via Zoho's Desk API.
        Used by the widget to fill in the description for tickets — mainly
        email-originated ones — where the ticket object's own `description`
        field is blank. Returns an empty description rather than an error
        when Zoho OAuth isn't configured, so the widget can fall back
        gracefully instead of breaking.
        """
        try:
            content = zoho_client.fetch_ticket_conversation(ticket_id)
        except zoho_client.ZohoAuthError as exc:
            logger.warning("Zoho auth error fetching thread for ticket_id=%s: %s", ticket_id, exc)
            return jsonify({"description": "", "error": str(exc)}), 200

        return jsonify({"description": content or ""}), 200

    @app.post("/api/analyze")
    @limiter.limit("30 per hour")
    def analyze_ticket() -> Any:
        request_started_at = _dt.datetime.now(_dt.timezone.utc)
        payload = request.get_json(silent=True) or {}

        ticket = payload.get("ticket") or {}
        developer_mode_requested = bool(payload.get("developer_mode", False))
        requested_model = payload.get("model")
        model_override = requested_model if requested_model in config.allowed_gemini_models else None
        if requested_model and model_override is None:
            logger.warning(
                "Ignoring unrecognized model request %r — not in ALLOWED_GEMINI_MODELS", requested_model
            )

        validation_error = _validate_ticket_payload(ticket)
        if validation_error:
            logger.warning("Rejected /api/analyze request: %s", validation_error)
            return jsonify({"error": validation_error}), 400

        ticket_id = str(ticket.get("ticket_id"))
        logger.info("Analyzing ticket_id=%s subject=%r", ticket_id, ticket.get("subject"))

        ai_service = get_ai_service()

        try:
            result = ai_service.analyze_ticket(ticket, model_override=model_override)
        except Exception:  # noqa: BLE001 - last line of defense; never leak stack traces to client
            logger.exception("Unexpected error analyzing ticket_id=%s", ticket_id)
            return jsonify(
                {"error": "The AI service encountered an unexpected error. Please try again."}
            ), 502

        if not result.is_valid:
            logger.error(
                "Analysis failed validation for ticket_id=%s warnings=%s", ticket_id, result.warnings
            )
            # If the failure came from the provider itself (e.g. quota
            # exhaustion), result.warnings[0] holds that specific message —
            # surface it directly rather than a generic fallback, since it
            # tells the technician exactly what's going on and what to do.
            specific_message = result.warnings[0] if result.warnings else None
            response_body: dict[str, Any] = {
                "error": specific_message or "The AI analysis could not be completed. Please try again.",
            }
            if developer_mode_requested and config.developer_mode_allowed:
                response_body["developer"] = _build_developer_payload(
                    result, request_started_at, ticket_id
                )
            return jsonify(response_body), 502

        _persist_analysis(ticket_id=ticket_id, subject=ticket.get("subject"), result=result)

        logger.info(
            "Analysis complete for ticket_id=%s in %.2fs (model=%s)",
            ticket_id,
            result.processing_time,
            result.model,
        )

        response_body = {"analysis": result.data}
        if developer_mode_requested and config.developer_mode_allowed:
            response_body["developer"] = _build_developer_payload(result, request_started_at, ticket_id)

        return jsonify(response_body), 200


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(_err: Any) -> Any:
        return jsonify({"error": "Not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(_err: Any) -> Any:
        return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(500)
    def internal_error(err: Any) -> Any:
        logger.exception("Unhandled server error: %s", err)
        return jsonify({"error": "Internal server error."}), 500


def _validate_ticket_payload(ticket: dict[str, Any]) -> str | None:
    if not ticket:
        return "Request must include a non-empty 'ticket' object."
    for field_name in REQUIRED_TICKET_FIELDS:
        if not ticket.get(field_name):
            return f"Ticket is missing required field: '{field_name}'."
    return None


def _persist_analysis(*, ticket_id: str, subject: str | None, result: Any) -> None:
    try:
        with get_session() as session:
            record = TicketAnalysis(
                ticket_id=ticket_id,
                subject=subject,
                analysis_json=_to_json_string(result.data),
                model=result.model,
                processing_time=result.processing_time,
            )
            session.add(record)
    except Exception:  # noqa: BLE001 - persistence failures must never break the API response
        logger.exception("Failed to persist analysis for ticket_id=%s", ticket_id)


def _to_json_string(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, ensure_ascii=False)


def _build_developer_payload(result: Any, started_at: _dt.datetime, ticket_id: str) -> dict[str, Any]:
    """Assemble the diagnostic payload shown when Developer Mode is on."""
    return {
        "ticket_id": ticket_id,
        "prompt_sent": result.prompt,
        "raw_response": result.raw_response,
        "processing_time_seconds": round(result.processing_time, 3),
        "model": result.model,
        "estimated_tokens": _estimate_tokens(result.prompt, result.raw_response),
        "validation_status": "valid" if result.is_valid else "invalid",
        "warnings": result.warnings,
        "request_timestamp": started_at.isoformat(),
    }


def _estimate_tokens(prompt: str, response: str) -> int:
    """
    Rough token estimate (~4 characters per token) shown only in Developer
    Mode for convenience. Not billing-accurate.
    """
    return round((len(prompt) + len(response)) / 4)


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.port, debug=config.debug)