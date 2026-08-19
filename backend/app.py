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
import functools
import hmac
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from flask import Flask, Response, jsonify, request
from flask import session as flask_session
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from ai_service import get_ai_service
from config import config
from database import get_session, init_db
import kb_service
from models import KBArticle, TicketAnalysis
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


def _validate_kb_session() -> Optional[str]:
    """
    Returns the logged-in KB reviewer's username if the session is valid
    (present, not idle-expired), refreshing the idle timer on every call —
    a sliding window, not a fixed one. Returns None (and clears the
    session if it was merely stale) otherwise.

    Uses flask_session (aliased on import) rather than `session`, since
    nearly every route in this file already binds `session` locally to a
    SQLAlchemy session via `with get_session() as session:` — importing
    Flask's session under the same name would silently shadow it inside
    those blocks, badly enough that `session.get("kb_admin_user")` would
    resolve to SQLAlchemy's `Session.get()` (an ORM primary-key lookup)
    instead of a dict-style read, a real and easy-to-miss bug.
    """
    username = flask_session.get("kb_admin_user")
    last_activity_str = flask_session.get("kb_admin_last_activity")
    if not username or not last_activity_str:
        return None

    try:
        last_activity = _dt.datetime.fromisoformat(last_activity_str)
    except ValueError:
        flask_session.clear()
        return None
    if last_activity.tzinfo is None:
        last_activity = last_activity.replace(tzinfo=_dt.timezone.utc)

    idle_for = _dt.datetime.now(_dt.timezone.utc) - last_activity
    if idle_for.total_seconds() > config.kb_admin_idle_timeout_minutes * 60:
        flask_session.clear()
        return None

    # Still valid — refresh so an actively-working reviewer never gets
    # logged out mid-session; only genuine inactivity trips the timeout.
    flask_session["kb_admin_last_activity"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return username


def _kb_session_required(view_func):
    """
    Session-auth guard for the KB review queue routes. Replaces Basic
    Auth specifically because Basic Auth cannot support logout or idle
    timeout at all — browsers cache and silently resend those credentials
    on every request forever, with no clean way to invalidate them
    short of the server permanently changing the password. A real login
    form + server-side session with a sliding idle window is the
    straightforward way to get both.

    If no admin accounts are configured, the routes are disabled entirely
    (403) rather than left open.
    """

    @functools.wraps(view_func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not config.kb_admin_users:
            return jsonify({"error": "KB review is not configured on this server."}), 403
        if not _validate_kb_session():
            return jsonify({"error": "Not authenticated."}), 401
        return view_func(*args, **kwargs)

    return wrapped


def create_app() -> Flask:
    """Application factory. Keeps test setup and WSGI config clean."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.secret_key
    # Hard ceiling on the KB review session regardless of activity — the
    # idle timeout (see _validate_kb_session) is a sliding window on top
    # of this; whichever limit is hit first ends the session.
    app.permanent_session_lifetime = _dt.timedelta(hours=config.kb_admin_session_lifetime_hours)

    allowed_origins = (
        "*" if config.cors_allowed_origins.strip() == "*" else config.cors_allowed_origins.split(",")
    )
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
    limiter.init_app(app)

    init_db()
    register_routes(app)
    register_error_handlers(app)
    _maybe_start_kb_scheduler()
    return app


def _maybe_start_kb_scheduler() -> None:
    """
    Start the background KB scan job if enabled. In-process APScheduler —
    no new infra. Guarded so importing app.py for tests, or running with
    KB_SCAN_ENABLED unset, never spins up a background thread hitting the
    Zoho/Gemini APIs unexpectedly.
    """
    if not config.kb_scan_enabled:
        logger.info("KB scan scheduler disabled (KB_SCAN_ENABLED is not set).")
        return

    from apscheduler.schedulers.background import BackgroundScheduler

    def _run_scan() -> None:
        try:
            kb_service.run_scan_pass(get_ai_service())
        except Exception:  # noqa: BLE001 - a bad pass must never kill the scheduler thread
            logger.exception("KB scan pass raised an unexpected error.")

    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        _run_scan,
        "interval",
        minutes=config.kb_scan_interval_minutes,
        next_run_time=_dt.datetime.now(),  # run once immediately on startup, then on the interval
        id="kb_scan",
    )
    scheduler.start()
    logger.info("KB scan scheduler started (every %s minutes).", config.kb_scan_interval_minutes)


def register_routes(app: Flask) -> None:
    @app.get("/ping")
    @limiter.exempt
    def ping() -> Any:
        """
        Trivial liveness check — no AI call, no DB query, no Zoho call.
        Exists purely so an external keep-alive pinger (e.g. a GitHub
        Actions cron every ~14 min) can stop Render's free-tier service
        from spinning down after 15 min of inactivity, without wasting
        Gemini quota the way pinging /api/health would (that route
        deliberately calls ai_service.health_check() to verify real
        reachability — not what a keep-alive ping needs). Exempted from
        rate limiting since a keep-alive job legitimately calls this very
        often, on a predictable schedule, and that's fine.
        """
        return jsonify({"status": "ok"}), 200

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
            reply, warnings, web_sources = ai_service.chat(
                ticket, clean_history, message, analysis=analysis, model_override=model_override
            )
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error in chat for ticket_id=%s", ticket.get("ticket_id"))
            return jsonify({"error": "The chat request encountered an unexpected error."}), 502

        if warnings:
            return jsonify({"error": warnings[0]}), 502

        return jsonify({"reply": reply, "web_sources": web_sources}), 200

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

    @app.post("/api/kb/suggest")
    @limiter.limit("20 per hour")
    def suggest_kb_article() -> Any:
        """
        Lets an agent manually suggest a KB article for the ticket they're
        working, from the widget — pre-filled client-side from Analyze's
        output, editable before submit. Complements the automatic scan
        (which only covers closed IT-department tickets on a schedule):
        this covers a fix documented before closure, one the automatic
        extraction would've missed (e.g. resolved verbally, never clearly
        stated in the transcript), or outside the scanned department.

        Deliberately public (same trust boundary as /api/analyze, /api/chat
        — rate-limited, not admin-session-gated) since any agent should be
        able to suggest, not just KB reviewers. ALWAYS lands as
        pending_review via match_or_create_article — an agent's
        submission is a draft suggestion, same quality gate as everything
        the auto-scan produces, never auto-approved.
        """
        payload = request.get_json(silent=True) or {}
        ticket_number = str(payload.get("ticket_number") or "").strip()
        title = str(payload.get("title") or "").strip()
        symptoms = str(payload.get("symptoms") or "").strip()
        cause = str(payload.get("cause") or "").strip()
        resolution = str(payload.get("resolution") or "").strip()

        if not (ticket_number and title and resolution):
            return jsonify({"error": "ticket_number, title, and resolution are required."}), 400

        keywords = payload.get("keywords") if isinstance(payload.get("keywords"), list) else []
        related_systems = payload.get("related_systems") if isinstance(payload.get("related_systems"), list) else []

        ai_service = get_ai_service()
        result = kb_service.match_or_create_article(
            ai_service, ticket_number, title, symptoms, cause, resolution, keywords, related_systems
        )
        return jsonify(result), 201

    @app.get("/api/ticket/<ticket_id>/kb-relevant")
    @limiter.limit("30 per hour")
    def get_relevant_kb_articles(ticket_id: str) -> Any:
        """
        Semantic search over this org's own APPROVED KB articles for the
        given query text — used by the widget to feed real KB context
        into Analyze and Chat, in two stages: a preliminary lookup on raw
        ticket text at ticket-view time, and a refined lookup on Analyze's
        own clean `summary` right after analysis completes (see
        loadRelevantKBArticles / refineRelevantKBArticles in app.js — the
        widget concatenates whatever text it wants matched into `subject`
        and `description`; this route doesn't distinguish their origin).

        Costs one embedding call per invocation — the widget calls this
        at most twice per ticket view (once preliminary, once refined),
        not on every keystroke.
        """
        subject = (request.args.get("subject") or "").strip()
        description = (request.args.get("description") or "").strip()
        query_text = f"{subject}\n\n{description}".strip()
        if not query_text:
            return jsonify({"articles": [], "count": 0}), 200

        ai_service = get_ai_service()
        articles = kb_service.find_relevant_articles(ai_service, query_text)
        return jsonify({"articles": articles, "count": len(articles)}), 200

    @app.get("/api/ticket/<ticket_id>/number")
    @limiter.limit("30 per hour")
    def get_ticket_display_number(ticket_id: str) -> Any:
        """
        Resolves Zoho's internal numeric ticket ID to the short display
        number an agent actually recognizes (e.g. "1483") — used by the
        widget's "Suggest for Knowledge Base" submit action, since
        currentTicket.ticket_id in the widget is the internal ID (needed
        for the Zoho API calls this widget already makes throughout), not
        the display number. Fetched lazily at submit time rather than
        eagerly on every ticket view, since it's only needed if the agent
        actually submits a suggestion.
        """
        ticket_number = zoho_client.get_ticket_number(ticket_id)
        return jsonify({"ticket_number": ticket_number}), 200

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

    # ----------------------------------------------------------------
    # Knowledge Base Builder
    # ----------------------------------------------------------------

    @app.get("/api/kb/search")
    def kb_search() -> Any:
        """Public (agent-facing) search over APPROVED articles only — used by the widget's KB tab."""
        query = (request.args.get("q") or "").strip()
        if not query:
            return jsonify({"articles": [], "count": 0}), 200
        articles = kb_service.search_articles(query)
        return jsonify({"articles": articles, "count": len(articles)}), 200

    @app.post("/admin/kb/login")
    @limiter.limit("10 per minute")
    def kb_admin_login() -> Any:
        if not config.kb_admin_users:
            return jsonify({"error": "KB review is not configured on this server."}), 403

        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "")

        expected_password = config.kb_admin_users.get(username)
        # hmac.compare_digest still runs even when the username itself
        # doesn't exist (comparing against an empty string) — avoids
        # leaking "valid username, wrong password" vs. "no such user"
        # via a timing difference.
        valid = expected_password is not None and hmac.compare_digest(password, expected_password)
        if not valid:
            return jsonify({"error": "Invalid username or password."}), 401

        flask_session.clear()
        flask_session["kb_admin_user"] = username
        flask_session["kb_admin_last_activity"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        flask_session.permanent = True
        return jsonify({"user": username, "idle_timeout_minutes": config.kb_admin_idle_timeout_minutes}), 200

    @app.post("/admin/kb/logout")
    def kb_admin_logout() -> Any:
        flask_session.clear()
        return jsonify({"ok": True}), 200

    @app.get("/admin/kb/session")
    def kb_admin_session_check() -> Any:
        """
        Lets the review page ask "am I still logged in?" on load without
        guessing — returns the current user if the session is valid
        (refreshing its idle timer, same as any other authenticated
        call), or 401 if not. Also how the page learns the configured
        idle timeout, so its client-side warning timer matches the
        server's actual enforcement rather than a hardcoded guess.
        """
        username = _validate_kb_session()
        if not username:
            return jsonify({"error": "Not authenticated."}), 401
        return jsonify({"user": username, "idle_timeout_minutes": config.kb_admin_idle_timeout_minutes}), 200

    @app.get("/admin/kb/departments")
    @_kb_session_required
    def kb_list_departments() -> Any:
        """
        Lists Zoho departments so you can find the ID to set as
        ZOHO_IT_DEPARTMENT_ID — one-time setup helper, not used by the
        scan itself at runtime.
        """
        return jsonify({"departments": zoho_client.list_departments()}), 200

    @app.get("/admin/kb/articles")
    @_kb_session_required
    def kb_list_articles() -> Any:
        """
        Paginated, optionally-filtered/searched article listing — the
        review page's single data source for all three tabs (Pending /
        Approved / Rejected) plus its search box. Replaces the old
        separate /admin/kb/pending route: pending is just status=
        pending_review here now, so there's one paginated code path
        instead of two (one of which didn't paginate at all), which
        matters once there are hundreds of rows.
        """
        status_filter = request.args.get("status")
        search_query = (request.args.get("q") or "").strip()
        page = max(1, request.args.get("page", 1, type=int) or 1)
        page_size = min(100, max(1, request.args.get("page_size", 20, type=int) or 20))

        with get_session() as session:
            query = session.query(KBArticle)
            if status_filter in ("pending_review", "approved", "rejected"):
                query = query.filter(KBArticle.status == status_filter)
            if search_query:
                like = f"%{search_query}%"
                query = query.filter(
                    KBArticle.title.ilike(like)
                    | KBArticle.symptoms.ilike(like)
                    | KBArticle.cause.ilike(like)
                    | KBArticle.resolution.ilike(like)
                )

            total = query.count()
            rows = (
                query.order_by(KBArticle.occurrence_count.desc(), KBArticle.last_seen_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )
            return (
                jsonify(
                    {
                        "articles": [r.to_dict() for r in rows],
                        "total": total,
                        "page": page,
                        "page_size": page_size,
                        "pages": max(1, -(-total // page_size)),  # ceiling division
                    }
                ),
                200,
            )

    @app.post("/admin/kb")
    @_kb_session_required
    def kb_create_article() -> Any:
        """
        Manually author a KB article from scratch — for gaps the auto-scan
        won't cover (e.g. proactive documentation, or a fix that predates
        the KB Builder). Goes straight to `approved`: a human deliberately
        wrote this, so routing it through the pending queue for that same
        human (or another reviewer) to re-approve adds a step without
        adding safety. Still gets embedded, so future resolved tickets can
        match against it and reinforce it like any other article.
        """
        payload = request.get_json(silent=True) or {}
        title = str(payload.get("title") or "").strip()
        symptoms = str(payload.get("symptoms") or "").strip()
        cause = str(payload.get("cause") or "").strip()
        resolution = str(payload.get("resolution") or "").strip()
        if not title or not resolution:
            return jsonify({"error": "title and resolution are required."}), 400

        keywords = payload.get("keywords") if isinstance(payload.get("keywords"), list) else []
        related_systems = payload.get("related_systems") if isinstance(payload.get("related_systems"), list) else []

        ai_service = get_ai_service()
        embed_input = kb_service._embedding_text(symptoms, cause, resolution)
        embedding = ai_service.embed_text(embed_input, task_type="SEMANTIC_SIMILARITY")
        retrieval_embedding = ai_service.embed_text(embed_input, task_type="RETRIEVAL_DOCUMENT")

        now = _dt.datetime.now(_dt.timezone.utc)
        with get_session() as session:
            article = KBArticle(
                title=title,
                symptoms=symptoms,
                cause=cause,
                resolution=resolution,
                keywords_json=_to_json_string(keywords),
                related_systems_json=_to_json_string(related_systems),
                embedding_json=_to_json_string(embedding or []),
                retrieval_embedding_json=_to_json_string(retrieval_embedding or []),
                status="approved",
                occurrence_count=1,
                source_ticket_ids_json=_to_json_string([]),
                first_ticket_id="",
                reviewed_by=flask_session.get("kb_admin_user"),
                reviewed_at=now,
                created_at=now,
                last_seen_at=now,
            )
            session.add(article)
            session.flush()
            return jsonify({"article": article.to_dict()}), 201

    @app.patch("/admin/kb/<int:article_id>")
    @_kb_session_required
    def kb_update_article(article_id: int) -> Any:
        """
        Edit an existing article's content — unlike /approve, this works
        regardless of current status (pending, approved, or rejected), and
        does NOT change status. Re-embeds if symptoms/cause/resolution
        changed, so the similarity match used for future reinforcement
        stays accurate to the edited text rather than the original
        AI-generated draft.
        """
        payload = request.get_json(silent=True) or {}
        with get_session() as session:
            article = session.get(KBArticle, article_id)
            if article is None:
                return jsonify({"error": "Article not found."}), 404

            content_changed = False
            if "title" in payload:
                article.title = str(payload["title"]).strip() or article.title
            if "symptoms" in payload:
                article.symptoms = str(payload["symptoms"]).strip()
                content_changed = True
            if "cause" in payload:
                article.cause = str(payload["cause"]).strip()
                content_changed = True
            if "resolution" in payload:
                article.resolution = str(payload["resolution"]).strip()
                content_changed = True
            if "keywords" in payload and isinstance(payload["keywords"], list):
                article.keywords_json = _to_json_string(payload["keywords"])
            if "related_systems" in payload and isinstance(payload["related_systems"], list):
                article.related_systems_json = _to_json_string(payload["related_systems"])

            if content_changed:
                ai_service = get_ai_service()
                embed_input = kb_service._embedding_text(article.symptoms, article.cause, article.resolution)
                new_embedding = ai_service.embed_text(embed_input, task_type="SEMANTIC_SIMILARITY")
                if new_embedding:
                    article.embedding_json = _to_json_string(new_embedding)
                new_retrieval_embedding = ai_service.embed_text(embed_input, task_type="RETRIEVAL_DOCUMENT")
                if new_retrieval_embedding:
                    article.retrieval_embedding_json = _to_json_string(new_retrieval_embedding)

            session.flush()
            return jsonify({"article": article.to_dict()}), 200

    @app.delete("/admin/kb/<int:article_id>")
    @_kb_session_required
    def kb_delete_article(article_id: int) -> Any:
        with get_session() as session:
            article = session.get(KBArticle, article_id)
            if article is None:
                return jsonify({"error": "Article not found."}), 404
            session.delete(article)
            return jsonify({"deleted": article_id}), 200

    @app.post("/admin/kb/<int:article_id>/resync-ticket-numbers")
    @_kb_session_required
    def kb_resync_ticket_numbers(article_id: int) -> Any:
        """
        Re-resolve an article's source ticket references from Zoho's
        internal IDs to the display ticket numbers agents actually see —
        for articles created before that mapping was fixed in the scan
        pipeline (see kb_service.process_resolved_ticket). Safe to run
        more than once.
        """
        with get_session() as session:
            article = session.get(KBArticle, article_id)
            if article is None:
                return jsonify({"error": "Article not found."}), 404
            resolved_count = kb_service.resync_ticket_numbers(article)
            session.flush()
            return jsonify({"article": article.to_dict(), "resolved_count": resolved_count}), 200

    @app.get("/admin/kb/<int:article_id>")
    @_kb_session_required
    def kb_get_article(article_id: int) -> Any:
        with get_session() as session:
            article = session.get(KBArticle, article_id)
            if article is None:
                return jsonify({"error": "Article not found."}), 404
            return jsonify({"article": article.to_dict()}), 200

    @app.post("/admin/kb/<int:article_id>/approve")
    @_kb_session_required
    def kb_approve_article(article_id: int) -> Any:
        """
        Approve a draft, optionally with reviewer edits to the extracted
        fields (title/symptoms/cause/resolution/keywords/related_systems)
        sent in the request body — the reviewer is the last line of
        quality control before something becomes searchable.
        """
        payload = request.get_json(silent=True) or {}
        with get_session() as session:
            article = session.get(KBArticle, article_id)
            if article is None:
                return jsonify({"error": "Article not found."}), 404
            if article.status != "pending_review":
                return jsonify({"error": f"Article is already '{article.status}', not pending review."}), 400

            if "title" in payload:
                article.title = str(payload["title"]).strip() or article.title
            if "symptoms" in payload:
                article.symptoms = str(payload["symptoms"]).strip()
            if "cause" in payload:
                article.cause = str(payload["cause"]).strip()
            if "resolution" in payload:
                article.resolution = str(payload["resolution"]).strip()
            if "keywords" in payload and isinstance(payload["keywords"], list):
                article.keywords_json = _to_json_string(payload["keywords"])
            if "related_systems" in payload and isinstance(payload["related_systems"], list):
                article.related_systems_json = _to_json_string(payload["related_systems"])

            article.status = "approved"
            article.reviewed_by = flask_session.get("kb_admin_user")
            article.reviewed_at = _dt.datetime.now(_dt.timezone.utc)
            session.flush()
            return jsonify({"article": article.to_dict()}), 200

    @app.post("/admin/kb/<int:article_id>/reject")
    @_kb_session_required
    def kb_reject_article(article_id: int) -> Any:
        with get_session() as session:
            article = session.get(KBArticle, article_id)
            if article is None:
                return jsonify({"error": "Article not found."}), 404
            article.status = "rejected"
            article.reviewed_by = flask_session.get("kb_admin_user")
            article.reviewed_at = _dt.datetime.now(_dt.timezone.utc)
            session.flush()
            return jsonify({"article": article.to_dict()}), 200

    @app.post("/internal/kb/scan")
    @limiter.exempt
    def kb_cron_trigger_scan() -> Any:
        """
        Scan trigger for an EXTERNAL scheduler (e.g. the GitHub Actions
        workflow), not a logged-in reviewer — deliberately separate from
        /admin/kb/scan (session-gated) since a cron job can't hold a
        login session. Auth is a shared secret instead, checked via
        header, since a URL query-string token would land in Render's
        access logs in plaintext.

        This exists because Render's free tier can't run a reliable
        in-process background scheduler: the same 15-minute spin-down
        that idles HTTP traffic also kills any in-process thread,
        APScheduler included — so KB_SCAN_ENABLED's in-process scheduler
        silently stops running whenever the process has been idle. An
        external trigger doesn't depend on the web process having stayed
        alive continuously.
        """
        if not config.kb_scan_cron_secret:
            return jsonify({"error": "KB_SCAN_CRON_SECRET is not configured on this server."}), 403
        provided = request.headers.get("X-Scan-Secret", "")
        if not hmac.compare_digest(provided, config.kb_scan_cron_secret):
            return jsonify({"error": "Invalid or missing scan secret."}), 401

        result = kb_service.run_scan_pass(get_ai_service())
        return jsonify(
            {
                "tickets_seen": result.tickets_seen,
                "articles_created": result.articles_created,
                "articles_reinforced": result.articles_reinforced,
                "tickets_skipped_not_extractable": result.tickets_skipped_not_extractable,
                "tickets_skipped_error": result.tickets_skipped_error,
            }
        ), 200

    @app.post("/admin/kb/scan")
    @_kb_session_required
    def kb_trigger_scan() -> Any:
        """Manual scan trigger — useful for testing without waiting for the scheduler interval."""
        result = kb_service.run_scan_pass(get_ai_service())
        return jsonify(
            {
                "tickets_seen": result.tickets_seen,
                "articles_created": result.articles_created,
                "articles_reinforced": result.articles_reinforced,
                "tickets_skipped_not_extractable": result.tickets_skipped_not_extractable,
                "tickets_skipped_error": result.tickets_skipped_error,
            }
        ), 200

    @app.get("/admin/kb-review")
    def kb_review_page() -> Any:
        """Serves the standalone review-queue page shell. Auth is enforced by the /admin/kb/* API calls it makes (session-based), not this static shell."""
        if not config.kb_admin_users:
            return jsonify({"error": "KB review is not configured on this server."}), 403
        from flask import send_from_directory

        return send_from_directory(Path(__file__).resolve().parent / "admin", "kb_review.html")


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


def _to_json_string(data: Any) -> str:
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
