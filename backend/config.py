"""
config.py
---------
Centralized application configuration.

All configuration is loaded from environment variables (via a .env file in
development). No secrets are ever hardcoded here. This module is the single
source of truth for configuration values used across the application.

Future providers / features (OpenAI, Azure OpenAI, webhooks, AD integration,
etc.) should add their configuration here rather than scattering `os.getenv`
calls throughout the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a .env file if present. This is a no-op
# in production environments where real environment variables are injected
# by the hosting platform.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _get_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean-ish environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    """Immutable application configuration, populated from the environment."""

    # --- Flask ---
    flask_env: str = field(default_factory=lambda: os.getenv("FLASK_ENV", "development"))
    debug: bool = field(default_factory=lambda: _get_bool("FLASK_DEBUG", default=False))
    secret_key: str = field(default_factory=lambda: os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "5000")))

    # --- CORS ---
    # Zoho Desk widgets are served from Zoho's own domains; restrict in
    # production via CORS_ALLOWED_ORIGINS (comma separated).
    cors_allowed_origins: str = field(default_factory=lambda: os.getenv("CORS_ALLOWED_ORIGINS", "*"))

    # --- Database ---
    # Defaults to a local SQLite file — fine for local dev, but NOT
    # durable on most free-tier hosts (e.g. Render's free web services
    # have an ephemeral filesystem; a local SQLite file is wiped on every
    # redeploy/restart/spin-down). For a persistent deployment, point
    # this at a hosted Postgres instance instead — e.g. Neon
    # (https://neon.tech) has a genuinely free tier with no forced
    # expiry. Copy the connection string Neon gives you directly into
    # DATABASE_URL; see database.py for one small defensive fixup this
    # app applies to that URL (postgres:// -> postgresql://).
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'service_desk_copilot.db'}")
    )

    # --- AI Provider (Gemini today, swappable later) ---
    ai_provider: str = field(default_factory=lambda: os.getenv("AI_PROVIDER", "gemini"))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.6-flash"))
    ai_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("AI_TIMEOUT_SECONDS", "30")))

    # --- Developer Mode ---
    # Global server-side kill switch. Even if the widget requests developer
    # diagnostics, the backend will withhold them unless this is enabled.
    developer_mode_allowed: bool = field(default_factory=lambda: _get_bool("DEVELOPER_MODE_ALLOWED", default=True))

    # --- Zoho Desk OAuth (server-side, for fetching thread/email content) ---
    # Many tickets — especially email-originated ones — have a blank
    # top-level description; the actual message lives in the ticket's
    # thread stream. These credentials let the backend fetch that content
    # directly via Zoho's Desk REST API, via a Self Client (see README).
    zoho_client_id: str = field(default_factory=lambda: os.getenv("ZOHO_CLIENT_ID", ""))
    zoho_client_secret: str = field(default_factory=lambda: os.getenv("ZOHO_CLIENT_SECRET", ""))
    zoho_refresh_token: str = field(default_factory=lambda: os.getenv("ZOHO_REFRESH_TOKEN", ""))
    zoho_org_id: str = field(default_factory=lambda: os.getenv("ZOHO_ORG_ID", ""))
    # Restricts the KB Builder's ticket scan to a single department (e.g.
    # IT). Leave blank to scan all departments. This is the single
    # biggest lever on AI token spend for the scan: every closed ticket
    # scanned costs one extraction call plus one embedding call
    # regardless of whether it's ever going to become a useful IT KB
    # entry — scoping out other departments (HR, Finance, etc.) up front
    # means we're never paying to extract "KB articles" that don't belong
    # in this KB anyway, not just saving tokens for their own sake.
    # GET /admin/kb/departments (once deployed) lists department IDs so
    # you can find the right value to set here.
    zoho_it_department_id: str = field(default_factory=lambda: os.getenv("ZOHO_IT_DEPARTMENT_ID", ""))
    zoho_accounts_domain: str = field(default_factory=lambda: os.getenv("ZOHO_ACCOUNTS_DOMAIN", "https://accounts.zoho.com"))
    zoho_api_domain: str = field(default_factory=lambda: os.getenv("ZOHO_API_DOMAIN", "https://desk.zoho.com"))

    # --- Ticket Categories ---
    # Comma-separated lists matching your Zoho Desk org's actual Category
    # and Sub-Category field values (Setup > Customization > Fields).
    # Constrains the AI to pick from real values instead of inventing its
    # own. Leave blank to let the AI suggest freely.
    ticket_categories: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            c.strip() for c in os.getenv("TICKET_CATEGORIES", "").split(",") if c.strip()
        )
    )
    ticket_sub_categories: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            c.strip() for c in os.getenv("TICKET_SUB_CATEGORIES", "").split(",") if c.strip()
        )
    )

    # --- Selectable models ---
    # Comma-separated allowlist the widget's model dropdown may request at
    # runtime. Requests for anything outside this list fall back to
    # gemini_model above — prevents arbitrary/typo'd model strings from
    # reaching the API.
    allowed_gemini_models: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            m.strip()
            for m in os.getenv(
                "ALLOWED_GEMINI_MODELS", "gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite"
            ).split(",")
            if m.strip()
        )
    )

    # --- Related Tickets (Location + Category match) ---
    # API name of the custom field storing branch/location (Setup >
    # Customization > Fields > Tickets > Location > API Name).
    zoho_location_field: str = field(default_factory=lambda: os.getenv("ZOHO_LOCATION_FIELD", "cf_location"))
    # How far back to look when finding related tickets at the same
    # location with the same category.
    related_tickets_months: int = field(default_factory=lambda: int(os.getenv("RELATED_TICKETS_MONTHS", "3")))

    # --- Knowledge Base Builder ---
    # Embedding model used to compare a newly-resolved ticket's extracted
    # summary against existing KB articles for the dedup/reinforcement
    # check (see kb_service.py). Separate from gemini_model since
    # embedding models are a distinct model family, not a chat model.
    gemini_embedding_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    )
    # Cosine similarity (0-1) above which a new ticket is folded into an
    # existing article (occurrence_count bumped) instead of creating a new
    # draft. Embeds symptoms+cause+resolution together (see kb_service.py)
    # so this threshold is comparing causal shape, not just symptom text —
    # tune by hand against real duplicates in your data before relying on it.
    kb_similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("KB_SIMILARITY_THRESHOLD", "0.85"))
    )
    # Above this similarity, a reinforcement is treated as close enough to
    # certainly-the-same-issue that the extra "does this add new detail?"
    # AI call (see kb_service.process_resolved_ticket) is skipped entirely
    # — just bump occurrence_count. That check exists to catch genuinely
    # borderline matches with a new variant worth folding in; a
    # near-identical match doesn't need a second AI call to confirm that.
    # Must be >= kb_similarity_threshold to make sense.
    kb_merge_check_skip_threshold: float = field(
        default_factory=lambda: float(os.getenv("KB_MERGE_CHECK_SKIP_THRESHOLD", "0.97"))
    )
    # Similarity bar for surfacing a KB article to a technician as
    # "relevant to this ticket" during Analyze/Chat — deliberately lower
    # than kb_similarity_threshold (0.85), which asks "is this literally
    # the same recurring issue." This one asks "would a technician find
    # this worth glancing at," a lower bar by design. 0.65 rather than a
    # more intuitive-sounding 0.5: Gemini's embedding models commonly
    # score even unrelated text pairs at 0.5+, so 0.5 let weak/unrelated
    # matches through regardless of correct task_type usage.
    kb_relevance_threshold: float = field(
        default_factory=lambda: float(os.getenv("KB_RELEVANCE_THRESHOLD", "0.65"))
    )
    # Above this, an embedding match is trusted alone (a genuine paraphrase
    # with little/no literal word overlap) — below it but still above
    # kb_relevance_threshold, a match ALSO needs at least one shared
    # substantive word with the query (see
    # kb_service.find_relevant_articles) to guard against short tickets
    # matching topically-unrelated articles on embedding similarity alone.
    kb_relevance_high_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("KB_RELEVANCE_HIGH_CONFIDENCE_THRESHOLD", "0.8"))
    )
    # Max KB articles surfaced per ticket for Analyze/Chat context — kept
    # small deliberately: every article included costs extraction/chat
    # prompt tokens, and a technician scanning results benefits from the
    # 2-3 best matches, not an exhaustive list.
    kb_relevant_articles_limit: int = field(
        default_factory=lambda: int(os.getenv("KB_RELEVANT_ARTICLES_LIMIT", "3"))
    )
    # Enables Gemini's built-in Google Search grounding for the Chat
    # feature only (NOT Analyze — Gemini's API rejects combining Search
    # grounding with structured JSON output, a real, current API
    # constraint, not a choice made here). The model decides per-message
    # whether a search is actually warranted; when it uses one, real
    # source URLs are returned and surfaced to the technician. This is a
    # separate, metered Google Search cost outside normal token pricing —
    # leave this off if that cost isn't wanted.
    kb_enable_web_grounding_in_chat: bool = field(
        default_factory=lambda: _get_bool("ENABLE_WEB_GROUNDING_IN_CHAT", default=True)
    )
    # Master on/off switch for the background scan. Off by default so a
    # fresh deploy doesn't start scanning/spending AI quota until someone
    # deliberately enables it.
    kb_scan_enabled: bool = field(default_factory=lambda: _get_bool("KB_SCAN_ENABLED", default=False))
    kb_scan_interval_minutes: int = field(
        default_factory=lambda: int(os.getenv("KB_SCAN_INTERVAL_MINUTES", "30"))
    )
    # How many resolved/closed tickets to pull per scan pass — a safety
    # ceiling, not a typical count.
    kb_scan_batch_size: int = field(default_factory=lambda: int(os.getenv("KB_SCAN_BATCH_SIZE", "50")))
    # Basic auth for the standalone /admin/kb-review page — this is an
    # internal reviewer tool, not agent- or customer-facing, and isn't
    # covered by Zoho's own auth since it's served outside the widget
    # iframe. Leave blank to disable the page entirely (safer default than
    # an unauthenticated internal tool).
    #
    # Supports multiple named reviewers via KB_ADMIN_USERS
    # ("alice:pw1,bob:pw2") so reviewed_by reflects real distinct people,
    # not one shared "admin" string. KB_ADMIN_USERNAME/PASSWORD (single
    # account) still work and are folded in automatically for anyone
    # already using them — no breaking change on upgrade.
    #
    # Passwords are plaintext in env, same trust model as the single
    # account this replaces (an internal tool, gated at the deployment
    # level) — checked with a constant-time comparison to avoid timing
    # attacks, but not hashed at rest. If that trade-off stops being
    # acceptable, this is the place to swap in werkzeug's
    # generate_password_hash/check_password_hash.
    kb_admin_username: str = field(default_factory=lambda: os.getenv("KB_ADMIN_USERNAME", ""))
    kb_admin_password: str = field(default_factory=lambda: os.getenv("KB_ADMIN_PASSWORD", ""))
    kb_admin_users_raw: str = field(default_factory=lambda: os.getenv("KB_ADMIN_USERS", ""))
    # Sliding idle timeout — a reviewer inactive this long is logged out
    # automatically and must log back in. Refreshed on every authenticated
    # request (see app._validate_kb_session).
    kb_admin_idle_timeout_minutes: int = field(
        default_factory=lambda: int(os.getenv("KB_ADMIN_IDLE_TIMEOUT_MINUTES", "20"))
    )
    # Hard ceiling regardless of activity — caps how long a single login
    # can last even if the reviewer never goes idle, so a stolen/forgotten
    # session cookie doesn't stay valid indefinitely.
    kb_admin_session_lifetime_hours: int = field(
        default_factory=lambda: int(os.getenv("KB_ADMIN_SESSION_LIFETIME_HOURS", "8"))
    )

    @property
    def kb_admin_users(self) -> dict[str, str]:
        """Effective {username: password} map, merging KB_ADMIN_USERS with the legacy single-account vars."""
        users: dict[str, str] = {}
        if self.kb_admin_username and self.kb_admin_password:
            users[self.kb_admin_username] = self.kb_admin_password
        for pair in self.kb_admin_users_raw.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            name, _, pw = pair.partition(":")
            name = name.strip()
            if name and pw:
                users[name] = pw
        return users

    # --- Logging ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def is_production(self) -> bool:
        return self.flask_env.lower() == "production"


config = Config()