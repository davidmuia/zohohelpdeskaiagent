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

    # --- Logging ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def is_production(self) -> bool:
        return self.flask_env.lower() == "production"


config = Config()