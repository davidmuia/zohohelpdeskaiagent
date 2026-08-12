"""
zoho_client.py
---------------
Minimal server-side client for Zoho Desk's REST API, used ONLY to fetch a
ticket's latest thread content (the actual message body — including email
content — which the widget SDK's `ZOHODESK.get('ticket')` does not expose).

This is intentionally separate from ai_service.py: it's a Zoho-specific data
source, not an AI provider. Uses the standard OAuth "Self Client" refresh
token flow (see README for setup), which is a stable, well-documented Zoho
mechanism — independent of the extension widget SDK's own auth system.

Nothing here writes to Zoho Desk; this module only performs GET requests.
"""

from __future__ import annotations

import datetime as _dt
import html as html_module
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import requests

from config import config

logger = logging.getLogger(__name__)

# Cache the access token in memory between requests; Zoho access tokens are
# short-lived (~1 hour) but the refresh token is long-lived, so we only need
# to re-authenticate occasionally, not on every request.
_token_cache: dict[str, Any] = {"access_token": None, "expires_at": 0.0}


class ZohoAuthError(Exception):
    """Raised when Zoho OAuth credentials are missing or a token request fails."""


class ZohoAPIError(Exception):
    """Raised when a Zoho Desk API call itself fails."""


def _get_access_token() -> str:
    """Return a valid access token, refreshing it if expired or absent."""
    now = time.monotonic()
    if _token_cache["access_token"] and now < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    if not (config.zoho_client_id and config.zoho_client_secret and config.zoho_refresh_token):
        raise ZohoAuthError(
            "Zoho OAuth credentials are not configured (ZOHO_CLIENT_ID / "
            "ZOHO_CLIENT_SECRET / ZOHO_REFRESH_TOKEN)."
        )

    try:
        response = requests.post(
            f"{config.zoho_accounts_domain}/oauth/v2/token",
            data={
                "refresh_token": config.zoho_refresh_token,
                "client_id": config.zoho_client_id,
                "client_secret": config.zoho_client_secret,
                "grant_type": "refresh_token",
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to refresh Zoho access token")
        raise ZohoAuthError(f"Could not refresh Zoho access token: {exc}") from exc

    access_token = payload.get("access_token")
    if not access_token:
        raise ZohoAuthError(f"Zoho token response missing access_token: {payload}")

    # Refresh a little early (60s buffer) rather than cutting it exactly at expiry.
    expires_in = int(payload.get("expires_in", 3600))
    _token_cache["access_token"] = access_token
    _token_cache["expires_at"] = now + max(expires_in - 60, 60)

    return access_token


def _html_to_text(raw_html: str) -> str:
    """
    Strip HTML tags and unescape entities, collapsing whitespace. Good
    enough for feeding email-derived content to the AI prompt and for
    display — not intended as a general-purpose HTML sanitizer.
    """
    if not raw_html:
        return ""
    # Turn common block-level breaks into newlines before stripping tags,
    # so paragraphs don't run together into one wall of text.
    text = re.sub(r"(?i)<(br|/p|/div|/li)\s*/?>", "\n", raw_html)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_module.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_requester_ticket_history(
    email: str, exclude_ticket_id: Optional[str] = None, limit: int = 5
) -> list[dict[str, Any]]:
    """
    Fetch a requester's recent ticket history by email — a deterministic
    lookup (no AI involved). Used to surface recurring-issue context: e.g.
    "this requester has filed 3 tickets in the last 30 days."

    Uses /api/v1/tickets/search?email=... (confirmed working endpoint, as
    opposed to the more commonly guessed /api/v1/tickets?email=...).
    Results are sorted defensively in Python by createdTime rather than
    trusting API ordering, same approach as fetch_latest_thread_content.

    Returns an empty list (rather than raising) if credentials aren't
    configured or the lookup fails, so callers can render "no history
    available" gracefully instead of breaking the ticket view.
    """
    if not (config.zoho_client_id and config.zoho_client_secret and config.zoho_refresh_token):
        logger.info("Zoho OAuth not configured — skipping requester history lookup.")
        return []
    if not email:
        return []

    try:
        access_token = _get_access_token()
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        if config.zoho_org_id:
            headers["orgId"] = config.zoho_org_id

        response = requests.get(
            f"{config.zoho_api_domain}/api/v1/tickets/search",
            headers=headers,
            params={"email": email, "limit": 50},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except ZohoAuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch requester history for email=%s: %s", email, exc)
        return []

    tickets = payload.get("data") or []
    tickets.sort(key=lambda t: t.get("createdTime") or "", reverse=True)

    results = []
    for t in tickets:
        ticket_id = str(t.get("id") or "")
        if exclude_ticket_id and ticket_id == str(exclude_ticket_id):
            continue
        results.append(
            {
                "ticket_id": ticket_id,
                "ticket_number": t.get("ticketNumber") or "",
                "subject": t.get("subject") or "",
                "status": t.get("status") or "",
                "created_time": t.get("createdTime") or "",
            }
        )
        if len(results) >= limit:
            break

    return results


def get_ticket_details(ticket_id: str) -> dict[str, Any]:
    """
    Fetch a ticket's own category, sub-category, and location (custom
    field) directly from the full REST ticket object — these aren't
    reliably present in the lighter widget SDK's ZOHODESK.get('ticket')
    call (we hit this same gap with `description` earlier), so the backend
    fetches them directly instead of trusting the widget to supply them.

    Returns a dict with possibly-empty string values rather than raising,
    so callers can degrade gracefully (e.g. show "Related Tickets"
    unavailable) instead of breaking the ticket view.
    """
    empty = {"category": "", "sub_category": "", "location": ""}
    if not (config.zoho_client_id and config.zoho_client_secret and config.zoho_refresh_token):
        return empty

    try:
        access_token = _get_access_token()
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        if config.zoho_org_id:
            headers["orgId"] = config.zoho_org_id

        response = requests.get(
            f"{config.zoho_api_domain}/api/v1/tickets/{ticket_id}",
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except ZohoAuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch ticket details for ticket_id=%s: %s", ticket_id, exc)
        return empty

    location = ""
    cf = payload.get("cf") or {}
    if config.zoho_location_field in cf:
        location = cf.get(config.zoho_location_field) or ""
    elif config.zoho_location_field in payload:
        # Defensive fallback in case custom fields aren't nested under "cf"
        # for this API version/plan.
        location = payload.get(config.zoho_location_field) or ""
    else:
        logger.info(
            "Location field '%s' not found on ticket_id=%s (checked payload['cf'] and "
            "top-level payload) — Related Tickets will be unavailable for this ticket.",
            config.zoho_location_field,
            ticket_id,
        )

    return {
        "category": payload.get("category") or "",
        "sub_category": payload.get("subCategory") or "",
        "location": location,
    }


def get_related_tickets(
    location: str, sub_category: str, exclude_ticket_id: Optional[str] = None, limit: int = 10
) -> list[dict[str, Any]]:
    """
    Find tickets at the same Location with the same Sub-Category, created
    within the last `config.related_tickets_months` months — deterministic,
    identity-independent.

    Two-stage lookup: the bulk "List Tickets" endpoint returns
    `category`/`subCategory` directly, but never returns `cf` (custom
    fields, where Location lives) — that field only appears on the
    single-ticket detail endpoint. So: Stage 1 filters the ticket list by
    Sub-Category + date window down to a candidate set; Stage 2 fetches
    each candidate's full detail to check Location.

    Both stages run their independent HTTP calls CONCURRENTLY via a thread
    pool — sequential fetching here was the main cause of 1-2 minute
    response times at this org's ticket volume (~1200/month, meaning ~36
    pages just to cover a 3-month window, plus up to dozens of per-ticket
    detail lookups). Concurrency doesn't reduce the total number of API
    calls, just how long they take in wall-clock time.
    """
    if not (location and sub_category):
        return []
    if not (config.zoho_client_id and config.zoho_client_secret and config.zoho_refresh_token):
        return []

    # Raised from 60 now that Stage 2 runs concurrently and is cheap at
    # scale — reduces the risk of real matches falling outside the
    # candidate window at this org's ticket volume.
    MAX_CANDIDATES = 150
    PAGE_BATCH_SIZE = 5  # pages fetched concurrently per round
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=30 * config.related_tickets_months)

    try:
        access_token = _get_access_token()
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        if config.zoho_org_id:
            headers["orgId"] = config.zoho_org_id
    except ZohoAuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to authenticate for related-tickets lookup: %s", exc)
        return []

    def _fetch_page(from_index: int) -> list[dict[str, Any]]:
        try:
            response = requests.get(
                f"{config.zoho_api_domain}/api/v1/tickets",
                headers=headers,
                params={"from": from_index, "limit": 100, "sortBy": "-createdTime"},
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("data") or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch ticket page (from=%s) for related-tickets lookup: %s", from_index, exc)
            return []

    # --- Stage 1: filter by Sub-Category + date window (list endpoint) ---
    # Fetch pages in concurrent batches of PAGE_BATCH_SIZE. Within each
    # completed batch, stop as soon as we find a ticket older than cutoff
    # (results are sorted newest-first, so once one page in the batch runs
    # past the cutoff, later pages will too).
    candidates: list[dict[str, Any]] = []
    max_pages = 40  # safety ceiling — see original sizing rationale below
    # At ~1200 tickets/month, a 3-month window is ~3,600 tickets; 100 * 40
    # covers that with margin. This is a safety ceiling, not the typical
    # stopping point — the cutoff check below usually stops things sooner.

    page_starts = [i * 100 for i in range(max_pages)]
    stop = False
    for batch_start in range(0, len(page_starts), PAGE_BATCH_SIZE):
        if stop or len(candidates) >= MAX_CANDIDATES:
            break
        batch = page_starts[batch_start : batch_start + PAGE_BATCH_SIZE]

        with ThreadPoolExecutor(max_workers=PAGE_BATCH_SIZE) as executor:
            batch_results = list(executor.map(_fetch_page, batch))

        for page_tickets in batch_results:
            if not page_tickets:
                stop = True
                continue

            for t in page_tickets:
                created_time_str = t.get("createdTime") or ""
                try:
                    created_time = _dt.datetime.fromisoformat(created_time_str.replace("Z", "+00:00"))
                except ValueError:
                    continue

                if created_time < cutoff:
                    stop = True
                    break

                ticket_id = str(t.get("id") or "")
                if exclude_ticket_id and ticket_id == str(exclude_ticket_id):
                    continue

                if (t.get("subCategory") or "") == sub_category:
                    candidates.append(
                        {
                            "ticket_id": ticket_id,
                            "ticket_number": t.get("ticketNumber") or "",
                            "subject": t.get("subject") or "",
                            "status": t.get("status") or "",
                            "created_time": created_time_str,
                        }
                    )
                    if len(candidates) >= MAX_CANDIDATES:
                        break

            if len(candidates) >= MAX_CANDIDATES:
                break

    # --- Stage 2: check Location on each candidate, concurrently ---
    def _check_candidate(candidate: dict[str, Any]) -> Optional[dict[str, Any]]:
        details = get_ticket_details(candidate["ticket_id"])
        return candidate if details["location"] == location else None

    results: list[dict[str, Any]] = []
    if candidates:
        with ThreadPoolExecutor(max_workers=15) as executor:
            for match in executor.map(_check_candidate, candidates):
                if match:
                    results.append(match)

    # Candidates were collected newest-first within each page, but thread
    # pool completion order doesn't preserve that — re-sort before capping
    # to `limit` so we keep the most recent matches, not an arbitrary subset.
    results.sort(key=lambda r: r.get("created_time") or "", reverse=True)
    return results[:limit]


def get_agent_resolution_context(ticket_id: str, max_messages: int = 3, max_chars: int = 900) -> Optional[str]:
    """
    Fetch the last few agent-authored messages on a ticket (not just the
    single most recent one) — used to give chat/analysis on OTHER tickets a
    sense of "what was done" to resolve them.

    Deliberately captures multiple recent messages rather than only the
    last: a ticket's final agent message is very often a canned closing
    note ("Your ticket has been resolved, thank you!"), while the actual
    resolution detail ("replaced the toner cartridge, tested printing")
    typically sits one or two messages earlier. Rather than trying to
    detect canned responses with a fragile keyword heuristic (which would
    be brittle and specific to this org's exact templates), this hands the
    LLM a short window of recent agent messages and lets it judge which
    part is the substantive resolution — that's a better fit for an LLM's
    judgment than ours.

    No extra API cost versus the old single-message version: the full
    thread list is fetched either way in one call: we're just keeping more
    of what's already there instead of discarding all but the last entry.

    Returns None if no agent messages are found or the fetch fails.
    """
    if not (config.zoho_client_id and config.zoho_client_secret and config.zoho_refresh_token):
        return None

    try:
        access_token = _get_access_token()
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        if config.zoho_org_id:
            headers["orgId"] = config.zoho_org_id

        response = requests.get(
            f"{config.zoho_api_domain}/api/v1/tickets/{ticket_id}/threads",
            headers=headers,
            params={"limit": 100},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch threads for agent resolution context, ticket_id=%s: %s", ticket_id, exc)
        return None

    threads = payload.get("data") or []
    agent_threads = [t for t in threads if t.get("direction") != "in"]
    if not agent_threads:
        return None

    # Chronological order, then take the last `max_messages`.
    agent_threads.sort(key=lambda t: t.get("createdTime") or "")
    recent = agent_threads[-max_messages:]

    cleaned_messages = []
    for t in recent:
        raw = t.get("plainText") or t.get("content") or t.get("summary") or ""
        cleaned = _html_to_text(raw)
        if cleaned:
            cleaned_messages.append(cleaned)

    if not cleaned_messages:
        return None

    combined = " | ".join(cleaned_messages)
    return combined[:max_chars]


def fetch_ticket_conversation(ticket_id: str, max_chars: int = 4000) -> Optional[str]:
    """
    Fetch the FULL conversation for a ticket — every thread, in
    chronological order, each labeled [Customer] or [Agent] — rather than
    just the single latest message. The earlier version of this function
    only pulled the most recent incoming thread, which meant a customer's
    ORIGINAL message could be silently dropped once they'd sent a follow-up
    (e.g. an initial "printer not working" email, then two days later "also
    the network drive is down" — only the second would reach the AI). This
    version preserves the full narrative while still keeping the
    customer/agent attribution fix from before (never lets an agent's
    words be mistaken for the customer's).

    If the combined transcript exceeds `max_chars`, keeps the FIRST message
    (the original issue) in full, plus as many of the most recent messages
    as fit in the remaining budget, with a clear omission marker between —
    preserving both "how this started" and "where it stands now" rather
    than truncating from one end only.

    Returns None (rather than raising) if credentials aren't configured or
    no threads are found, so callers can gracefully fall back to whatever
    description the widget already sent.
    """
    if not (config.zoho_client_id and config.zoho_client_secret and config.zoho_refresh_token):
        logger.info("Zoho OAuth not configured — skipping conversation fetch.")
        return None

    try:
        access_token = _get_access_token()
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        if config.zoho_org_id:
            headers["orgId"] = config.zoho_org_id

        response = requests.get(
            f"{config.zoho_api_domain}/api/v1/tickets/{ticket_id}/threads",
            headers=headers,
            params={"limit": 100},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except ZohoAuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch threads for ticket_id=%s: %s", ticket_id, exc)
        return None

    threads = payload.get("data") or []
    if not threads:
        return None

    # Chronological order (oldest first) — defensive sort rather than
    # trusting API ordering, same approach used elsewhere in this module.
    threads.sort(key=lambda t: t.get("createdTime") or "")

    entries: list[str] = []
    for t in threads:
        raw = t.get("plainText") or t.get("content") or t.get("summary") or ""
        cleaned = _html_to_text(raw)
        if not cleaned:
            continue

        is_agent = t.get("direction") != "in"
        author = t.get("author") or {}
        sender_name = author.get("name") or t.get("fromEmailAddress") or ""

        if is_agent:
            label = f"Agent — {sender_name}" if sender_name else "Agent"
        else:
            label = f"Customer — {sender_name}" if sender_name else "Customer"

        entries.append(f"[{label}] {cleaned}")

    if not entries:
        return None

    full_transcript = "\n\n".join(entries)
    if len(full_transcript) <= max_chars:
        return full_transcript

    # Too long — keep the first message (original issue) in full, then fill
    # remaining budget with the most recent messages, newest-relevant-first
    # from the end, re-assembled back into chronological order.
    first_entry = entries[0]
    omission_marker = "\n\n[... earlier messages omitted for length ...]\n\n"
    budget = max_chars - len(first_entry) - len(omission_marker)

    tail_entries: list[str] = []
    used = 0
    for entry in reversed(entries[1:]):
        if used + len(entry) + 2 > budget:
            break
        tail_entries.insert(0, entry)
        used += len(entry) + 2

    if not tail_entries:
        # Even one recent message doesn't fit — just hard-truncate the first.
        return first_entry[:max_chars]

    return first_entry + omission_marker + "\n\n".join(tail_entries)