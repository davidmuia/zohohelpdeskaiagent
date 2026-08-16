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


def list_departments() -> list[dict[str, Any]]:
    """
    List all departments in the Zoho Desk org — used purely to help find
    the value for ZOHO_IT_DEPARTMENT_ID (see config.py). Returns an empty
    list (never raises) on failure.
    """
    if not (config.zoho_client_id and config.zoho_client_secret and config.zoho_refresh_token):
        return []
    try:
        access_token = _get_access_token()
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        if config.zoho_org_id:
            headers["orgId"] = config.zoho_org_id
        response = requests.get(
            f"{config.zoho_api_domain}/api/v1/departments",
            headers=headers,
            params={"limit": 100},
            timeout=10,
        )
        response.raise_for_status()
        departments = response.json().get("data") or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to list departments: %s", exc)
        return []

    return [{"id": str(d.get("id") or ""), "name": d.get("name") or ""} for d in departments]


def get_ticket_number(ticket_id: str) -> str:
    """
    Resolve Zoho's internal numeric ticket ID to the short display number
    (e.g. "1042") agents actually see in the Zoho Desk UI. Used by the KB
    Builder's admin resync action to fix articles whose source_ticket_ids
    were stored as internal IDs before ticket_number was threaded through
    the scan pipeline (see kb_service.process_resolved_ticket).

    Returns "" (never raises) on failure — the caller treats an
    unresolved ID as "leave it as-is" rather than losing the reference
    entirely.
    """
    if not (config.zoho_client_id and config.zoho_client_secret and config.zoho_refresh_token):
        return ""
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
        return str(response.json().get("ticketNumber") or "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to resolve ticket_number for ticket_id=%s: %s", ticket_id, exc)
        return ""


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


def list_recently_resolved_tickets(since: _dt.datetime, limit: int = 50) -> list[dict[str, Any]]:
    """
    Fetch tickets whose status is Closed or Resolved and whose
    modifiedTime is after `since` — the KB scan job's source of "what
    became resolvable KB material since the last pass."

    Deliberately filters on Zoho's own statusType via the `status`
    query param rather than pulling everything and filtering client-side
    (this org's ticket volume makes that wasteful), then re-checks
    modifiedTime defensively in Python since API-side time filtering
    behavior isn't guaranteed exact across Desk API versions.

    Returns an empty list (rather than raising) on any failure, so a
    scan pass that hits a transient error simply does nothing that
    round rather than crashing the scheduler — the same `since`
    watermark will be retried next pass.
    """
    if not (config.zoho_client_id and config.zoho_client_secret and config.zoho_refresh_token):
        logger.info("Zoho OAuth not configured — skipping resolved-tickets scan.")
        return []

    try:
        access_token = _get_access_token()
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        if config.zoho_org_id:
            headers["orgId"] = config.zoho_org_id
    except ZohoAuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to authenticate for resolved-tickets scan: %s", exc)
        return []

    results: list[dict[str, Any]] = []
    max_pages = 20  # safety ceiling; a 30-min poll interval should rarely need more than a page or two
    PAGE_SIZE = 100

    for page in range(max_pages):
        try:
            response = requests.get(
                f"{config.zoho_api_domain}/api/v1/tickets",
                headers=headers,
                params={
                    "from": page * PAGE_SIZE,
                    "limit": PAGE_SIZE,
                    "status": "Closed,Resolved",
                    "sortBy": "-modifiedTime",
                },
                timeout=10,
            )
            response.raise_for_status()
            batch = response.json().get("data") or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch resolved-tickets page (from=%s): %s", page * PAGE_SIZE, exc)
            break

        if not batch:
            break

        stop = False
        for t in batch:
            modified_str = t.get("modifiedTime") or ""
            try:
                modified_dt = _dt.datetime.fromisoformat(modified_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            # Results are sorted newest-modified-first, so once we hit one
            # at or before the watermark, everything after it is too.
            if modified_dt <= since:
                stop = True
                break

            results.append(
                {
                    "ticket_id": str(t.get("id") or ""),
                    "ticket_number": t.get("ticketNumber") or "",
                    "subject": t.get("subject") or "",
                    "status": t.get("status") or "",
                    "modified_time": modified_str,
                }
            )
            if len(results) >= limit:
                stop = True
                break

        if stop or len(batch) < PAGE_SIZE:
            break

    return results


def list_recently_resolved_tickets(since: _dt.datetime, limit: int = 50) -> list[dict[str, Any]]:
    """
    Fetch tickets that moved to a Closed-type status, modified after
    `since`. Used by the KB Builder's scan job — polling rather than a
    webhook, consistent with this module's read-only, no-webhooks design.

    Detecting "resolved" does NOT rely on matching the literal status
    label "Closed"/"Resolved" — Zoho Desk lets each org define its own
    custom status names (e.g. "Solved", "Fixed", "Done"), grouped under
    Open / On Hold / Closed *types*. Matching on literal English words
    would silently match nothing for any org using different labels. The
    reliable signal is Zoho's `closedTime` field, which Zoho itself sets
    whenever a ticket enters a Closed-type status, regardless of the
    status's display name. `RESOLVED_STATUS_LABELS` below is kept as a
    secondary check purely as a fallback/diagnostic aid, and every ticket
    status encountered is logged at DEBUG so a scan that finds nothing can
    be diagnosed from the logs rather than being a silent black box.

    Zoho's List Tickets endpoint doesn't support filtering by status set +
    modified-time server-side, so this pages through tickets sorted
    newest-modified-first and stops as soon as a page runs older than
    `since` (same defensive-pagination pattern as get_related_tickets).
    Capped to `limit` results per call — the scan job's watermark means a
    ticket is only ever picked up once, so this doesn't need to catch
    everything in one pass; a slow month just spreads across more scan
    cycles.

    Returns an empty list (rather than raising) if credentials aren't
    configured or the lookup fails, so a scan cycle can log and retry next
    time instead of crashing the scheduler.
    """
    if not (config.zoho_client_id and config.zoho_client_secret and config.zoho_refresh_token):
        logger.info("Zoho OAuth not configured — skipping KB resolved-ticket scan.")
        return []

    # Defensive normalization to match _get_or_init_watermark's contract:
    # every comparison below assumes `since` is UTC-aware. A naive value
    # here would otherwise either raise when compared against the
    # timezone-aware timestamps parsed below, or — worse — silently
    # compare naive-to-naive if _parse_zoho_time also ever returns a naive
    # value, which would compare two datetimes that don't actually share a
    # reference frame without any error to signal it.
    if since.tzinfo is None:
        since = since.replace(tzinfo=_dt.timezone.utc)

    RESOLVED_STATUS_LABELS = {"closed", "resolved"}  # fallback only — see docstring
    MAX_PAGES = 50  # widened from 20: without a trusted sort order we can no longer
    # early-exit as soon as we hit old tickets, so a large backlog needs more pages
    # to fully cover. Watermark still keeps steady-state runs far smaller than this.

    try:
        access_token = _get_access_token()
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        if config.zoho_org_id:
            headers["orgId"] = config.zoho_org_id
    except ZohoAuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to authenticate for KB resolved-ticket scan: %s", exc)
        return []

    results: list[dict[str, Any]] = []
    seen_statuses: set[str] = set()  # for the one-line diagnostic summary below
    total_seen = 0

    pages_fetched = 0
    raw_tickets_seen = 0  # every ticket returned by any page, regardless of modified time

    for page in range(MAX_PAGES):
        try:
            request_params: dict[str, Any] = {"from": page * 100, "limit": 100}
            if config.zoho_it_department_id:
                # Filtered server-side by Zoho, not client-side after the
                # fact — this is what actually cuts the number of tickets
                # fetched (and therefore extracted/embedded) per scan,
                # rather than just hiding non-IT tickets after paying for
                # them anyway.
                request_params["departmentId"] = config.zoho_it_department_id
            response = requests.get(
                f"{config.zoho_api_domain}/api/v1/tickets",
                headers=headers,
                # No sortBy relied upon here — Zoho's default/actual list
                # order is not guaranteed to be newest-modified-first (see
                # docstring), so pagination below does NOT assume ordering
                # and does NOT early-exit on the first old ticket it finds.
                params=request_params,
                timeout=10,
            )
            response.raise_for_status()
            batch = response.json().get("data") or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch ticket page during KB scan (page=%s): %s", page, exc)
            break

        pages_fetched += 1
        if not batch:
            break  # true end of the ticket list

        raw_tickets_seen += len(batch)

        for t in batch:
            # closedTime is preferred over modifiedTime: it's set
            # specifically when a ticket enters a Closed-type status,
            # which is exactly the event the KB Builder cares about — a
            # ticket can be modified for unrelated reasons well before or
            # after it actually closes. It also sidesteps modifiedTime
            # coming back blank on this endpoint for some orgs/ticket
            # states (see the diagnostic log above), rather than debugging
            # why that field is empty. Falls back to modifiedTime only if
            # closedTime itself is missing.
            reference_time_str = t.get("closedTime") or t.get("modifiedTime") or ""
            reference_time = _parse_zoho_time(reference_time_str)
            if reference_time is None or reference_time <= since:
                continue  # NOT a break — order isn't trusted, so keep scanning the rest of this page/later pages

            total_seen += 1
            status_label = (t.get("status") or "").strip()
            seen_statuses.add(status_label or "(blank)")

            is_closed = bool(t.get("closedTime")) or status_label.lower() in RESOLVED_STATUS_LABELS
            if not is_closed:
                continue

            results.append(
                {
                    "ticket_id": str(t.get("id") or ""),
                    "ticket_number": t.get("ticketNumber") or "",
                    "subject": t.get("subject") or "",
                    "status": status_label,
                    "modified_time": reference_time_str,
                }
            )
            if len(results) >= limit:
                logger.info(
                    "KB scan: found %d closed ticket(s) across %d page(s) (statuses seen: %s).",
                    len(results), pages_fetched, sorted(seen_statuses),
                )
                return results

        if len(batch) < 100:
            break  # short page — that was the last one

    if raw_tickets_seen == 0:
        # The API call itself returned nothing at all — not even old
        # tickets. This points away from a modified-time/status problem
        # and toward auth/scope/org config: either credentials are wrong,
        # the OAuth scope doesn't include ticket read access, or orgId is
        # missing/incorrect for a multi-department account.
        logger.warning(
            "KB scan: the Zoho tickets API returned zero tickets across %d page(s) fetched — "
            "this looks like an auth/scope/orgId problem rather than a filtering problem. "
            "Check ZOHO_ORG_ID and that the OAuth token has ticket-read scope.",
            pages_fetched,
        )
    elif total_seen and not results:
        # This is the exact case reported as "no newly-resolved tickets" —
        # tickets WERE modified since the watermark, but none looked
        # closed. Surfacing the actual status labels seen turns "nothing
        # happened" into "here's why," without needing to add print
        # statements to debug it.
        logger.info(
            "KB scan: %d ticket(s) modified since last scan, but none had closedTime set or a "
            "recognized closed-type status label. Status labels seen: %s. If your Zoho org uses a "
            "custom status name for 'closed', it should still be caught via closedTime — if not, "
            "this list tells you what label to add.",
            total_seen, sorted(seen_statuses),
        )
    elif raw_tickets_seen and not total_seen:
        # Tickets came back, but none were newer than the watermark. If
        # you just closed a test ticket and still land here, either its
        # modifiedTime genuinely wasn't updated by the close action (some
        # custom workflows update status without touching modifiedTime),
        # it's further back than the %d pages scanned this cycle, or the
        # watermark itself is set later than the ticket's actual
        # modifiedTime (a timezone/clock mismatch between this server and
        # Zoho). Check the ticket's modifiedTime directly against the
        # watermark logged by run_scan() to tell which.
        logger.info(
            "KB scan: %d ticket(s) fetched across %d page(s), none closed/modified after the watermark (%s).",
            raw_tickets_seen, pages_fetched, since.isoformat(),
        )
    elif results:
        logger.info("KB scan: found %d closed ticket(s) (statuses seen: %s).", len(results), sorted(seen_statuses))

    return results


def _parse_zoho_time(value: str) -> Optional[_dt.datetime]:
    """
    Parse Zoho's timestamp format into a UTC-aware datetime.

    Handles two cases explicitly rather than trusting the string always
    carries an offset: if it ends in "Z" or has an explicit +HH:MM offset,
    fromisoformat already returns an aware datetime. If it has neither
    (e.g. "2026-08-15T05:50:00.000", no zone indicator at all), Zoho is
    still documented as returning UTC timestamps, so the naive result is
    explicitly labeled UTC rather than left ambiguous — leaving it naive
    is what caused watermark comparisons to be silently unsafe (see
    kb_service._get_or_init_watermark). A parse failure is logged (not
    just silently swallowed) since a bad format here previously looked
    identical to "ticket too old" in the scan's results.
    """
    if not value:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("KB scan: could not parse ticket timestamp %r — treating as unknown/skip.", value)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed


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


def fetch_ticket_comments(ticket_id: str) -> list[dict[str, Any]]:
    """
    Fetch internal/private notes (Zoho Desk "Comments", isPublic=false) for
    a ticket. Distinct from `/threads`, which is customer-facing email —
    Zoho keeps these as two separate endpoints. This matters a lot for KB
    extraction specifically: a technician's actual diagnosis and fix often
    live ONLY in an internal note (e.g. "reset the inkpad counter"), while
    the customer-facing reply just says "issue resolved." Without this,
    extraction would see no stated cause/resolution at all for tickets
    handled that way.

    Returns an empty list (never raises) on any failure, so a missing
    scope or transient error degrades to "conversation-only" rather than
    blocking extraction entirely.
    """
    if not (config.zoho_client_id and config.zoho_client_secret and config.zoho_refresh_token):
        return []

    try:
        access_token = _get_access_token()
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        if config.zoho_org_id:
            headers["orgId"] = config.zoho_org_id

        response = requests.get(
            f"{config.zoho_api_domain}/api/v1/tickets/{ticket_id}/comments",
            headers=headers,
            params={"limit": 100},
            timeout=10,
        )
        response.raise_for_status()
        comments = response.json().get("data") or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch internal comments for ticket_id=%s: %s", ticket_id, exc)
        return []

    # Public comments (isPublic=true) are customer-visible replies posted
    # via the comments API rather than threads — including them here too
    # would double them up against fetch_ticket_conversation's own thread
    # pull in most Desk configurations, so this function only surfaces the
    # private ones; that's the content /threads structurally cannot see.
    return [c for c in comments if c.get("isPublic") is False]


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
    comments = fetch_ticket_comments(ticket_id)
    if not threads and not comments:
        return None

    # Chronological order (oldest first) — defensive sort rather than
    # trusting API ordering, same approach used elsewhere in this module.
    threads.sort(key=lambda t: t.get("createdTime") or "")
    comments.sort(key=lambda c: c.get("commentedTime") or "")

    entries: list[tuple[str, str]] = []  # (sort_key, formatted_entry)
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

        entries.append((t.get("createdTime") or "", f"[{label}] {cleaned}"))

    for c in comments:
        raw = c.get("content") or ""
        cleaned = _html_to_text(raw)
        if not cleaned:
            continue
        commenter = c.get("commenter") or {}
        sender_name = commenter.get("name") or ""
        # Explicitly labeled "Internal Note" rather than folded into plain
        # "Agent" — an agent's internal note is where the real diagnosis
        # usually lives (it's private, so agents write frankly there in a
        # way they may not in a customer-facing reply), so it's worth
        # keeping visibly distinct in the transcript, not just correctly
        # attributed to the agent.
        label = f"Agent Internal Note — {sender_name}" if sender_name else "Agent Internal Note"
        entries.append((c.get("commentedTime") or "", f"[{label}] {cleaned}"))

    if not entries:
        return None

    entries.sort(key=lambda e: e[0])
    entries = [e[1] for e in entries]

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