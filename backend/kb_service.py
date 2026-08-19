"""
kb_service.py
-------------
Knowledge Base Builder orchestration.

Turns resolved tickets into KB articles automatically, without letting the
KB balloon into near-duplicates. Every newly-resolved ticket is extracted
into a structured summary (symptoms/cause/resolution/keywords/related
systems), embedded, and compared against existing articles' embeddings. A
strong match reinforces the existing article (bumps occurrence_count,
optionally folds in a new detail) instead of creating a new draft; anything
else becomes a new `pending_review` draft.

This module owns the scan job (`run_scan_pass`) and is the only place that
talks to both `zoho_client` and `ai_service` for KB purposes — routes in
app.py call into this, never the other way around.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Optional

from ai_service import AIService
from config import config
from database import get_session
from models import KBArticle, KBScanState
import zoho_client

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Summary of one scan pass, for logging/observability."""

    tickets_seen: int = 0
    articles_created: int = 0
    articles_reinforced: int = 0
    tickets_skipped_not_extractable: int = 0
    tickets_skipped_error: int = 0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Standard cosine similarity. Returns 0.0 for degenerate (empty/zero) vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embedding_text(symptoms: str, cause: str, resolution: str) -> str:
    """
    The text actually embedded for similarity comparison. Deliberately
    includes cause + resolution alongside symptoms, not symptoms alone —
    two tickets with the same surface symptom but a genuinely different
    diagnosed cause/fix should NOT be merged into one article (that would
    present a wrong or incomplete fix as authoritative). Weighting the
    embedding toward the full causal shape, not just the symptom text,
    keeps those cases separate while still merging true repeats.
    """
    return f"{symptoms}\n\nCause: {cause}\n\nResolution: {resolution}"


def _get_or_init_watermark(session) -> _dt.datetime:
    """Read the scan watermark, initializing it to 'now' on first run so nothing is backfilled."""
    state = session.get(KBScanState, 1)
    if state is None:
        now = _dt.datetime.now(_dt.timezone.utc)
        state = KBScanState(id=1, last_scanned_at=now)
        session.add(state)
        session.flush()
        return now
    # SQLAlchemy's plain DateTime column, over SQLite, does not round-trip
    # tzinfo — a value written as UTC-aware (see _advance_watermark) comes
    # back naive on read. Every write in this module uses
    # datetime.now(timezone.utc), so a naive value read here always
    # represents UTC; re-attaching that label explicitly (rather than
    # leaving it naive) is what makes comparisons against the
    # timezone-aware timestamps parsed from Zoho's API safe and correct,
    # instead of silently comparing incompatible or misinterpreted values.
    watermark = state.last_scanned_at
    if watermark.tzinfo is None:
        watermark = watermark.replace(tzinfo=_dt.timezone.utc)
    return watermark


def _advance_watermark(session, new_time: _dt.datetime) -> None:
    state = session.get(KBScanState, 1)
    if state is None:
        session.add(KBScanState(id=1, last_scanned_at=new_time))
    else:
        state.last_scanned_at = new_time


def _find_best_match(session, embedding: list[float]) -> tuple[Optional[KBArticle], float]:
    """
    Compare `embedding` against every existing article's stored embedding
    and return the best match with its score. Linear scan in Python — at
    MVP scale (hundreds to low thousands of articles) this is fast and
    simple; SQLite has no native vector index, so a dedicated vector store
    (sqlite-vec, pgvector) is the natural upgrade if the KB grows past
    that, not something needed now.
    """
    best_article: Optional[KBArticle] = None
    best_score = 0.0
    # Compare against both approved and pending articles — a ticket
    # shouldn't create a second draft for something already awaiting
    # review, only against rejected ones (a reviewer explicitly said that
    # extraction was wrong, so it shouldn't keep absorbing reinforcements).
    articles = session.query(KBArticle).filter(KBArticle.status != "rejected").all()
    for article in articles:
        try:
            article_embedding = json.loads(article.embedding_json or "[]")
        except json.JSONDecodeError:
            continue
        score = _cosine_similarity(embedding, article_embedding)
        if score > best_score:
            best_score = score
            best_article = article
    return best_article, best_score


def match_or_create_article(
    ai_service: AIService,
    display_id: str,
    title: str,
    symptoms: str,
    cause: str,
    resolution: str,
    keywords: list[str],
    related_systems: list[str],
) -> dict[str, Any]:
    """
    Given already-extracted KB fields (from either the auto-scan's AI
    extraction, or an agent's own edited pre-fill via the manual
    "Suggest for KB" widget action), embed and either reinforce an
    existing matching article or create a new pending_review draft.
    This is the shared back half of process_resolved_ticket, factored
    out so a manual suggestion doesn't need to fabricate a fake ticket
    conversation just to reuse the matching logic — it already has the
    fields, extraction is skipped entirely.

    Returns {"action": "created"|"reinforced", "article_id": int}.
    """
    embedding = ai_service.embed_text(
        _embedding_text(symptoms, cause, resolution), task_type="SEMANTIC_SIMILARITY"
    )
    now = _dt.datetime.now(_dt.timezone.utc)

    with get_session() as session:
        best_article, best_score = (None, 0.0)
        if embedding:
            best_article, best_score = _find_best_match(session, embedding)

        if best_article is not None and best_score >= config.kb_similarity_threshold:
            source_ids = json.loads(best_article.source_ticket_ids_json or "[]")
            if display_id not in source_ids:
                source_ids.append(display_id)
            best_article.source_ticket_ids_json = json.dumps(source_ids)
            best_article.occurrence_count += 1
            best_article.last_seen_at = now

            merge_check = None
            if best_score < config.kb_merge_check_skip_threshold:
                merge_check = ai_service.check_new_detail(
                    best_article.to_dict(),
                    {
                        "symptoms": symptoms, "cause": cause,
                        "resolution": resolution, "related_systems": related_systems,
                    },
                )
            if merge_check and merge_check.get("has_new_detail"):
                updated_symptoms = str(merge_check.get("updated_symptoms") or "").strip()
                updated_systems = merge_check.get("updated_related_systems") or []
                if updated_symptoms:
                    best_article.symptoms = updated_symptoms
                if updated_systems:
                    best_article.related_systems_json = json.dumps(updated_systems)
                new_embedding = ai_service.embed_text(
                    _embedding_text(best_article.symptoms, best_article.cause, best_article.resolution),
                    task_type="SEMANTIC_SIMILARITY",
                )
                if new_embedding:
                    best_article.embedding_json = json.dumps(new_embedding)
                new_retrieval_embedding = ai_service.embed_text(
                    _embedding_text(best_article.symptoms, best_article.cause, best_article.resolution),
                    task_type="RETRIEVAL_DOCUMENT",
                )
                if new_retrieval_embedding:
                    best_article.retrieval_embedding_json = json.dumps(new_retrieval_embedding)
                logger.info(
                    "KB: ticket=%s reinforced article id=%s with new detail: %s",
                    display_id, best_article.id, merge_check.get("note"),
                )
            else:
                logger.info(
                    "KB: ticket=%s reinforced article id=%s (occurrence_count=%s, score=%.3f%s)",
                    display_id, best_article.id, best_article.occurrence_count, best_score,
                    ", merge check skipped" if merge_check is None else "",
                )
            return {"action": "reinforced", "article_id": best_article.id}

        # No strong match — create a new draft. Always pending_review,
        # even for a manually-suggested article — an agent's suggestion
        # is still a draft, not an approved fact, same quality gate as
        # everything the auto-scan produces.
        retrieval_embedding = ai_service.embed_text(
            _embedding_text(symptoms, cause, resolution), task_type="RETRIEVAL_DOCUMENT"
        )
        article = KBArticle(
            title=title,
            symptoms=symptoms,
            cause=cause,
            resolution=resolution,
            keywords_json=json.dumps(keywords),
            related_systems_json=json.dumps(related_systems),
            embedding_json=json.dumps(embedding or []),
            retrieval_embedding_json=json.dumps(retrieval_embedding or []),
            status="pending_review",
            occurrence_count=1,
            source_ticket_ids_json=json.dumps([display_id]),
            first_ticket_id=display_id,
            created_at=now,
            last_seen_at=now,
        )
        session.add(article)
        session.flush()
        logger.info("KB: ticket=%s created new draft article id=%s (best_score=%.3f)", display_id, article.id, best_score)
        return {"action": "created", "article_id": article.id}


def process_resolved_ticket(ai_service: AIService, ticket_id: str, ticket_number: str, subject: str) -> str:
    """
    Process a single resolved ticket end to end: fetch its conversation,
    extract KB fields, then delegate to match_or_create_article. Returns
    one of: "created", "reinforced", "not_extractable", "error" — used by
    the caller to tally a ScanResult.

    `ticket_id` (Zoho's internal numeric ID) is used for every Zoho API
    call, since that's what the API requires. `ticket_number` (the short
    display ID, e.g. "1042") is what actually gets stored in
    source_ticket_ids/first_ticket_id — that's what an agent sees in the
    Zoho Desk UI and can look up, whereas the internal ID means nothing to
    them. Falls back to ticket_id only if ticket_number is somehow blank,
    so a lookup failure degrades rather than losing the reference.
    """
    display_id = ticket_number or ticket_id
    try:
        conversation = zoho_client.fetch_ticket_conversation(ticket_id)
    except zoho_client.ZohoAuthError as exc:
        logger.warning("KB scan: auth error fetching conversation for ticket=%s (internal id=%s): %s", display_id, ticket_id, exc)
        return "error"

    if not conversation:
        logger.info("KB scan: no conversation content for ticket=%s — skipping.", display_id)
        return "not_extractable"

    extraction = ai_service.extract_kb_fields(ticket_id, subject, conversation)
    if not extraction.is_valid:
        logger.warning("KB scan: extraction failed for ticket=%s: %s", display_id, extraction.warnings)
        return "error"
    if not extraction.extractable:
        logger.info("KB scan: ticket=%s had no clear cause/resolution — not extractable.", display_id)
        return "not_extractable"

    data = extraction.data
    symptoms = str(data.get("symptoms", "")).strip()
    cause = str(data.get("cause", "")).strip()
    resolution = str(data.get("resolution", "")).strip()
    title = str(data.get("title", "")).strip() or "Untitled issue"
    keywords = data.get("keywords") or []
    related_systems = data.get("related_systems") or []

    result = match_or_create_article(ai_service, display_id, title, symptoms, cause, resolution, keywords, related_systems)
    return result["action"]


def resync_ticket_numbers(article: KBArticle) -> int:
    """
    Re-resolve an article's stored source_ticket_ids (and first_ticket_id)
    from Zoho's internal IDs to display ticket numbers, for articles
    created before ticket_number was threaded through the scan pipeline.
    Mutates `article` in place; caller is expected to be inside a
    `get_session()` block so the change is committed on exit.

    Returns the count of IDs actually changed. An ID that fails to
    resolve (already a ticket_number, or the lookup failed) is left
    as-is — this makes the action safe to run more than once.
    """
    source_ids = json.loads(article.source_ticket_ids_json or "[]")
    resolved_count = 0
    new_ids = []
    id_map: dict[str, str] = {}

    for tid in source_ids:
        # A stored ID that's already short/non-numeric-looking is very
        # likely already a display ticket number from a post-fix scan —
        # skip the lookup rather than spend an API call confirming it.
        # Zoho's internal IDs are long numeric strings (15+ digits);
        # ticket numbers are typically short. This is a heuristic, not a
        # guarantee, which is exactly why a failed/no-op lookup below is
        # harmless rather than destructive.
        if len(tid) < 10:
            new_ids.append(tid)
            continue
        resolved = zoho_client.get_ticket_number(tid)
        if resolved:
            new_ids.append(resolved)
            id_map[tid] = resolved
            resolved_count += 1
        else:
            new_ids.append(tid)  # leave unresolved rather than drop it

    if resolved_count:
        article.source_ticket_ids_json = json.dumps(new_ids)
        if article.first_ticket_id in id_map:
            article.first_ticket_id = id_map[article.first_ticket_id]

    return resolved_count


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "is", "are", "was", "were", "be", "been", "being", "with", "this", "that",
    "it", "its", "as", "by", "from", "has", "have", "had", "not", "no",
    "issue", "issues", "problem", "problems", "ticket", "tickets", "please",
    "help", "need", "needed", "getting", "get", "cant", "can't", "unable",
    "working", "work", "still", "again", "please", "kindly", "asap", "urgent",
}


def _significant_words(text: str) -> set[str]:
    """
    Extracts substantive, topic-bearing words for the lexical overlap
    check in find_relevant_articles — lowercased, punctuation-stripped,
    common English words and generic ticket-boilerplate terms (e.g.
    "issue", "not working", "urgent") filtered out, since those appear in
    nearly every ticket regardless of topic and would make the overlap
    check meaningless. Words of 2 characters or fewer are dropped as too
    short to be a reliable topic signal; 3-letter acronyms like "EMR"
    still pass.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def find_relevant_articles(
    ai_service: AIService, query_text: str, limit: Optional[int] = None
) -> list[dict[str, Any]]:
    """
    Semantic search over APPROVED articles for a given query (typically a
    ticket's subject + description), used to feed real KB context into
    Analyze/Chat. Distinct from search_articles() (LIKE-based, for the
    widget's manual free-text KB search box) — this one uses embeddings,
    so it surfaces genuinely related content even when the wording
    differs from the article's text.

    Uses RETRIEVAL_QUERY for the query embedding and compares against each
    article's RETRIEVAL_DOCUMENT-typed embedding (article.
    retrieval_embedding_json) — a deliberately different, asymmetric
    embedding type from the SEMANTIC_SIMILARITY one used for dedup
    (article.embedding_json).

    Embedding similarity ALONE is not trusted as sufficient — short ticket
    subjects/descriptions don't carry enough signal for pure vector
    similarity to reliably separate "genuinely related" from "same general
    IT-support register, different topic" (observed directly: a ticket
    about EMR slowness matching an unrelated Wi-Fi article). An article
    must pass ONE of:
      (a) embedding score >= kb_relevance_high_confidence_threshold alone
          (default 0.8) — high enough that a paraphrase with zero literal
          word overlap is still trusted, or
      (b) embedding score >= kb_relevance_threshold (default 0.65) AND
          shares at least 2 substantive words with the query — a single
          shared word (e.g. this org's own system name appearing in
          nearly every article) isn't reliable topical corroboration on
          its own.
    This is a standard hybrid lexical+semantic pattern for exactly this
    failure mode, not a workaround.

    Self-healing backfill: articles created before retrieval_embedding_json
    existed have it as "[]" — computed and persisted the first time such
    an article is encountered here, no separate migration step needed.

    Returns [] (never raises) on an embedding failure — Analyze/Chat
    should degrade to "no KB context" rather than fail outright over this.
    """
    query_embedding = ai_service.embed_text(query_text, task_type="RETRIEVAL_QUERY")
    if not query_embedding:
        return []

    query_words = _significant_words(query_text)
    limit = limit or config.kb_relevant_articles_limit

    with get_session() as session:
        rows = session.query(KBArticle).filter(KBArticle.status == "approved").all()
        scored: list[tuple[float, KBArticle]] = []
        backfilled = 0
        for article in rows:
            try:
                article_embedding = json.loads(article.retrieval_embedding_json or "[]")
            except json.JSONDecodeError:
                article_embedding = []

            if not article_embedding:
                # Self-healing backfill for pre-existing articles.
                new_embedding = ai_service.embed_text(
                    _embedding_text(article.symptoms, article.cause, article.resolution),
                    task_type="RETRIEVAL_DOCUMENT",
                )
                if new_embedding:
                    article.retrieval_embedding_json = json.dumps(new_embedding)
                    article_embedding = new_embedding
                    backfilled += 1
                else:
                    continue  # embedding failed — skip this article for this search rather than block it

            score = _cosine_similarity(query_embedding, article_embedding)
            if score < config.kb_relevance_threshold:
                continue

            if score < config.kb_relevance_high_confidence_threshold:
                article_words = _significant_words(
                    f"{article.title} {article.symptoms} {article.cause} "
                    + " ".join(json.loads(article.keywords_json or "[]"))
                )
                # Requires 2+ shared words, not just 1 — a single shared
                # word is too easy to satisfy by coincidence via a
                # generic org-wide term (e.g. this org's core system
                # name, like "EMR," appearing in nearly every article
                # regardless of actual topic) rather than genuine topical
                # overlap. A genuinely high embedding score (path a,
                # above) can still rescue a real single-concept match
                # that happens to share only one word.
                if len(query_words & article_words) < 2:
                    continue

            scored.append((score, article))

        if backfilled:
            logger.info("KB relevance search: backfilled retrieval_embedding_json for %d older article(s).", backfilled)

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [{"score": round(score, 3), **article.to_dict()} for score, article in scored[:limit]]


def run_scan_pass(ai_service: AIService) -> ScanResult:
    """
    One full scan pass: find tickets resolved since the last watermark,
    process each in chronological order, advancing the watermark after
    every ticket (not just at the end of the batch). Safe to call
    repeatedly (e.g. by the scheduler) — a failed, timed-out, or
    externally-killed pass leaves the watermark at the last ticket it
    actually finished, so already-processed tickets are never silently
    skipped AND never endlessly re-fetched from the original starting
    point either.

    Per-ticket watermark advancement (rather than one commit at the end)
    matters specifically on a free-tier host: kb_scan_batch_size tickets,
    each needing a Zoho fetch plus one or more Gemini calls, can easily
    take longer than the process has before something (gunicorn's worker
    timeout, Render's own proxy) kills the request. Committing progress as
    we go means that kill loses at most the one ticket in flight, not the
    whole pass — and the next scheduled run picks up right after it
    instead of re-walking the same early tickets forever.
    """
    result = ScanResult()

    with get_session() as session:
        since = _get_or_init_watermark(session)

    scan_started_at = _dt.datetime.now(_dt.timezone.utc)
    tickets = zoho_client.list_recently_resolved_tickets(since, limit=config.kb_scan_batch_size)
    result.tickets_seen = len(tickets)

    if not tickets:
        logger.info("KB scan: no newly-resolved tickets since %s.", since.isoformat())
        with get_session() as session:
            _advance_watermark(session, scan_started_at)
        return result

    # zoho_client.list_recently_resolved_tickets does NOT guarantee
    # newest/oldest ordering (see its docstring) — sort here so that
    # "advance the watermark to this ticket's own time" is actually safe:
    # once ticket i is processed, every ticket at or before its time in
    # this batch is guaranteed to have been attempted already. Tickets
    # with an unparseable timestamp are pushed to the end and processed,
    # but never used to advance the watermark, since we can't be sure
    # nothing earlier is still waiting behind them.
    def _sort_key(t: dict[str, Any]) -> tuple[int, _dt.datetime]:
        parsed = zoho_client._parse_zoho_time(t.get("modified_time") or "")
        if parsed is None:
            return (1, scan_started_at)  # unparseable — sort last, never advances the watermark
        return (0, parsed)

    tickets_sorted = sorted(tickets, key=_sort_key)

    for t in tickets_sorted:
        try:
            outcome = process_resolved_ticket(
                ai_service, t["ticket_id"], t.get("ticket_number", ""), t.get("subject", "")
            )
        except Exception:  # noqa: BLE001 - one bad ticket (e.g. a Gemini call that
            # still fails despite the timeout fix) must not lose progress on
            # every other ticket already handled this pass.
            logger.exception(
                "KB scan: unhandled error processing ticket=%s — counting as error, continuing.",
                t.get("ticket_number") or t.get("ticket_id"),
            )
            outcome = "error"

        if outcome == "created":
            result.articles_created += 1
        elif outcome == "reinforced":
            result.articles_reinforced += 1
        elif outcome == "not_extractable":
            result.tickets_skipped_not_extractable += 1
        else:
            result.tickets_skipped_error += 1

        reference_time = zoho_client._parse_zoho_time(t.get("modified_time") or "")
        if reference_time is not None:
            with get_session() as session:
                _advance_watermark(session, reference_time)

    # Only jump the watermark all the way to scan_started_at once every
    # ticket in the batch has actually been attempted — otherwise this
    # would skip anything past the last one processed if the batch was
    # truncated by kb_scan_batch_size (more tickets exist than we fetched).
    if len(tickets_sorted) < config.kb_scan_batch_size:
        with get_session() as session:
            _advance_watermark(session, scan_started_at)

    logger.info(
        "KB scan complete: seen=%s created=%s reinforced=%s not_extractable=%s errors=%s",
        result.tickets_seen, result.articles_created, result.articles_reinforced,
        result.tickets_skipped_not_extractable, result.tickets_skipped_error,
    )
    return result


def search_articles(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Simple keyword search over APPROVED articles only — LIKE-based, which
    is plenty at MVP scale. Ranked by occurrence_count (a repeatedly-seen,
    reviewer-approved article is more likely to be the right answer than
    one seen once), then recency.
    """
    like_query = f"%{query.strip()}%"
    with get_session() as session:
        rows = (
            session.query(KBArticle)
            .filter(KBArticle.status == "approved")
            .filter(
                (KBArticle.title.ilike(like_query))
                | (KBArticle.symptoms.ilike(like_query))
                | (KBArticle.cause.ilike(like_query))
                | (KBArticle.resolution.ilike(like_query))
                | (KBArticle.keywords_json.ilike(like_query))
            )
            .order_by(KBArticle.occurrence_count.desc(), KBArticle.last_seen_at.desc())
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in rows]
