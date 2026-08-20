"""
models.py
---------
SQLAlchemy ORM models.

Currently defines a single table, `ticket_analysis`, which stores every AI
analysis performed. This is intentionally append-only (no updates/deletes in
the MVP) so it can double as an audit trail and, later, a dataset for
reporting and prompt evaluation.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Optional

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


class TicketAnalysis(Base):
    """
    Stores the result of a single AI analysis run for a ticket.

    Fields
    ------
    id: Primary key.
    ticket_id: The Zoho Desk ticket ID that was analyzed.
    subject: The ticket subject at the time of analysis (denormalized for
        quick reporting without needing to re-fetch from Zoho).
    analysis_json: The full structured AI response, stored as a JSON string.
    model: The AI model identifier used to produce this analysis
        (e.g. "gemini-2.0-flash").
    processing_time: Wall-clock seconds the AI call took.
    created_at: UTC timestamp of when the analysis was stored.
    """

    __tablename__ = "ticket_analysis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    subject: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    analysis_json: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    processing_time: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[_dt.datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: _dt.datetime.now(_dt.timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the row for API responses."""
        return {
            "id": self.id,
            "ticket_id": self.ticket_id,
            "subject": self.subject,
            "model": self.model,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class KBArticle(Base):
    """
    A Knowledge Base article, either auto-generated from a resolved ticket
    or reinforced (folded together) from several similar resolved tickets.

    Lifecycle: created as `pending_review` by the KB scan job. A reviewer
    either approves it (becomes searchable via /api/kb/search) or rejects
    it. Subsequent similar tickets bump `occurrence_count` on an already-
    approved (or still-pending) article rather than creating a duplicate —
    see kb_service.py for the similarity/reinforcement logic. Kept
    append-only on the ticket-linkage side (`source_ticket_ids` only ever
    grows) even though `status` and the extracted fields are mutable,
    so the article's full provenance is always recoverable.

    Fields
    ------
    id: Primary key.
    title: Short human-readable title (AI-generated, editable by reviewer).
    symptoms: What the customer/user observed.
    cause: Root cause, as diagnosed by the resolving agent.
    resolution: What actually fixed it.
    keywords_json: JSON list of search keywords.
    related_systems_json: JSON list of related systems/services.
    embedding_json: JSON list of floats — the embedding used for
        similarity matching, computed from symptoms+cause+resolution.
        Recomputed whenever those fields change materially.
    status: "pending_review" | "approved" | "rejected".
    occurrence_count: How many resolved tickets have matched this article.
        Higher = more confidently a real, recurring, correctly-diagnosed
        issue — a useful ranking signal in search, independent of recency.
    source_ticket_ids_json: JSON list of every Zoho ticket_id that
        contributed to this article (first one that created it, plus every
        one later folded in as a reinforcement).
    first_ticket_id: The ticket_id that originally created this article
        (denormalized for quick display without parsing the JSON list).
        "" for an article created manually by a reviewer rather than
        generated from a ticket (see is_manual on the API response).
    reviewed_by: Free-text identifier of who approved/rejected (from
        basic-auth username on the admin page).
    reviewed_at: When the review decision was made.
    created_at: When this article was first generated.
    last_seen_at: When it was last reinforced by a new matching ticket.
    """

    __tablename__ = "kb_article"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    symptoms: Mapped[str] = mapped_column(Text, nullable=False, default="")
    cause: Mapped[str] = mapped_column(Text, nullable=False, default="")
    resolution: Mapped[str] = mapped_column(Text, nullable=False, default="")
    keywords_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    related_systems_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    # Separate embedding, typed for retrieval (Gemini task_type=
    # RETRIEVAL_DOCUMENT), used only by the Analyze/Chat relevance search
    # (kb_service.find_relevant_articles). Deliberately distinct from
    # embedding_json (task_type=SEMANTIC_SIMILARITY, used for the
    # dedup/reinforcement check) — a short ticket query matched against a
    # stored article is a different comparison shape than "are these two
    # ticket summaries the same issue," and Gemini's embedding model
    # genuinely optimizes differently for each; sharing one vector across
    # both was the root cause of search returning weakly-related results.
    # "[]" for articles created before this field existed — see
    # find_relevant_articles for the one-time self-healing backfill.
    retrieval_embedding_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending_review", index=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    source_ticket_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    first_ticket_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    reviewed_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    reviewed_at: Mapped[Optional[_dt.datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[_dt.datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: _dt.datetime.now(_dt.timezone.utc)
    )
    last_seen_at: Mapped[_dt.datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: _dt.datetime.now(_dt.timezone.utc)
    )

    def to_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        import json as _json

        data = {
            "id": self.id,
            "title": self.title,
            "symptoms": self.symptoms,
            "cause": self.cause,
            "resolution": self.resolution,
            "keywords": _json.loads(self.keywords_json or "[]"),
            "related_systems": _json.loads(self.related_systems_json or "[]"),
            "status": self.status,
            "occurrence_count": self.occurrence_count,
            "source_ticket_ids": _json.loads(self.source_ticket_ids_json or "[]"),
            "first_ticket_id": self.first_ticket_id,
            "is_manual": not self.first_ticket_id,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
        }
        if include_embedding:
            data["embedding"] = _json.loads(self.embedding_json or "[]")
        return data


class KBScanState(Base):
    """
    Single-row table tracking the KB scan job's watermark — the timestamp
    of the last successful scan. The next scan only looks at tickets
    resolved/closed after this, so each ticket is processed once ("forward
    from today," never backfilled). A single fixed row (id=1) is used
    rather than a generic key-value table since there is exactly one thing
    to track.
    """

    __tablename__ = "kb_scan_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    last_scanned_at: Mapped[_dt.datetime] = mapped_column(DateTime, nullable=False)


class KBSubmissionIdempotency(Base):
    """
    Guards POST /api/kb/suggest against being processed twice for what was
    really a single agent action — specifically the widget's own
    cold-start/network-drop retry, which can genuinely re-send the same
    submission if the first attempt's response never made it back to the
    browser even though the server went on to complete it. This is a
    different problem from (and complementary to) KBArticle's own
    similarity-based dedup: that one merges genuinely different
    submissions that happen to describe the same underlying issue;
    this one exists purely to make an exact retry of the same click a
    no-op instead of a second real submission.

    The unique constraint on idempotency_key is what actually enforces
    "claim once" under concurrency — the INSERT racing two near-
    simultaneous requests with the same key will succeed for exactly one
    of them, the other gets an IntegrityError and falls back to reading
    (or briefly waiting on) this row instead of proceeding independently.

    status: "processing" (claimed, work not yet finished) ->
        "completed" (result_action/result_article_id are populated) or
        "failed" (error_message populated; a later request with the same
        key is allowed to retry rather than being stuck behind a dead
        attempt forever).
    """

    __tablename__ = "kb_submission_idempotency"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="processing")
    result_action: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    result_article_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[_dt.datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: _dt.datetime.now(_dt.timezone.utc)
    )
    completed_at: Mapped[Optional[_dt.datetime]] = mapped_column(DateTime, nullable=True)

    def to_result_dict(self) -> dict[str, Any]:
        return {"action": self.result_action, "article_id": self.result_article_id}
