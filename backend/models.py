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
