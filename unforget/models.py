"""State models for unforget.

Everything unforget decides lives on the graph state, so the checkpointer persists it with
the thread and the playground's State tab shows it next to ``context`` and
``context_summary``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from tenxgraph.core.state import AgentState, Message


Action = Literal["keep", "compress", "archive", "drop", "pin"]
ACTIONS: tuple[Action, ...] = ("keep", "compress", "archive", "drop", "pin")


class ArchivedTurn(BaseModel):
    """A whole turn taken out of the context, stored verbatim so it can be recalled."""

    id: int
    turn: int
    kind: Action  # the action that archived it: archive, compress or pin
    label: str
    tokens: int
    messages: list[Message] = Field(default_factory=list)


class Note(BaseModel):
    """A short note that stands in for a compressed turn, kept in turn order."""

    turn: int
    text: str
    archive_id: int


class Fact(BaseModel):
    """One durable fact in the ledger. Superseded facts are kept, never deleted."""

    key: str
    value: str
    turn: int
    archive_id: int | None = None
    status: Literal["current", "superseded"] = "current"
    superseded_by_turn: int | None = None


class Decision(BaseModel):
    """What happened to one turn, as shown in the decision log."""

    turn: int
    trim: int
    suggested: Action  # Clef's top choice
    action: Action  # what was applied (keep when the confidence gate held it back)
    probability: float
    probabilities: dict[str, float] = Field(default_factory=dict)
    applied: bool
    reason: str
    label: str
    tokens: int
    tokens_freed: int = 0
    archive_id: int | None = None
    background: bool = False  # decided by a background job, applied on a later call


class UnforgetStats(BaseModel):
    """Running totals and the outcome of the latest trim."""

    token_budget: int = 0
    context_tokens: int = 0  # estimate at the latest check, summary included
    clef_calls: int = 0
    clef_input_tokens: int = 0
    turns_decided: int = 0
    tokens_freed: int = 0
    last_tokens_before: int = 0
    last_tokens_after: int = 0
    last_event: str = ""
    # Background mode: jobs started, applied, failed (Clef, writer or timeout; cooldown) and
    # discarded (cancelled, or every turn left the context before pickup).
    background_jobs_started: int = 0
    background_jobs_applied: int = 0
    background_jobs_failed: int = 0
    background_jobs_discarded: int = 0


class UnforgetState(AgentState):
    """AgentState plus the fields unforget reads and writes.

    Subclass this for your graph's state (add your own fields as usual).
    """

    ledger: list[Fact] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)
    archive: dict[int, ArchivedTurn] = Field(default_factory=dict)
    decision_log: list[Decision] = Field(default_factory=list)
    decided_turns: list[int] = Field(default_factory=list)
    next_turn_number: int = 1
    next_archive_id: int = 1
    unforget_stats: UnforgetStats = Field(default_factory=UnforgetStats)
