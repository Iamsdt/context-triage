"""State models for unforget, persisted with the thread and shown in the playground."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from tenxgraph.core.state import AgentState


Action = Literal["keep", "compress", "drop"]
ACTIONS: tuple[Action, ...] = ("keep", "compress", "drop")


class Note(BaseModel):
    """A short note that replaced a compressed turn."""

    label: str = ""
    text: str


class Decision(BaseModel):
    """What happened to one turn. Plain strings, so threads from older versions still load."""

    label: str = ""
    action: str  # what was applied
    suggested: str  # Clef's top choice
    probability: float = 0.0


class UnforgetStats(BaseModel):
    context_tokens: int = 0  # size of the latest checked reply: prompt plus completion
    token_budget: int = 0
    last_event: str = ""
    clef_calls: int = 0
    turns_compressed: int = 0
    turns_dropped: int = 0


class UnforgetState(AgentState):
    """AgentState plus the fields unforget reads and writes. Subclass it for your graph."""

    notes: list[Note] = Field(default_factory=list)
    decision_log: list[Decision] = Field(default_factory=list)
    unforget_last_checked: str | int | None = None  # message_id of the last checked reply
    unforget_stats: UnforgetStats = Field(default_factory=UnforgetStats)
