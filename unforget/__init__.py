"""Unforget: context compression that can always undo.

Cloudflare's Clef decides, per conversation turn, what to keep, compress, archive, drop or
pin. Archived turns are stored verbatim in the graph state and the ``recall`` tool brings
them back.
"""

from unforget.clef import ClefClient, ClefError, ClefResult
from unforget.manager import UnforgetContextManager
from unforget.models import (
    ArchivedTurn,
    Decision,
    Fact,
    Note,
    UnforgetState,
    UnforgetStats,
)
from unforget.recall import recall


__all__ = [
    "ArchivedTurn",
    "ClefClient",
    "ClefError",
    "ClefResult",
    "Decision",
    "Fact",
    "Note",
    "UnforgetContextManager",
    "UnforgetState",
    "UnforgetStats",
    "recall",
]
