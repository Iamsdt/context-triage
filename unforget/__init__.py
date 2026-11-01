"""Unforget: context compression where Cloudflare's Clef decides, per old conversation turn,
whether to keep it, compress it into a short note, or drop it.
"""

from unforget.clef import ClefClient, ClefError, ClefResult
from unforget.manager import UnforgetContextManager
from unforget.models import Decision, Note, UnforgetState, UnforgetStats


__all__ = [
    "ClefClient",
    "ClefError",
    "ClefResult",
    "Decision",
    "Note",
    "UnforgetContextManager",
    "UnforgetState",
    "UnforgetStats",
]
