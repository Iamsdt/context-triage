"""The ``recall`` tool: reads an archived turn back as text.

It returns the turn as a tool result; it does not put the old messages back into the
context. The archive itself is never modified, so recalling twice gives the same text.
"""

from __future__ import annotations

from typing import Any

from unforget.spans import render_turn


def recall(archive_id: int, state: Any = None) -> str:
    """Read an archived conversation turn back in full.

    Use this when the memory summary lists an archived item (for example "#3") and you
    need its exact details, such as a log line, a number or a file name.

    Args:
        archive_id: The number after "#" in the archived turns list.
    """
    archive = getattr(state, "archive", None) or {}
    try:
        key = int(archive_id)
    except (TypeError, ValueError):
        key = None
    item = archive.get(key) if key is not None else None
    if item is None:
        available = ", ".join(f"#{k}" for k in sorted(archive)) or "none"
        return f"No archived item #{archive_id}. Archived items: {available}."

    body = render_turn(item.messages) or "(the turn had no readable text)"
    return (
        f"Archived item #{item.id}: turn {item.turn}, {item.label} "
        f"(archived by {item.kind}).\n\n{body}"
    )
