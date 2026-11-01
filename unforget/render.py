"""Renders unforget's state into the single ``context_summary`` text block.

10xGraph places ``state.context_summary`` in the prompt ahead of the kept messages, so this
block is how the model sees what was pinned, compressed and archived.
"""

from __future__ import annotations

from unforget.models import ArchivedTurn, Fact, Note


HEADER = (
    "Memory from earlier in this conversation (managed by unforget). Older turns were "
    "pinned as facts, compressed into notes or archived. Nothing archived is lost."
)


def render_summary(
    ledger: list[Fact],
    notes: list[Note],
    archive: dict[int, ArchivedTurn],
    *,
    recall_tool: str = "recall",
) -> str | None:
    """Build the context summary text, or ``None`` when there is nothing to show."""
    current = [f for f in ledger if f.status == "current"]
    if not (current or notes or archive):
        return None

    parts = [HEADER]
    if current:
        lines = [f"- {f.key}: {f.value} (turn {f.turn})" for f in current]
        parts.append("Current facts (newer facts replace older ones):\n" + "\n".join(lines))
    if notes:
        lines = [
            f"- Turn {n.turn}: {n.text} (full turn: #{n.archive_id})"
            for n in sorted(notes, key=lambda n: n.turn)
        ]
        parts.append("Notes on earlier turns, oldest first:\n" + "\n".join(lines))
    if archive:
        lines = [f"- #{a.id}: turn {a.turn}, {a.label}" for a in _ordered(archive)]
        parts.append(
            f"Archived turns. Call {recall_tool}(archive_id) to read any of them in full "
            "when a detail matters:\n" + "\n".join(lines)
        )
    return "\n\n".join(parts)


def _ordered(archive: dict[int, ArchivedTurn]) -> list[ArchivedTurn]:
    return [archive[k] for k in sorted(archive)]
