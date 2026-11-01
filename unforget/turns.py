"""Turns: a user message and everything that answered it, tool calls and results included.

Removing whole turns keeps user/assistant alternation intact and never separates a tool
call from its result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tenxgraph.core.state import Message


_MEDIA_TYPES = ("image", "audio", "video", "document")
_LABEL_WORDS = 8


@dataclass
class Turn:
    messages: list[Message] = field(default_factory=list)
    preamble: bool = False  # messages before the first user message; never touched

    @property
    def label(self) -> str:
        """The first words of the user message."""
        text = next((m.text().strip() for m in self.messages if m.role == "user"), "")
        words = text.split()
        if not words:
            return "(no user text)"
        return " ".join(words[:_LABEL_WORDS])[:60] + ("..." if len(words) > _LABEL_WORDS else "")


def group_turns(messages: list[Message]) -> list[Turn]:
    """Split the non-system messages into turns; a new turn starts at every user message."""
    turns: list[Turn] = []
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "user" or not turns:
            turns.append(Turn(preamble=msg.role != "user"))
        turns[-1].messages.append(msg)
    return turns


def tool_calls_of(msg: Message) -> list[dict[str, Any]]:
    """Tool calls an assistant message made, as ``{id, name, args}`` dicts."""
    calls = [
        {"id": str(b.id), "name": b.name, "args": b.args or {}}
        for b in msg.content or []
        if getattr(b, "type", None) in ("tool_call", "remote_tool_call")
    ]
    seen = {c["id"] for c in calls}
    for raw in msg.tools_calls or []:
        if str(raw.get("id", "")) in seen:
            continue
        function = raw.get("function") or raw
        args = function.get("arguments", function.get("args", {}))
        calls.append({"id": str(raw.get("id", "")), "name": function.get("name", ""), "args": args})
    return calls


def render_turn(messages: list[Message]) -> str:
    """Render a turn as readable text, one line group per message."""
    lines: list[str] = []
    for msg in messages:
        texts: list[str] = []
        for block in msg.content or []:
            kind = getattr(block, "type", None)
            if kind == "text" and block.text:
                texts.append(block.text)
            elif kind == "tool_result":
                out = block.output
                text = out if isinstance(out, str) else json.dumps(out, default=str)
                lines.append(f"{'TOOL ERROR' if block.is_error else 'TOOL RESULT'}: {text}")
            elif kind in _MEDIA_TYPES:
                lines.append(f"{msg.role.upper()} ATTACHED: {kind}")
        if "".join(texts).strip():
            lines.append(f"{msg.role.upper()}: {''.join(texts).strip()}")
        lines.extend(
            f"TOOL CALL: {c['name']}({json.dumps(c['args'], default=str)})"
            for c in tool_calls_of(msg)
        )
    return "\n".join(lines)


def clip(text: str, limit: int) -> str:
    """Keep the head and tail of a long text within ``limit`` characters."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return f"{text[:head]} [... {len(text) - limit} chars omitted ...] {text[-tail:]}"
