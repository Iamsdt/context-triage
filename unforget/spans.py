"""Turn grouping, token estimates and plain-text rendering of turns.

A turn is one user message through the final assistant reply, with every tool call and
tool result in between. Removing whole turns keeps user/assistant alternation intact and
never separates a tool call from its result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tenxgraph.core.state import Message


TURN_KEY = "unforget_turn"

# Characters standing in for one media block (image, audio, video, document).
_MEDIA_BLOCK_CHARS = 4000
_MEDIA_TYPES = ("image", "audio", "video", "document")
_CHARS_PER_TOKEN = 4
_LABEL_WORDS = 8
_LABEL_CHARS = 60


@dataclass
class Turn:
    """One user message and everything that answered it (system messages excluded)."""

    messages: list[Message] = field(default_factory=list)
    number: int | None = None  # stable number stamped on the first message
    preamble: bool = False  # messages before the first user message; never a candidate

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.messages)

    @property
    def tool_names(self) -> list[str]:
        names: list[str] = []
        for msg in self.messages:
            for call in tool_calls_of(msg):
                if call["name"] not in names:
                    names.append(call["name"])
        return names

    @property
    def has_tool_output(self) -> bool:
        return any(msg.role == "tool" for msg in self.messages)

    @property
    def first_user_text(self) -> str:
        for msg in self.messages:
            if msg.role == "user":
                return msg.text().strip()
        return ""


def group_turns(messages: list[Message]) -> list[Turn]:
    """Split the non-system messages of a context into turns, in order.

    A new turn starts at every user message. Messages that come before the first user
    message form a preamble turn, which is never offered for removal.
    """
    turns: list[Turn] = []
    current: Turn | None = None
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "user":
            current = Turn(messages=[msg], number=_stamp_of(msg))
            turns.append(current)
        elif current is None:
            current = Turn(messages=[msg], preamble=True)
            turns.append(current)
        else:
            current.messages.append(msg)
    return turns


def stamp_turns(turns: list[Turn], next_number: int) -> int:
    """Give every unnumbered turn the next stable number. Returns the new counter.

    The number is written to the first message's metadata, so it survives checkpointing
    and stays the same however many earlier turns are removed later.
    """
    for turn in turns:
        if turn.preamble or turn.number is not None:
            continue
        turn.number = next_number
        turn.messages[0].metadata[TURN_KEY] = next_number
        next_number += 1
    return next_number


def _stamp_of(msg: Message) -> int | None:
    value = (msg.metadata or {}).get(TURN_KEY)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def tool_calls_of(msg: Message) -> list[dict[str, Any]]:
    """Tool calls an assistant message made, as ``{id, name, args}`` dicts."""
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in msg.content or []:
        if getattr(block, "type", None) in ("tool_call", "remote_tool_call"):
            call_id = str(getattr(block, "id", ""))
            seen.add(call_id)
            calls.append(
                {"id": call_id, "name": getattr(block, "name", ""), "args": block.args or {}}
            )
    for raw in msg.tools_calls or []:
        call_id = str(raw.get("id", ""))
        if call_id in seen:
            continue
        function = raw.get("function") or raw
        args: Any = function.get("arguments", function.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {"raw": args}
        calls.append({"id": call_id, "name": function.get("name", ""), "args": args})
    return calls


def _block_chars(block: Any) -> int:
    block_type = getattr(block, "type", None)
    if block_type == "text":
        return len(getattr(block, "text", "") or "")
    if block_type == "tool_result":
        output = getattr(block, "output", None)
        return len(output if isinstance(output, str) else json.dumps(output, default=str))
    if block_type in ("tool_call", "remote_tool_call"):
        return len(getattr(block, "name", "")) + len(json.dumps(block.args or {}, default=str))
    if block_type == "reasoning":
        return len(getattr(block, "summary", "") or "")
    if block_type in _MEDIA_TYPES:
        return _MEDIA_BLOCK_CHARS
    return 0


def estimate_tokens(messages: list[Message]) -> int:
    """Rough token estimate (about 4 characters per token) of a list of messages."""
    chars = 0
    for msg in messages:
        has_call_blocks = False
        for block in msg.content or []:
            chars += _block_chars(block)
            if getattr(block, "type", None) in ("tool_call", "remote_tool_call"):
                has_call_blocks = True
        if msg.tools_calls and not has_call_blocks:
            chars += len(json.dumps(msg.tools_calls, default=str))
    return chars // _CHARS_PER_TOKEN


def estimate_text_tokens(text: str | None) -> int:
    return len(text or "") // _CHARS_PER_TOKEN


def render_turn(turn_messages: list[Message]) -> str:
    """Render a turn as readable text: one line group per message."""
    lines: list[str] = []
    for msg in turn_messages:
        if msg.role == "tool":
            for block in msg.content or []:
                if getattr(block, "type", None) == "tool_result":
                    output = block.output
                    text = output if isinstance(output, str) else json.dumps(output, default=str)
                    tag = "TOOL ERROR" if block.is_error else "TOOL RESULT"
                    lines.append(f"{tag}: {text}")
            continue
        text = "".join(
            block.text
            for block in msg.content or []
            if getattr(block, "type", None) == "text" and block.text
        ).strip()
        if text:
            lines.append(f"{msg.role.upper()}: {text}")
        lines.extend(
            f"TOOL CALL: {call['name']}({json.dumps(call['args'], default=str)})"
            for call in tool_calls_of(msg)
        )
        media = [
            block.type for block in msg.content or [] if getattr(block, "type", "") in _MEDIA_TYPES
        ]
        if media:
            lines.append(f"{msg.role.upper()} ATTACHED: {', '.join(media)}")
    return "\n".join(lines)


def clip(text: str, limit: int) -> str:
    """Keep the head and tail of a long text within ``limit`` characters."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = max(limit - head, 0)
    omitted = len(text) - head - tail
    return f"{text[:head]} [... {omitted} chars omitted ...] {text[-tail:] if tail else ''}"


def turn_label(turn: Turn) -> str:
    """One-line label for an archived turn, derived from its text (no LLM)."""
    words = turn.first_user_text.split()
    opener = " ".join(words[:_LABEL_WORDS])
    if len(words) > _LABEL_WORDS:
        opener += "..."
    if len(opener) > _LABEL_CHARS:
        opener = opener[: _LABEL_CHARS - 3].rstrip() + "..."
    label = f'"{opener}"' if opener else "(no user text)"
    tools = turn.tool_names
    if tools:
        label += f" + tool output ({', '.join(tools)})"
    return f"{label}, {turn.tokens} tok"
