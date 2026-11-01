"""Writes the short notes that replace compressed turns, in one batched LLM call."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from anthropic import AsyncAnthropic


# (system_prompt, prompt) -> raw model text that should hold one JSON object.
LLMFn = Callable[[str, str], Awaitable[str]]

COMPRESS_SYSTEM = (
    "You write short memory notes that replace old turns of a conversation between a user "
    "and an AI assistant. For each turn, write 1 to 3 sentences that keep what matters for "
    "continuing the conversation: names, numbers, results, decisions, preferences and open "
    "questions. Write in third person, past tense. Do not invent anything.\n"
    'Return JSON only: {"notes": [{"turn": <turn number>, "note": "<text>"}]} '
    "with one entry per turn."
)


def make_call_llm(model: str, *, max_tokens: int = 1024) -> LLMFn:
    """An ``LLMFn`` that calls Claude through the Anthropic SDK (reads ANTHROPIC_API_KEY).

    Core's ``call_llm`` is not used because it forwards ``temperature``, which this SDK
    version rejects. The client is created on the first call, since the key may not be set
    when the writer is built.
    """
    client: AsyncAnthropic | None = None

    async def _call(system_prompt: str, prompt: str) -> str:
        nonlocal client
        if client is None:
            client = AsyncAnthropic()
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    return _call


async def write_notes(llm: LLMFn, turns: dict[int, str]) -> dict[int, str]:
    """Turn ``{number: text}`` into ``{number: note}``. Skipped turns are left out."""
    prompt = "\n\n".join(f"=== Turn {n} ===\n{text}" for n, text in turns.items())
    reply = await llm(COMPRESS_SYSTEM, prompt)
    data = json.loads(reply[reply.find("{") : reply.rfind("}") + 1])  # tolerates code fences
    wanted = {str(n): n for n in turns}
    notes: dict[int, str] = {}
    for item in data.get("notes") or []:
        number = wanted.get(str(item.get("turn")).strip()) if isinstance(item, dict) else None
        note = str(item.get("note") or "").strip() if number is not None else ""
        if number is not None and note:
            notes[number] = note
    return notes
