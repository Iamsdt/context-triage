"""The cheap LLM work behind ``compress`` and ``pin``: one batched call per action per trim."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from tenxgraph.core.llm.caller import call_llm

from unforget.models import Fact


logger = logging.getLogger("unforget.writer")

# (system_prompt, prompt) -> raw model text that should hold one JSON object.
LLMFn = Callable[[str, str], Awaitable[str]]

COMPRESS_SYSTEM = (
    "You write short memory notes that replace old turns of a conversation between a user "
    "and an AI assistant. For each turn, write 1 to 3 sentences that keep what matters for "
    "continuing the conversation: names, numbers, results, conclusions and open questions. "
    "Write in third person, past tense. Do not invent anything.\n"
    'Return JSON only: {"notes": [{"turn": <turn number>, "note": "<text>"}]} '
    "with one entry per turn."
)

PIN_SYSTEM = (
    "You maintain a ledger of durable facts for an AI assistant: user preferences, "
    "decisions, constraints, names, settings and agreed plans. Extract every durable fact "
    "stated in the given turns. Use short snake_case keys and short values. If a new fact "
    "updates, reverses or contradicts a fact already in the ledger, list that fact's key in "
    '"replaces"; reuse the same key when it is the same subject. Do not invent anything.\n'
    'Return JSON only: {"facts": [{"turn": <turn number>, "key": "<key>", '
    '"value": "<value>", "replaces": ["<existing key>"]}]}'
)


class WriterError(RuntimeError):
    """The LLM call failed or returned something that is not the expected JSON."""


@dataclass
class TurnText:
    turn: int
    text: str


@dataclass
class ExtractedFact:
    turn: int
    key: str
    value: str
    replaces: list[str] = field(default_factory=list)


def make_call_llm(model: str, *, max_tokens: int = 1024) -> LLMFn:
    """An ``LLMFn`` backed by tenxgraph's ``call_llm`` (provider detected from ``model``)."""

    async def _call(system_prompt: str, prompt: str) -> str:
        text, *_ = await call_llm(
            model,
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            json_mode=True,
            # call_llm sends temperature=0.3 unless told otherwise, and current Claude
            # models (claude-haiku-5-5, claude-sonnet-5-5) reject non-default values with a
            # 400. 1.0 is the API default, so it is accepted.
            temperature=1.0,
        )
        return text

    return _call


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model reply (tolerates code fences)."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise WriterError("LLM reply holds no JSON object")
    try:
        data = json.loads(cleaned[start : end + 1])
    except ValueError as exc:
        raise WriterError(f"LLM reply is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise WriterError("LLM reply JSON is not an object")
    return data


def _turns_block(turns: list[TurnText]) -> str:
    return "\n\n".join(f"=== Turn {t.turn} ===\n{t.text}" for t in turns)


async def write_notes(llm: LLMFn, turns: list[TurnText]) -> dict[int, str]:
    """One call for every turn being compressed. Returns ``{turn: note}``.

    Turns the model skipped are missing from the result; the caller decides what to do.
    """
    if not turns:
        return {}
    try:
        reply = await llm(COMPRESS_SYSTEM, _turns_block(turns))
    except Exception as exc:
        raise WriterError(f"compress call failed: {type(exc).__name__}: {exc}") from exc
    data = parse_json_object(reply)
    raw_notes = data.get("notes")
    if not isinstance(raw_notes, list):
        raise WriterError('compress reply has no "notes" list')
    wanted = {t.turn for t in turns}
    notes: dict[int, str] = {}
    for item in raw_notes:
        if not isinstance(item, dict):
            continue
        turn, note = _as_int(item.get("turn")), str(item.get("note") or "").strip()
        if turn in wanted and note:
            notes[turn] = note
    return notes


async def extract_facts(
    llm: LLMFn, turns: list[TurnText], current: list[Fact]
) -> list[ExtractedFact]:
    """One call for every turn being pinned, given the current ledger."""
    if not turns:
        return []
    ledger = "\n".join(f"- {f.key}: {f.value}" for f in current) or "(empty)"
    prompt = f"Current ledger:\n{ledger}\n\nTurns:\n\n{_turns_block(turns)}"
    try:
        reply = await llm(PIN_SYSTEM, prompt)
    except Exception as exc:
        raise WriterError(f"pin call failed: {type(exc).__name__}: {exc}") from exc
    data = parse_json_object(reply)
    raw_facts = data.get("facts")
    if not isinstance(raw_facts, list):
        raise WriterError('pin reply has no "facts" list')
    wanted = {t.turn for t in turns}
    facts: list[ExtractedFact] = []
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        turn = _as_int(item.get("turn"))
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        if turn not in wanted or not key or not value:
            continue
        replaces = item.get("replaces") or []
        if not isinstance(replaces, list):
            replaces = [replaces]
        facts.append(
            ExtractedFact(
                turn=turn,
                key=key,
                value=value,
                replaces=[str(r).strip() for r in replaces if str(r).strip()],
            )
        )
    return facts


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None
