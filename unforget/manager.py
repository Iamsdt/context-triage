"""UnforgetContextManager: Clef decides per old turn whether to keep, compress or drop it.

Core awaits ``atrim_context`` at the end of every run, and before each model call when the
agent sets ``trim_context=True``. A check runs at most once per new assistant reply:

1. Size: the latest reply's real usage (prompt, cached input and completion). At or under
   ``token_budget`` nothing happens.
2. Candidates: every turn except the last ``keep_recent_turns``, oldest first, 64 at most.
3. One Clef call asks keep, compress or drop for each; below ``min_confidence`` means keep.
4. One writer call turns the compressed turns into 1 to 3 sentence notes. Compressed and
   dropped turns leave the context, and ``context_summary`` is rebuilt from the notes.

Any failure leaves the context unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from tenxgraph.core.state import BaseContextManager
from tenxgraph.utils.callable_utils import run_coroutine

from unforget.clef import MAX_QUESTIONS, ChoiceAnswer, ClefClient, DecisionClient
from unforget.models import Decision, Note, UnforgetState
from unforget.turns import Turn, clip, group_turns, render_turn, tool_calls_of
from unforget.writer import LLMFn, make_call_llm, write_notes


logger = logging.getLogger("unforget")

CLEF_TASK = (
    "An AI assistant's conversation is over its context budget. For each listed turn, decide "
    "what should happen to it so the context gets smaller without losing anything the "
    "assistant still needs. A compressed turn is replaced by a short note; a dropped turn is "
    "gone. The latest user message shows what the conversation is about right now."
)

ACTION_CRITERIA = {
    "keep": "Needed verbatim: the current work depends on its exact wording, code or numbers.",
    "compress": (
        "Useful but verbose, such as tool output, logs or listings, or it states preferences "
        "or decisions: a 1 to 3 sentence note would keep what matters."
    ),
    "drop": "Greetings, thanks, small talk or chatter with nothing worth reusing.",
}

SUMMARY_HEADER = "Notes on earlier turns of this conversation, oldest first:"
DECISION_LOG_SIZE = 100

# Core's OpenAI, OpenAI Responses and Google converters report prompt tokens that already
# include cached input. Anthropic's input_tokens leave out cache reads and cache writes.
_CACHE_INCLUSIVE_PROVIDERS = ("openai", "google_genai")


class UnforgetContextManager[S: UnforgetState](BaseContextManager[S]):
    """Context manager where Clef decides, per old turn, keep, compress or drop.

    Args:
        token_budget: Check only when the latest reply's prompt plus completion exceeds this.
        keep_recent_turns: The most recent turns are never offered to Clef (at least 1).
        min_confidence: Apply Clef's top option only at or above this probability.
        clef_model: Clef model name (the REST endpoint accepts only ``"clef"``).
        writer_model: Cheap LLM that writes the compress notes (tenxgraph ``call_llm``).
        clef_client: Decision client; defaults to :class:`ClefClient` (Cloudflare env vars).
        llm: Replaces the writer call, ``async (system_prompt, prompt) -> str``.
        clef_chars_per_turn: Each turn's text is clipped to this for Clef.
        writer_chars_per_turn: Each turn's text is clipped to this for the writer.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        token_budget: int = 8000,
        keep_recent_turns: int = 4,
        min_confidence: float = 0.7,
        clef_model: str = "clef",
        writer_model: str = "claude-haiku-5-5",
        clef_client: DecisionClient | None = None,
        llm: LLMFn | None = None,
        clef_chars_per_turn: int = 800,
        writer_chars_per_turn: int = 6000,
    ) -> None:
        if keep_recent_turns < 1:
            raise ValueError("keep_recent_turns must be >= 1")  # the last turn may be in progress
        self.token_budget = token_budget
        self.keep_recent_turns = keep_recent_turns
        self.min_confidence = min_confidence
        self.clef_model = clef_model
        self.clef: DecisionClient = clef_client or ClefClient()
        self.llm: LLMFn = llm or make_call_llm(writer_model)
        self.clef_chars_per_turn = clef_chars_per_turn
        self.writer_chars_per_turn = writer_chars_per_turn

    def trim_context(self, state: S) -> S:
        return run_coroutine(self.atrim_context(state))

    async def atrim_context(self, state: S) -> S:
        """Trim ``state`` in place and return it."""
        messages = list(state.context or [])
        reply = self._due(state, messages)
        if reply is None:
            return state
        stats = state.unforget_stats
        state.unforget_last_checked = reply.message_id
        turns = [t for t in group_turns(messages) if not t.preamble]
        candidates = turns[: max(len(turns) - self.keep_recent_turns, 0)][:MAX_QUESTIONS]
        over = f"{stats.context_tokens} > {self.token_budget} tok"
        if not candidates:
            stats.last_event = f"Over budget ({over}) but only protected recent turns are left."
            return state
        if not self.clef.configured:
            return _fail(state, "Clef is not configured (CLOUDFLARE_ACCOUNT_ID/API_TOKEN)")

        try:
            answers = await self._decide(candidates, messages)
            stats.clef_calls += 1
            plan = [
                (t, a, a.choice if a.probability >= self.min_confidence else "keep")
                for t, a in zip(candidates, answers, strict=True)
            ]
            to_compress = {
                i: clip(render_turn(t.messages), self.writer_chars_per_turn)
                for i, (t, _, action) in enumerate(plan, 1)
                if action == "compress"
            }
            notes = await write_notes(self.llm, to_compress) if to_compress else {}
        except Exception as exc:
            return _fail(state, f"Clef or note writer failed: {exc!r}")

        applied = self._apply(state, messages, plan, notes)
        stats.last_event = (
            f"Clef call {stats.clef_calls} ({over}): of {len(candidates)} older turns, "
            f"{applied.count('keep')} kept, {applied.count('compress')} compressed, "
            f"{applied.count('drop')} dropped."
        )
        logger.info("unforget: %s", stats.last_event)
        return state

    def _due(self, state: S, messages: list[Any]) -> Any | None:
        """The latest reply with usage when a check should run for it, else None."""
        if not isinstance(state, UnforgetState):
            logger.warning("unforget needs a state that subclasses UnforgetState")
            return None
        reply = _latest_usage(messages)
        if reply is None or reply.message_id == state.unforget_last_checked:
            return None
        # Never change the context mid tool round: the model is about to read the results.
        if _in_tool_round(messages):
            return None
        stats = state.unforget_stats
        stats.context_tokens = _prompt_size(reply) + (reply.usages.completion_tokens or 0)
        stats.token_budget = self.token_budget
        return reply if stats.context_tokens > self.token_budget else None

    async def _decide(self, turns: list[Turn], messages: list[Any]) -> list[ChoiceAnswer]:
        """One Clef call with one keep/compress/drop question per turn, by position."""
        ids = [f"turn_{i}" for i in range(1, len(turns) + 1)]
        latest_user = next((m.text().strip() for m in reversed(messages) if m.role == "user"), "")
        state = {
            "task": CLEF_TASK,
            "latest_user_message": clip(latest_user, 400),
            "turns": [
                {"id": qid, "text": clip(render_turn(t.messages), self.clef_chars_per_turn)}
                for qid, t in zip(ids, turns, strict=True)
            ],
        }
        ask = "The entry with id {} in turns: what should happen to this turn?"
        questions = {
            qid: {"type": "choice", "instructions": ask.format(qid), "criteria": ACTION_CRITERIA}
            for qid in ids
        }
        result = await self.clef.decide(state, questions, self.clef_model)
        return [result.answers[qid] for qid in ids]

    def _apply(
        self,
        state: UnforgetState,
        messages: list[Any],
        plan: list[tuple[Turn, ChoiceAnswer, str]],
        notes: dict[int, str],
    ) -> list[str]:
        """Remove compressed and dropped turns, add notes, log, rebuild the summary.

        Returns the action applied to each turn.
        """
        stats = state.unforget_stats
        removed: set[int] = set()
        applied: list[str] = []
        for i, (turn, answer, planned) in enumerate(plan, 1):
            action = "keep" if planned == "compress" and i not in notes else planned
            applied.append(action)
            if action == "compress":
                state.notes.append(Note(label=turn.label, text=notes[i]))
                stats.turns_compressed += 1
            elif action == "drop":
                stats.turns_dropped += 1
            if action != "keep":
                removed.update(id(m) for m in turn.messages)
            p = round(answer.probability, 3)
            state.decision_log.append(
                Decision(label=turn.label, action=action, suggested=answer.choice, probability=p)
            )
        state.decision_log = state.decision_log[-DECISION_LOG_SIZE:]

        context = [m for m in messages if id(m) not in removed]
        lines = [f"- {n.text}" for n in state.notes]
        summary = "\n".join([SUMMARY_HEADER, *lines]) if lines else None
        if removed or summary != state.context_summary:
            context = _strip_stale_thinking(context)
        state.context = context
        state.context_summary = summary
        return applied


def _fail[S: UnforgetState](state: S, message: str) -> S:
    logger.warning("unforget: %s; context left unchanged", message)
    state.unforget_stats.last_event = f"{message}; context left unchanged."
    return state


def _latest_usage(messages: list[Any]) -> Any | None:
    """The latest assistant message whose usage has a prompt size."""
    replies = (m for m in reversed(messages) if m.role == "assistant" and m.usages is not None)
    return next((m for m in replies if _prompt_size(m) > 0), None)


def _prompt_size(msg: Any) -> int:
    """The call's whole prompt, cached input included.

    Core's Anthropic converter reports ``input_tokens``, which leave out cache reads and
    writes, so those are added. For an unknown provider the cache fields are added only
    when they exceed ``prompt_tokens`` (they cannot be part of it then).
    """
    usage = msg.usages
    prompt = usage.prompt_tokens or 0
    cached = (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0)
    provider = (msg.metadata or {}).get("provider")
    if provider in _CACHE_INCLUSIVE_PROVIDERS:
        return prompt
    if provider == "anthropic" or cached > prompt:
        return prompt + cached
    return prompt


def _in_tool_round(messages: list[Any]) -> bool:
    """True when the last non-system message is a tool result or an unanswered tool call."""
    last = next((m for m in reversed(messages) if m.role != "system"), None)
    if last is None:
        return False
    if last.role == "tool" or any(
        getattr(b, "type", None) == "tool_result" for b in last.content or []
    ):
        return True
    return last.role == "assistant" and bool(tool_calls_of(last))


def _strip_stale_thinking(messages: list[Any]) -> list[Any]:
    """Copies of ``messages`` without signed or redacted reasoning blocks.

    Claude signs a thinking block against the whole prefix before it, so once turns are
    removed or the summary changes, kept blocks would fail with "Invalid signature in
    thinking block". Input messages are not mutated.
    """
    out: list[Any] = []
    for msg in messages:
        blocks = msg.content if isinstance(msg.content, list) else []
        kept = [b for b in blocks if not _signed_thinking(b)]
        out.append(msg if len(kept) == len(blocks) else msg.model_copy(update={"content": kept}))
    return out


def _signed_thinking(block: Any) -> bool:
    signed = getattr(block, "signature", None) or getattr(block, "redacted_data", None)
    return getattr(block, "type", None) == "reasoning" and bool(signed)
