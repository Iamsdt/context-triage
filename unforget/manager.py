"""UnforgetContextManager: Clef decides per turn what to keep, compress, archive, drop or pin.

How a trim works:

1. The context is split into turns (a user message through the final reply, tool calls and
   results included). Each turn gets a stable number, stored in its first message's metadata.
2. Nothing happens until the estimated tokens of the context (plus the summary block) exceed
   ``token_budget``.
3. The oldest undecided turns outside the last ``keep_recent_turns`` are picked until removing
   them would bring the estimate down to ``target_ratio * token_budget`` (64 at most).
4. One Clef call asks one ``choice`` question per picked turn. The top option is applied only
   when its probability reaches ``min_confidence``; otherwise the turn is kept.
5. One cheap LLM call writes notes for all ``compress`` turns, one extracts facts for all
   ``pin`` turns. Archived, compressed and pinned turns are stored verbatim in
   ``state.archive``; the ``recall`` tool reads them back.
6. Every decided turn is recorded, so later trims never ask Clef about it again.

Background mode (``background=True``, the default): 10xGraph awaits ``atrim_context`` before
every model call and when it saves the state, so steps 4 and 5 would sit in the user's request
path. Instead, a trim snapshots the picked turns, runs steps 4 and 5 as a task on the graph's
``BackgroundTaskManager`` and returns the context unchanged. The next ``atrim_context`` for the
same thread applies the finished decisions (no network). One job runs per thread. Past
``hard_budget_ratio * token_budget`` the call waits for the running job, or trims inline when
there is none, so the context cannot grow without bound during one long run.

Any error (Clef, LLM, missing credentials, malformed reply) leaves the context unchanged.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from injectq import InjectQ
from tenxgraph.core.state import BaseContextManager
from tenxgraph.utils.background_task_manager import BackgroundTaskManager, current_task_manager
from tenxgraph.utils.callable_utils import run_coroutine

from unforget.clef import CLEF_MODELS, MAX_QUESTIONS, ChoiceAnswer, ClefClient, DecisionClient
from unforget.models import (
    Action,
    ArchivedTurn,
    Decision,
    Fact,
    Note,
    UnforgetState,
)
from unforget.render import render_summary
from unforget.spans import (
    Turn,
    clip,
    estimate_text_tokens,
    estimate_tokens,
    group_turns,
    render_turn,
    stamp_turns,
    turn_label,
)
from unforget.writer import (
    ExtractedFact,
    LLMFn,
    TurnText,
    extract_facts,
    make_call_llm,
    write_notes,
)


S = TypeVar("S", bound=UnforgetState)

logger = logging.getLogger("unforget")

CLEF_TASK = (
    "An AI assistant's conversation is over its context budget. For each listed turn, decide "
    "what should happen to it so the context gets smaller without losing anything the "
    "assistant still needs. Archived, compressed and pinned turns stay recoverable; dropped "
    "turns do not. The latest user message shows what the conversation is about right now."
)

# Option descriptions sent as the criteria of every choice question.
ACTION_CRITERIA: dict[str, str] = {
    "keep": "Needed verbatim: the current work depends on its exact wording, code or numbers.",
    "compress": "Useful but verbose: a 1 to 3 sentence note would keep what matters.",
    "archive": (
        "Large or detailed content, such as tool output, logs or listings, that is not "
        "needed right now but may be needed later."
    ),
    "drop": "Greetings, thanks, small talk or chatter with nothing worth reusing.",
    "pin": (
        "States durable facts, user preferences, decisions or constraints that must keep "
        "applying later."
    ),
}

_REMOVING: tuple[Action, ...] = ("compress", "archive", "drop", "pin")
_ARCHIVING: tuple[Action, ...] = ("compress", "archive", "pin")

# Leak guard for background jobs (process memory only). A finished job nobody picks up within
# _RESULT_TTL seconds is forgotten; its turns are simply asked again later.
_RESULT_TTL = 600.0
_MAX_JOBS = 256
# The job enforces job_timeout itself (a timeout counts as a failure, with cooldown). The task
# manager's own timeout is a backstop a little later; it cancels, which is discarded silently.
_TASK_TIMEOUT_GRACE = 10.0

# (turn, Clef's answer, action to apply, whether the confidence gate let it through)
Plan = list[tuple[Turn, ChoiceAnswer, Action, bool]]


@dataclass
class _Snapshot:
    """What the Clef and writer calls read, taken from the state when a trim starts."""

    batch: list[Turn]
    latest_user: str
    current_facts: list[Fact]
    before: int


@dataclass
class _Outcome:
    """Result of the Clef and writer calls; ``_commit`` applies it to a state."""

    plan: Plan = field(default_factory=list)
    notes: dict[int, str] = field(default_factory=dict)
    facts: list[ExtractedFact] = field(default_factory=list)
    clef_input_tokens: int | None = None  # None until Clef has answered
    error: str | None = None


@dataclass
class _Job:
    """A background trim for one thread, held in process memory until it is applied."""

    task: asyncio.Task[_Outcome]
    turns: list[int]
    loop: asyncio.AbstractEventLoop
    started: float  # time.monotonic()
    finished_at: float | None = None


class UnforgetContextManager(BaseContextManager[S]):
    """Context manager where Clef decides per turn, and nothing archived is ever lost.

    Args:
        token_budget: Trim only when the estimated context tokens exceed this.
        target_ratio: A trim aims to bring the estimate down to ``target_ratio *
            token_budget``.
        keep_recent_turns: The most recent turns are never offered to Clef (at least 1, since
            the newest turn may still be in progress).
        min_confidence: Apply Clef's top option only at or above this probability;
            below it the turn is kept.
        clef_model: ``"clef"`` (default and the only model the REST endpoint accepts).
        writer_model: Cheap LLM for compress notes and pinned facts (tenxgraph ``call_llm``).
        clef_client: Decision client. Defaults to :class:`ClefClient` reading
            ``CLOUDFLARE_ACCOUNT_ID`` and ``CLOUDFLARE_API_TOKEN`` at call time.
        llm: Replaces the writer call, ``async (system_prompt, prompt) -> str``. Defaults to
            ``call_llm`` with ``writer_model``.
        max_questions: Upper bound on turns per Clef call (Clef allows 64).
        clef_chars_per_turn: Each turn's text is clipped to this for Clef, to keep cost low.
        writer_chars_per_turn: Each turn's text is clipped to this for the writer LLM.
        failure_cooldown: Seconds to skip trimming after a Clef or LLM failure, so an outage
            does not add a timeout to every model call.
        recall_tool_name: Tool name the summary tells the model to call.
        background: Run the Clef and writer calls as a background task and apply their
            decisions on the next call for the same thread, so the user never waits for
            them. ``False`` awaits them inside the trim. The sync ``trim_context`` always
            trims inline.
        hard_budget_ratio: In background mode, once the estimate passes ``hard_budget_ratio *
            token_budget`` the trim waits for the thread's running job, or trims inline when
            none is running. Must be greater than 1.
        job_timeout: Seconds a background job (one Clef call plus the writer calls) may take
            before it counts as failed.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        token_budget: int = 8000,
        target_ratio: float = 0.6,
        keep_recent_turns: int = 4,
        min_confidence: float = 0.7,
        clef_model: str = "clef",
        writer_model: str = "claude-haiku-5-5",
        clef_client: DecisionClient | None = None,
        llm: LLMFn | None = None,
        max_questions: int = MAX_QUESTIONS,
        clef_chars_per_turn: int = 800,
        writer_chars_per_turn: int = 6000,
        failure_cooldown: float = 60.0,
        recall_tool_name: str = "recall",
        background: bool = True,
        hard_budget_ratio: float = 2.0,
        job_timeout: float = 120.0,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if not 0 < target_ratio <= 1:
            raise ValueError("target_ratio must be in (0, 1]")
        if keep_recent_turns < 1:
            # The newest turn may still be in progress (a tool call waiting for its result).
            raise ValueError("keep_recent_turns must be >= 1")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        if clef_model not in CLEF_MODELS:
            raise ValueError(f"clef_model must be one of {CLEF_MODELS}")
        if not 1 <= max_questions <= MAX_QUESTIONS:
            raise ValueError(f"max_questions must be in [1, {MAX_QUESTIONS}]")
        if hard_budget_ratio <= 1:
            raise ValueError("hard_budget_ratio must be greater than 1")
        if job_timeout <= 0:
            raise ValueError("job_timeout must be positive")

        self.token_budget = token_budget
        self.target_ratio = target_ratio
        self.keep_recent_turns = keep_recent_turns
        self.min_confidence = min_confidence
        self.clef_model = clef_model
        self.writer_model = writer_model
        self.clef: DecisionClient = clef_client or ClefClient()
        self.llm: LLMFn = llm or make_call_llm(writer_model)
        self.max_questions = max_questions
        self.clef_chars_per_turn = clef_chars_per_turn
        self.writer_chars_per_turn = writer_chars_per_turn
        self.failure_cooldown = failure_cooldown
        self.recall_tool_name = recall_tool_name
        self.background = background
        self.hard_budget_ratio = hard_budget_ratio
        self.hard_budget = int(token_budget * hard_budget_ratio)
        self.job_timeout = job_timeout

        self._cooldown_until = 0.0
        self._warned: set[str] = set()
        # Background jobs by thread id: at most one per thread, running or waiting for pickup.
        self._jobs: dict[str, _Job] = {}

    # ------------------------------------------------------------------
    # BaseContextManager interface
    # ------------------------------------------------------------------

    def trim_context(self, state: S) -> S:
        # A background task cannot outlive the temporary event loop run_coroutine starts.
        return run_coroutine(self.atrim_context(state, inline=True))

    async def atrim_context(self, state: S, *, inline: bool = False) -> S:  # noqa: PLR0911, PLR0912
        """Trim ``state`` in place and return the same object.

        Args:
            state: The graph state; must subclass :class:`UnforgetState`.
            inline: Await the Clef and writer calls here even in background mode.
        """
        if not isinstance(state, UnforgetState):
            self._warn_once(
                "state",
                "unforget needs a state that subclasses UnforgetState; got %s. Context left "
                "unchanged.",
                type(state).__name__,
            )
            return state
        messages = list(state.context or [])
        if not messages:
            return state

        turns = group_turns(messages)
        state.next_turn_number = stamp_turns(turns, state.next_turn_number)

        stats = state.unforget_stats
        stats.token_budget = self.token_budget
        before = estimate_tokens(messages) + estimate_text_tokens(state.context_summary)
        stats.context_tokens = before

        thread = _thread_of(state) if self.background and not inline else None
        if thread is not None:
            self._sweep_jobs()
            job = self._jobs.get(thread)
            if job is not None and not job.task.done() and before > self.hard_budget:
                await self._wait_for(thread, job)
            if self._pickup(state, thread):
                messages = list(state.context or [])
                turns = group_turns(messages)
                before = estimate_tokens(messages) + estimate_text_tokens(state.context_summary)
                stats.context_tokens = before

        if before <= self.token_budget:
            return state

        running = self._jobs.get(thread) if thread is not None else None
        if running is not None:
            # One job per thread; its turns are not offered again while it runs.
            stats.last_event = (
                f"Over budget (~{before} > {self.token_budget} tok); background Clef job for "
                f"turns {_span(running.turns)} still running."
            )
            return state

        batch = self._pick_batch(turns, set(state.decided_turns), before)
        if not batch:
            event = (
                f"Over budget (~{before} > {self.token_budget} tok) but every older turn "
                f"is already decided; the last {self.keep_recent_turns} turns are protected. "
                "Nothing more to do."
            )
            if stats.last_event != event:
                logger.info("unforget: %s", event)
            stats.last_event = event
            return state

        if not self.clef.configured:
            self._warn_once(
                "creds",
                "unforget: CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set; context left "
                "unchanged (over budget: ~%d > %d tok).",
                before,
                self.token_budget,
            )
            stats.last_event = "Clef credentials missing; context left unchanged."
            return state

        if time.monotonic() < self._cooldown_until:
            logger.debug("unforget: in failure cooldown; context left unchanged")
            return state

        if thread is not None and before <= self.hard_budget:
            manager = self._task_manager()
            if manager is not None:
                return self._start_job(
                    state, thread, batch, manager=manager, messages=messages, before=before
                )

        return await self._trim(state, messages, batch, before)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _pick_batch(self, turns: list[Turn], decided: set[int], before: int) -> list[Turn]:
        """Oldest undecided, unprotected turns until removing them would reach the target."""
        real = [t for t in turns if not t.preamble and t.number is not None]
        cut = len(real) - self.keep_recent_turns
        eligible = [t for t in real[: max(cut, 0)] if t.number not in decided]
        need = before - int(self.token_budget * self.target_ratio)
        batch: list[Turn] = []
        total = 0
        for turn in eligible:
            batch.append(turn)
            total += turn.tokens
            if total >= need or len(batch) >= self.max_questions:
                break
        return batch

    async def _trim(self, state: S, messages: list[Any], batch: list[Turn], before: int) -> S:
        """Inline trim: the Clef and writer calls, then apply, all inside this call."""
        snapshot = self._snapshot(state, messages, batch, before, copy=False)
        outcome = await self._decide(snapshot, _Outcome())
        self._commit(state, messages, outcome, before=before)
        return state

    def _snapshot(
        self, state: S, messages: list[Any], batch: list[Turn], before: int, *, copy: bool
    ) -> _Snapshot:
        latest_user = next((m.text().strip() for m in reversed(messages) if m.role == "user"), "")
        current = [f for f in state.ledger if f.status == "current"]
        if copy:
            # A background job runs while the graph keeps changing the state: own copies.
            batch = [
                Turn(messages=[m.model_copy(deep=True) for m in t.messages], number=t.number)
                for t in batch
            ]
            current = [f.model_copy() for f in current]
        return _Snapshot(batch=batch, latest_user=latest_user, current_facts=current, before=before)

    async def _decide(self, snap: _Snapshot, outcome: _Outcome) -> _Outcome:
        """One Clef call plus the writer calls. Reads only ``snap``; errors land on ``outcome``.

        ``asyncio.CancelledError`` is never caught here.
        """
        questions = {
            _qid(t): {
                "type": "choice",
                "instructions": (
                    f"Turn {t.number} (the entry with id {_qid(t)} in turns): what should "
                    "happen to this turn?"
                ),
                "criteria": ACTION_CRITERIA,
            }
            for t in snap.batch
        }

        try:
            result = await self.clef.decide(self._clef_state(snap), questions, self.clef_model)
        except Exception as exc:
            outcome.error = f"Clef call failed: {exc}"
            return outcome
        outcome.clef_input_tokens = result.input_tokens

        try:
            for turn in snap.batch:
                answer = result.answers[_qid(turn)]
                applied = answer.probability >= self.min_confidence
                action: Action = answer.choice if applied else "keep"  # type: ignore[assignment]
                outcome.plan.append((turn, answer, action, applied))
        except Exception as exc:
            outcome.error = f"Clef response unusable: {exc!r}"
            return outcome

        try:
            outcome.notes, outcome.facts = await self._write(snap.current_facts, outcome.plan)
        except Exception as exc:
            outcome.error = f"note/fact writer failed: {exc}"
        return outcome

    def _commit(
        self,
        state: S,
        messages: list[Any],
        outcome: _Outcome,
        *,
        before: int,
        background: bool = False,
    ) -> bool:
        """Count the Clef call, then apply the outcome or fail. True when decisions applied."""
        stats = state.unforget_stats
        if outcome.clef_input_tokens is not None:
            stats.clef_calls += 1
            stats.clef_input_tokens += outcome.clef_input_tokens
        if outcome.error is not None:
            self._fail(state, outcome.error)
            return False
        try:
            self._apply(
                state,
                messages,
                outcome.plan,
                notes=outcome.notes,
                facts=outcome.facts,
                before=before,
                background=background,
            )
        except Exception as exc:
            self._fail(state, f"applying decisions failed: {exc!r}")
            return False
        return True

    def _apply(  # noqa: PLR0913, PLR0915
        self,
        state: S,
        messages: list[Any],
        plan: Plan,
        *,
        notes: dict[int, str],
        facts: list[ExtractedFact],
        before: int,
        background: bool = False,
    ) -> None:
        """Apply the plan to copies of the state fields, then commit them all at once."""
        stats = state.unforget_stats
        trim_no = stats.clef_calls
        ledger = [f.model_copy() for f in state.ledger]
        note_list = list(state.notes)
        archive = dict(state.archive)
        log = list(state.decision_log)
        decided = list(state.decided_turns)
        next_id = state.next_archive_id

        removed: set[int] = set()
        freed_total = 0
        for turn, answer, planned, applied in plan:
            action = planned
            reason = self._reason(answer, applied)
            if action == "compress" and turn.number not in notes:
                action = "archive"
                reason += "; the writer returned no note, so the turn was archived instead"
            label = turn_label(turn)
            tokens = turn.tokens
            archive_id = None
            if action in _ARCHIVING:
                archive_id = next_id
                next_id += 1
                archive[archive_id] = ArchivedTurn(
                    id=archive_id,
                    turn=turn.number,
                    kind=action,
                    label=label,
                    tokens=tokens,
                    messages=[m.model_copy(deep=True) for m in turn.messages],
                )
            if action == "compress":
                note_list.append(
                    Note(turn=turn.number, text=notes[turn.number], archive_id=archive_id)
                )
            if action == "pin":
                added = _apply_facts(ledger, turn.number, archive_id, facts)
                reason += f"; {added} fact(s) pinned" if added else "; no durable facts found"
            freed = tokens if action in _REMOVING else 0
            if action in _REMOVING:
                removed.update(id(m) for m in turn.messages)
                freed_total += freed

            log.append(
                Decision(
                    turn=turn.number,
                    trim=trim_no,
                    suggested=answer.choice,  # type: ignore[arg-type]
                    action=action,
                    probability=round(answer.probability, 4),
                    probabilities={k: round(v, 4) for k, v in answer.probabilities.items()},
                    applied=applied,
                    reason=reason,
                    label=label,
                    tokens=tokens,
                    tokens_freed=freed,
                    archive_id=archive_id,
                    background=background,
                )
            )
            decided.append(turn.number)
            logger.info(
                "unforget turn %d: %s (clef %s p=%.2f%s) %s",
                turn.number,
                action,
                answer.choice,
                answer.probability,
                "" if applied else ", gated",
                label,
            )

        context = [m for m in messages if id(m) not in removed]
        summary = render_summary(ledger, note_list, archive, recall_tool=self.recall_tool_name)
        after = estimate_tokens(context) + estimate_text_tokens(summary)

        # Commit: plain assignments only from here on.
        state.ledger = ledger
        state.notes = note_list
        state.archive = archive
        state.decision_log = log
        state.decided_turns = decided
        state.next_archive_id = next_id
        state.context = context
        if summary is not None:
            state.context_summary = summary

        stats.turns_decided += len(plan)
        stats.tokens_freed += freed_total
        stats.last_tokens_before = before
        stats.last_tokens_after = after
        stats.context_tokens = after
        what = (
            f"applied background decisions for turns {_span(t.number for t, *_ in plan)}"
            if background
            else f"decided {len(plan)} turn(s)"
        )
        stats.last_event = (
            f"Trim {trim_no}: {what}, ~{before} -> ~{after} tok (budget {self.token_budget})."
        )
        logger.info("unforget: %s", stats.last_event)

    def _clef_state(self, snap: _Snapshot) -> dict[str, Any]:
        return {
            "task": CLEF_TASK,
            "budget": {
                "estimated_tokens": snap.before,
                "token_budget": self.token_budget,
                "target_tokens": int(self.token_budget * self.target_ratio),
            },
            "latest_user_message": clip(snap.latest_user, 400),
            "current_facts": [{"key": f.key, "value": f.value} for f in snap.current_facts],
            "turns": [
                {
                    "id": _qid(t),
                    "turn": t.number,
                    "roles": " > ".join(m.role for m in t.messages),
                    "tokens": t.tokens,
                    "has_tool_output": t.has_tool_output,
                    "tools": t.tool_names,
                    "text": clip(render_turn(t.messages), self.clef_chars_per_turn),
                }
                for t in snap.batch
            ],
        }

    async def _write(
        self, current: list[Fact], plan: Plan
    ) -> tuple[dict[int, str], list[ExtractedFact]]:
        """Run the batched compress and pin calls (concurrently when both are needed)."""

        def texts(action: Action) -> list[TurnText]:
            return [
                TurnText(t.number, clip(render_turn(t.messages), self.writer_chars_per_turn))
                for t, _, a, _ in plan
                if a == action
            ]

        compress, pin = texts("compress"), texts("pin")
        notes_res, facts_res = await asyncio.gather(
            write_notes(self.llm, compress),
            extract_facts(self.llm, pin, current),
            return_exceptions=True,
        )
        for res in (notes_res, facts_res):
            if isinstance(res, BaseException):
                raise res
        return notes_res, facts_res  # type: ignore[return-value]

    def _reason(self, answer: ChoiceAnswer, applied: bool) -> str:
        if applied:
            return (
                f"Clef chose {answer.choice} with p={answer.probability:.2f} "
                f">= {self.min_confidence:.2f}"
            )
        return (
            f"Clef leaned {answer.choice} with p={answer.probability:.2f} "
            f"< {self.min_confidence:.2f}; kept"
        )

    def _fail(self, state: S, message: str) -> S:
        logger.warning("unforget: %s; context left unchanged", message)
        self._cooldown_until = time.monotonic() + self.failure_cooldown
        state.unforget_stats.last_event = f"{message}; context left unchanged."
        return state

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message, *args)

    # ------------------------------------------------------------------
    # Background jobs
    # ------------------------------------------------------------------

    def _task_manager(self) -> BackgroundTaskManager | None:
        """The running graph's task manager (as core resolves it), else the container's."""
        manager = current_task_manager.get()
        if manager is None:
            try:
                container = InjectQ.get_instance()
                # Only a bound manager: try_get alone autowires a fresh, unowned instance on
                # every call, whose tasks no graph's aclose would ever drain.
                if container.has(BackgroundTaskManager):
                    manager = container.try_get(BackgroundTaskManager)
            except Exception:
                manager = None
        if not isinstance(manager, BackgroundTaskManager):
            self._warn_once(
                "manager",
                "unforget: no BackgroundTaskManager available; trimming inline instead of in "
                "the background.",
            )
            return None
        return manager

    def _start_job(  # noqa: PLR0913
        self,
        state: S,
        thread: str,
        batch: list[Turn],
        *,
        manager: BackgroundTaskManager,
        messages: list[Any],
        before: int,
    ) -> S:
        """Snapshot the batch and start the Clef and writer calls; the context is unchanged."""
        stats = state.unforget_stats
        if not self._make_room():
            self._warn_once(
                "jobs-full",
                "unforget: %d background jobs in flight; trims are skipped until some finish.",
                len(self._jobs),
            )
            stats.last_event = "Too many background jobs in flight; trim skipped this time."
            return state

        numbers = [t.number for t in batch if t.number is not None]
        snapshot = self._snapshot(state, messages, batch, before, copy=True)
        task = manager.create_task(
            self._run_job(snapshot),
            name=f"unforget-trim:{thread}",
            timeout=self.job_timeout + _TASK_TIMEOUT_GRACE,
            context={"thread_id": thread, "turns": numbers},
        )
        if task is None:
            # Dropped by the manager's backpressure. Try again on the next call, no cooldown.
            stats.last_event = "Background task queue full; trim skipped this time."
            return state

        job = _Job(
            task=task, turns=numbers, loop=asyncio.get_running_loop(), started=time.monotonic()
        )
        task.add_done_callback(functools.partial(self._job_done, thread, job))
        self._jobs[thread] = job
        stats.background_jobs_started += 1
        stats.last_event = (
            f"Clef job started for {len(numbers)} turn(s) ({_span(numbers)}) in background "
            f"(~{before} > {self.token_budget} tok)."
        )
        logger.info("unforget: thread %s: %s", thread, stats.last_event)
        return state

    async def _run_job(self, snap: _Snapshot) -> _Outcome:
        """The background part of a trim: everything up to, not including, ``_apply``.

        ``asyncio.CancelledError`` (the graph's ``aclose`` cancelling leftover tasks) is not
        caught: the task ends cancelled and pickup discards it.
        """
        outcome = _Outcome()
        try:
            async with asyncio.timeout(self.job_timeout):
                await self._decide(snap, outcome)
        except TimeoutError:
            outcome.error = f"background job timed out after {self.job_timeout:g}s"
        return outcome

    def _job_done(self, thread: str, job: _Job, task: asyncio.Task[_Outcome]) -> None:
        """Done callback: stamp the finish time and log one line."""
        job.finished_at = time.monotonic()
        took = job.finished_at - job.started
        span = _span(job.turns)
        if task.cancelled():
            logger.info(
                "unforget: background job for thread %s (turns %s) cancelled after %.1fs",
                thread,
                span,
                took,
            )
            return
        exc = task.exception()
        error = repr(exc) if exc is not None else task.result().error
        if error is not None:
            logger.warning(
                "unforget: background job for thread %s (turns %s) failed after %.1fs: %s",
                thread,
                span,
                took,
                error,
            )
        else:
            logger.info(
                "unforget: background job for thread %s (turns %s) finished in %.1fs; "
                "applied on the next call",
                thread,
                span,
                took,
            )

    async def _wait_for(self, thread: str, job: _Job) -> None:
        """Hard ceiling: wait for the thread's running job instead of letting the context grow."""
        if job.loop is not asyncio.get_running_loop():
            # A task of another event loop cannot be awaited here. Forget it; its turns are
            # asked again.
            if self._jobs.get(thread) is job:
                del self._jobs[thread]
            logger.warning(
                "unforget: background job for thread %s runs on another event loop; dropped",
                thread,
            )
            return
        logger.info(
            "unforget: thread %s is over the hard budget (%d tok); waiting for its background "
            "job (turns %s)",
            thread,
            self.hard_budget,
            _span(job.turns),
        )
        # asyncio.wait does not cancel the job when this call is cancelled.
        await asyncio.wait({job.task})

    def _pickup(self, state: S, thread: str) -> bool:
        """Apply this thread's finished job to ``state``, if there is one. True if one was taken.

        Synchronous on purpose: from taking the job out of ``self._jobs`` to committing its
        decisions there is no await, so two calls on the same event loop cannot both apply it.
        """
        job = self._jobs.get(thread)
        if job is None or not job.task.done():
            return False
        if self._jobs.pop(thread, None) is not job:
            return False

        stats = state.unforget_stats
        task = job.task
        span = _span(job.turns)
        if task.cancelled():
            stats.background_jobs_discarded += 1
            stats.last_event = (
                f"Background Clef job for turns {span} was cancelled; discarded (those turns "
                "will be asked again)."
            )
            return True
        exc = task.exception()
        if exc is not None:
            stats.background_jobs_failed += 1
            self._fail(state, f"background job for turns {span} failed: {exc!r}")
            return True

        # Decisions are keyed by stable turn number. Apply them to the turns as they are in
        # the current context (so the original messages are archived and removed), skip
        # turns that left the context or were decided meanwhile, and never touch newer turns.
        messages = list(state.context or [])
        current = {t.number: t for t in group_turns(messages) if not t.preamble}
        decided = set(state.decided_turns)
        outcome = task.result()
        outcome.plan = [
            (current[t.number], answer, action, applied)
            for t, answer, action, applied in outcome.plan
            if t.number in current and t.number not in decided
        ]
        if outcome.error is None and not outcome.plan:
            if outcome.clef_input_tokens is not None:
                stats.clef_calls += 1
                stats.clef_input_tokens += outcome.clef_input_tokens
            stats.background_jobs_discarded += 1
            stats.last_event = (
                f"Background decisions for turns {span} are stale (those turns left the "
                "context or were already decided); discarded."
            )
            logger.info("unforget: thread %s: %s", thread, stats.last_event)
            return True

        before = estimate_tokens(messages) + estimate_text_tokens(state.context_summary)
        if self._commit(state, messages, outcome, before=before, background=True):
            stats.background_jobs_applied += 1
        else:
            stats.background_jobs_failed += 1
        return True

    def _make_room(self) -> bool:
        """Keep at most ``_MAX_JOBS`` jobs; evict the oldest finished one when full."""
        if len(self._jobs) < _MAX_JOBS:
            return True
        finished = [(j.finished_at, t) for t, j in self._jobs.items() if j.finished_at is not None]
        if not finished:
            return False
        _, thread = min(finished)
        del self._jobs[thread]
        logger.info("unforget: evicted the unused background result for thread %s", thread)
        return True

    def _sweep_jobs(self) -> None:
        """Leak guard: forget results nobody picked up and jobs that can no longer finish."""
        now = time.monotonic()
        for thread, job in list(self._jobs.items()):
            if job.finished_at is not None:
                expired = now - job.finished_at > _RESULT_TTL
            elif job.task.done():
                expired = False  # the done callback has not run yet
            else:
                expired = job.loop.is_closed() or (
                    now - job.started > self.job_timeout + _TASK_TIMEOUT_GRACE + _RESULT_TTL
                )
            if expired:
                del self._jobs[thread]
                logger.info(
                    "unforget: dropped the unused background job for thread %s (turns %s)",
                    thread,
                    _span(job.turns),
                )


def _qid(turn: Turn) -> str:
    return f"turn_{turn.number}"


def _thread_of(state: UnforgetState) -> str | None:
    """The thread id core stamps on the state; None means background mode is not possible."""
    thread = getattr(getattr(state, "execution_meta", None), "thread_id", None)
    return str(thread) if thread not in (None, "") else None


def _span(numbers: Iterable[int | None]) -> str:
    """Turn numbers as compact ranges, for example ``3-5, 8``."""
    ordered = sorted({n for n in numbers if n is not None})
    parts: list[str] = []
    start = prev = None
    for n in ordered:
        if prev is not None and n == prev + 1:
            prev = n
            continue
        if start is not None:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    if start is not None:
        parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def _apply_facts(
    ledger: list[Fact], turn: int, archive_id: int | None, facts: list[ExtractedFact]
) -> int:
    """Add this turn's facts; mark any current fact they replace as superseded (kept)."""
    added = 0
    for fact in (f for f in facts if f.turn == turn):
        replaced = {fact.key, *fact.replaces}
        for old in ledger:
            if old.status == "current" and old.key in replaced:
                old.status = "superseded"
                old.superseded_by_turn = turn
        ledger.append(Fact(key=fact.key, value=fact.value, turn=turn, archive_id=archive_id))
        added += 1
    return added
