# context-triage: notes for the README writer

Tagline: "Context triage for LLM agents: a small decision model decides what to keep, compress, archive, drop or pin."

Status: code complete, lint clean (`ruff check`, `ruff format --check`). Not tested: no test
suite, no live run with real keys, `10xgraph dev` not started. The user tests it.

## Layout

| Path | What |
|---|---|
| `unforget/manager.py` | `UnforgetContextManager` (subclass of 10xGraph `BaseContextManager`) |
| `unforget/clef.py` | `ClefClient` (httpx, async) and response parsing |
| `unforget/writer.py` | Batched cheap-LLM calls: compress notes and pinned facts (`call_llm`) |
| `unforget/spans.py` | Turn grouping, stable turn numbers, token estimate, labels |
| `unforget/render.py` | Builds the single `context_summary` block |
| `unforget/recall.py` | The `recall(archive_id)` tool |
| `unforget/models.py` | `UnforgetState` and the state models (ledger, notes, archive, log, stats) |
| `graph/agent.py` | ReactAgent wiring, SQLite checkpointer, demo knob values |
| `graph/tools.py` | Two demo tools with long-ish output (`read_service_log`, `list_project_files`) |
| `graph/state.py` | `UnforgetDemoState(UnforgetState)` |
| `sync.sh` | Builds `.venv` (uv, Python 3.13) from the local core and API folders |

Run: `./sync.sh`, copy `.env.example` to `.env` and fill it, then `.venv/bin/10xgraph dev`.
Threads persist in `unforget.db` (delete it to start over).

## Clef facts (verified 2026-11-01)

Sources: https://developers.cloudflare.com/workers-ai/models/clef/ , its
`schema-input.json` and `schema-output.json`, and https://huggingface.co/Cloudflare/clef .

- REST: `POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef`,
  `Authorization: Bearer <token>`.
- Body: `{"model": "clef", "state": <string or JSON>, "questions": {...}}`.
  The REST endpoint accepts only `clef`; `clef-flash` returned HTTP 422 Unsupported model in a live call.
- 1 to 64 questions; ids use letters, digits, `_`, `.`, `-` (max 100 chars).
- Question types: `noul` (yes/no), `choice`, `score`.
- `choice` criteria: an object mapping option id to its description (string, object, array or
  null), 2 to 255 options.
- `choice` answer: `{"type": "choice", "choice": <top option>, "probabilities": {option: p},
  "confidence": <0..1>}`; probabilities sum to 1.
- Response: `{"model", "answers": {question_id: answer}, "usage": {"input_tokens",
  "output_tokens"}}`.
- Context 65,536 tokens; long text state is truncated. $0.24 per million input tokens, no
  output charge.
- Clef is a 27B multimodal model; Clef-Flash is the smaller, faster variant. License on
  Hugging Face: Apache-2.0.
- Not documented: whether the REST reply is wrapped in Cloudflare's usual
  `{"result": ..., "success": ...}` envelope. The parser accepts both shapes.

## Design decisions

- Span = one whole turn (user message through the final reply, tool calls and results
  included). Removing whole turns keeps role alternation and tool call/result pairs intact.
  System messages and the last `keep_recent_turns` turns are never candidates.
- Stable turn numbers: each turn's first message gets `metadata["unforget_turn"] = n` the first
  time unforget sees it. Numbers survive checkpointing and do not shift when earlier turns are
  removed. Hysteresis uses them: `state.decided_turns` lists every turn Clef already decided,
  and those are never sent again.
- Trigger: only when estimated tokens (context plus the summary block, about 4 chars per token)
  exceed `token_budget`. A trim picks the oldest undecided eligible turns until removing them
  would reach `target_ratio * token_budget` (max 64), and makes one Clef call. If the context
  is over budget and every older turn is decided, it logs once and does nothing.
- Clef state: task text, budget numbers, the latest user message, current ledger facts, and
  per candidate turn: id, number, role sequence, token estimate, tool names, has_tool_output,
  text clipped to 800 chars. One `choice` question per turn with options keep / compress /
  archive / drop / pin, each described in plain words.
- Confidence gate: the top option applies only if its probability is >= `min_confidence`
  (0.7); otherwise the turn is kept. All probabilities are recorded in the decision log.
- Actions: keep (unchanged); drop (removed, not archived); archive (removed, stored verbatim in
  `state.archive[id]` with a label made from the first user words, tools and token count);
  compress (cheap LLM writes a 1-3 sentence note, turn removed and archived); pin (cheap LLM
  extracts `{key, value, replaces}` facts given the current ledger; replaced facts are marked
  `superseded` with the turn number, never deleted; turn removed and archived).
- One writer call for all compress turns and one for all pin turns per trim (run
  concurrently). A compress turn the writer skipped falls back to archive.
- Rendering: one text block in `state.context_summary` with current facts, notes in turn order,
  and the archive index `#id: turn N, label` plus the instruction to call `recall`. 10xGraph
  injects `context_summary` into the prompt ahead of the kept messages.
- `recall(archive_id)` returns the archived turn as text. It does not re-insert messages.
- Fail-safe: Clef errors, writer errors, missing credentials and malformed replies leave the
  context unchanged and log a warning. Decisions are applied to copies and committed at the
  end, so a failure mid-apply cannot leave half-written state. After a failure, trims are
  skipped for `failure_cooldown` seconds (60) so an outage does not add a timeout to every
  model call. Credentials are read at call time; the graph imports without any keys.
- Background mode (`background=True`, the default). 10xGraph awaits `atrim_context` before
  every model call (`Agent._trim_context`) and when it saves the state (`sync_data(trim=True)`
  at run end, error and stop), so an inline trim makes the user wait for one Clef call plus up
  to two writer calls. In background mode a trim never awaits them:
  1. If this thread has a finished job, its decisions are applied now (synchronous, no
     network), through the same copy-then-commit path as inline mode.
  2. Else, if over `token_budget` and no job runs for the thread, the picked turns, the latest
     user message and the current facts are deep-copied and the Clef + writer calls start as a
     task on the graph's `BackgroundTaskManager` (the run's own manager via
     `current_task_manager`, else the one in the InjectQ container). The context is returned
     unchanged.
  3. The call returns at once. One job per thread; while it runs, later calls only update
     `last_event`.
  - Job key: `state.execution_meta.thread_id`. No thread id, no task manager, or the sync
    `trim_context` (its temporary loop cannot host a task that outlives it) means inline.
  - Hard ceiling: past `hard_budget_ratio * token_budget` (2x) a call waits for the running job
    and applies it, or trims inline when none is running, so one long run cannot grow the
    context without bound.
  - Stale results: decisions are keyed by stable turn number and applied to the turns as they
    are in the current context (the original messages are archived). Turns that left the
    context or were decided meanwhile are skipped; turns added after the snapshot are never
    touched. Turns of a running job are not written to `decided_turns` until applied, so a
    restart simply asks them again.
  - Failures inside the job (Clef, writer, `job_timeout` of 120 s) are recorded on the job and
    handled at pickup like an inline failure: context unchanged, cooldown, `last_event`. A job
    cancelled by the graph's `aclose` is discarded at pickup with no cooldown. If the task
    manager drops the task (backpressure), the trim is skipped with no cooldown.
  - Leak guard: a finished job not picked up within 10 minutes is forgotten, at most 256 jobs
    are held (the oldest finished one is evicted first), and the sweep runs on every call.
  - Trade-offs: the budget becomes soft for one or two model calls (the context stays over
    budget until the job finishes); results live in process memory, so they are lost on a
    restart and not shared across workers (another worker re-asks the turns); the sync path
    stays inline. `background=False` keeps the old inline behaviour exactly.
- Dropped turns leave the model context only. The checkpointer also stores each thread's
  messages separately from `state.context`, so they remain in the thread's message history.
- `graph/agent.py` adds a user-role "Today Date is ..." entry to the system prompt, as
  weatherbot does. This keeps the first non-system message a user turn when the summary
  (sent by 10xGraph as an assistant message) is present.

## Knobs (`UnforgetContextManager`)

| Knob | Default | Demo value in `graph/agent.py` |
|---|---|---|
| `token_budget` | 8000 | 3000 |
| `target_ratio` | 0.6 | 0.6 |
| `keep_recent_turns` | 4 (min 1) | 4 |
| `min_confidence` | 0.7 | 0.7 |
| `clef_model` | `"clef"` | `"clef"` |
| `writer_model` | `"claude-haiku-5-5"` | `"claude-haiku-5-5"` |
| `max_questions` | 64 | default |
| `clef_chars_per_turn` | 800 | default |
| `writer_chars_per_turn` | 6000 | default |
| `failure_cooldown` | 60.0 s | default |
| `background` | `True` | `False` (off for the first live test) |
| `hard_budget_ratio` | 2.0 (must be > 1) | 2.0 |
| `job_timeout` | 120.0 s | default |
| `clef_client`, `llm` | real clients | default (swap for fakes) |

Main agent model: `claude-sonnet-5-5` (10xGraph detects `claude-*` as Anthropic).

## Seeing decisions in the playground

`10xgraph dev` opens the playground. In the chat view, open the Inspector and pick the
**State** tab, then press **Sync** after a reply (it fetches `GET /v1/threads/{id}/state`).
It shows:

- **Context**: the messages the model still sees (shrinks after a trim).
- **Context Summary**: the rendered facts, notes and archive index.
- **Dynamic fields**: `decision_log` (one entry per decided turn: turn, suggested, action,
  probability, probabilities, applied, reason, label, tokens, tokens_freed, archive_id,
  background), `ledger`, `notes`, `archive`, `decided_turns` and `unforget_stats`
  (context_tokens, clef_calls, clef_input_tokens, tokens_freed, last_tokens_before/after,
  last_event, background_jobs_started/applied/failed/discarded).

The Threads page shows the same state as raw JSON. Server logs (`unforget` logger) print one
line per decision, a summary line per trim, and one line per background job start and finish.
In background mode the decisions show up one reply later than the trim that started them:
`last_event` reads "Clef job started for N turn(s) ..." first, then "applied background
decisions for turns ...".

A demo flow that triggers trims with the 3000-token budget: state a preference ("I prefer
answers as bullet lists", "we deploy on Fridays only"), ask for a few service logs and a file
listing, chat a little, then ask about a detail from an early log. The model should call
`recall`.

## For the README writer

- `pyproject.toml` has no `readme` field because README.md did not exist yet; add
  `readme = "README.md"` once it does.
- Cloudflare token needs Workers AI access on the account.

## Framework issues found (not fixed; agentflow/ untouched)

1. `call_llm` always sends `temperature` (default 0.3) unless the model is in
   `ANTHROPIC_NO_SAMPLING_MODELS` (`tenxgraph/core/graph/agent_internal/constants.py`). That set
   lacks `claude-opus-5-5`, `claude-sonnet-5-5` and `claude-haiku-5-5`, which reject
   non-default sampling values with a 400. `SummaryContextManager` with `claude-haiku-5-5` would
   fail for the same reason. Workaround here: the writer passes `temperature=1.0` (the API
   default). The agent path is unaffected (it sends temperature only when asked).
2. `convert_messages` injects `context_summary` as an assistant message right after the system
   prompts. With Anthropic, if no user-role entry comes before it, the first message is an
   assistant turn, which the Anthropic docs say must be a user turn. Not verified live; worked
   around in `graph/agent.py` as described above.
3. `sync_data` (`tenxgraph/core/graph/utils/utils.py`) deep-copies the state and then, when
   trimming, passes the original `state` (not the copy) to `atrim_context`. The copy is wasted
   and the trim mutates the live state. Harmless for unforget; reported for accuracy.
