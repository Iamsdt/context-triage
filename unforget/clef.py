"""Minimal async client for Cloudflare's Clef decision model on Workers AI.

Clef takes a ``state`` (text or JSON) and up to 64 typed questions and returns one answer
per question with a probability per option. unforget asks one ``choice`` question per
conversation turn.

REST endpoint (POST, Bearer token)::

    https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef

Docs and schemas: https://developers.cloudflare.com/workers-ai/models/clef/
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from unforget.models import ACTIONS


logger = logging.getLogger("unforget.clef")

CLEF_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef"
CLEF_MODELS = ("clef", "clef-flash")
MAX_QUESTIONS = 64


class ClefError(RuntimeError):
    """Clef could not be reached, refused the request, or answered in an unexpected shape."""


@dataclass
class ChoiceAnswer:
    """One ``choice`` answer: the top option and the probability of every option."""

    choice: str
    probabilities: dict[str, float]
    confidence: float

    @property
    def probability(self) -> float:
        return self.probabilities.get(self.choice, 0.0)


@dataclass
class ClefResult:
    answers: dict[str, ChoiceAnswer] = field(default_factory=dict)
    input_tokens: int = 0
    model: str = ""


class DecisionClient(Protocol):
    """What the context manager needs from a Clef client (tests pass a fake)."""

    @property
    def configured(self) -> bool: ...

    async def decide(
        self, state: Any, questions: dict[str, dict[str, Any]], model: str
    ) -> ClefResult: ...


class ClefClient:
    """Calls Clef over the Cloudflare REST API with httpx.

    Args:
        account_id: Cloudflare account id. Defaults to ``CLOUDFLARE_ACCOUNT_ID``.
        api_token: API token with Workers AI access. Defaults to ``CLOUDFLARE_API_TOKEN``.
        timeout: Seconds before a request is abandoned.
        transport: Optional custom httpx transport (for example a proxy or mock transport).

    Credentials are read when a call is made, not at construction, so the graph imports
    without them.
    """

    def __init__(
        self,
        account_id: str | None = None,
        api_token: str | None = None,
        *,
        timeout: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._account_id = account_id
        self._api_token = api_token
        self.timeout = timeout
        self._transport = transport

    @property
    def account_id(self) -> str:
        return self._account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")

    @property
    def api_token(self) -> str:
        return self._api_token or os.getenv("CLOUDFLARE_API_TOKEN", "")

    @property
    def configured(self) -> bool:
        return bool(self.account_id and self.api_token)

    async def decide(
        self, state: Any, questions: dict[str, dict[str, Any]], model: str
    ) -> ClefResult:
        if not self.configured:
            raise ClefError("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are not set")
        if model not in CLEF_MODELS:
            raise ClefError(f"Unknown Clef model {model!r}; use one of {CLEF_MODELS}")
        if not 1 <= len(questions) <= MAX_QUESTIONS:
            raise ClefError(f"Clef takes 1 to {MAX_QUESTIONS} questions, got {len(questions)}")

        body = {"model": model, "state": state, "questions": questions}
        url = CLEF_URL.format(account_id=self.account_id)
        headers = {"Authorization": f"Bearer {self.api_token}"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                response = await client.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise ClefError(f"Clef request failed: {type(exc).__name__}: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ClefError(
                f"Clef returned HTTP {response.status_code} with a non-JSON body"
            ) from exc

        if response.status_code >= 400:  # noqa: PLR2004
            raise ClefError(f"Clef returned HTTP {response.status_code}: {_errors_of(payload)}")
        return parse_response(payload, list(questions))


def _errors_of(payload: Any) -> str:
    if isinstance(payload, dict) and payload.get("errors"):
        return "; ".join(str(e.get("message", e)) for e in payload["errors"] if e)
    return str(payload)[:300]


def parse_response(payload: Any, question_ids: list[str]) -> ClefResult:
    """Validate a Clef response and return its choice answers.

    Accepts both the bare output schema (``{"model", "answers", "usage"}``) and the
    Cloudflare REST envelope (``{"result": {...}, "success": true, ...}``).

    Raises:
        ClefError: when the response is not the documented shape, an answer is missing,
            or an answer names an option that was not offered.
    """
    if not isinstance(payload, dict):
        raise ClefError("Clef response is not a JSON object")
    if "answers" not in payload and isinstance(payload.get("result"), dict):
        if payload.get("success") is False:
            raise ClefError(f"Clef reported failure: {_errors_of(payload)}")
        payload = payload["result"]

    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ClefError("Clef response has no answers object")

    result = ClefResult(model=str(payload.get("model", "")))
    usage = payload.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int):
        result.input_tokens = usage["input_tokens"]

    for qid in question_ids:
        result.answers[qid] = _parse_choice(qid, answers.get(qid))
    return result


def _parse_choice(qid: str, raw: Any) -> ChoiceAnswer:
    if not isinstance(raw, dict):
        raise ClefError(f"Clef gave no answer for {qid}")
    if raw.get("type") not in (None, "choice"):
        raise ClefError(f"Clef answer for {qid} is type {raw.get('type')!r}, expected choice")
    probabilities = raw.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        raise ClefError(f"Clef answer for {qid} has no probabilities")
    try:
        probs = {str(k): float(v) for k, v in probabilities.items()}
    except (TypeError, ValueError) as exc:
        raise ClefError(f"Clef answer for {qid} has non-numeric probabilities") from exc
    unknown = set(probs) - set(ACTIONS)
    if unknown:
        raise ClefError(f"Clef answer for {qid} has unknown options {sorted(unknown)}")
    choice = raw.get("choice")
    if choice not in probs:
        choice = max(probs, key=lambda k: probs[k])
    confidence = raw.get("confidence")
    return ChoiceAnswer(
        choice=str(choice),
        probabilities=probs,
        confidence=float(confidence) if isinstance(confidence, int | float) else probs[choice],
    )
