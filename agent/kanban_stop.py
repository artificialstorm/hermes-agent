"""Turn-end guard for kanban workers, which must end with a terminal board tool that hands
the card to whoever owns it next (``kanban_complete``, ``kanban_block``,
``kanban_request_review``, ``kanban_request_changes``). Some models narrate the next step
and stop with no tool calls; Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``. Policy-only: return a bounded synthetic nudge so the loop continues
instead of exiting.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional

from agent.delegation_context import owned_kanban_task


# Every tool that ends this worker's responsibility for the card, not just the two that
# close it out: ``kanban_request_review`` moves it to ``review`` (goals.py's continuation /
# finalize prompts tell builders to call it) and ``kanban_request_changes`` returns it to
# ``ready`` (the sdlc-review skill tells reviewers to). Nudging after either asks a worker
# that did the right thing to ``kanban_complete`` a card it must not close.
_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_block",
    "kanban_schedule",
    "kanban_request_review",
    "kanban_request_changes",
})

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set for the dispatcher-owned worker, unless
    ``HERMES_KANBAN_STOP_NUDGE`` disables it. In-process delegate_task children and cron runs
    inherit the env var but own no board task and carry no kanban toolset."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(owned_kanban_task())


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True only for a successful terminal result, never a requested/denied call.

    Results normally omit the tool name, so correlate by call id. A named result
    can survive transcript compaction without its assistant call. Unknown results
    remain unconfirmed: the nudge asks for board readback before another mutation.
    """
    calls: dict[str, str] = {}
    owned = owned_kanban_task()
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if cid:
                    calls[str(cid)] = _tool_call_name(tc)
            continue
        if role != "tool":
            continue
        name = calls.get(str(msg.get("tool_call_id") or ""), str(msg.get("name") or ""))
        if name not in _TERMINAL_KANBAN_TOOLS:
            continue
        result = msg.get("content")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (ValueError, TypeError):
                continue
        if not isinstance(result, dict) or result.get("ok") is not True:
            continue
        if result.get("error") or result.get("success") is False:
            continue
        if owned and result.get("task_id") not in (None, owned):
            continue
        status = result.get("status")
        # Completion responses may omit status. Supplied malformed/unknown values
        # are unconfirmed, not exceptions that disable the bounded nudge.
        if "status" in result and (
            not isinstance(status, str)
            or status not in {"todo", "ready", "review", "scheduled", "blocked", "done"}
        ):
            continue
        if status == "todo" and not (
            name == "kanban_block" and result.get("block_kind") == "dependency"
        ):
            continue
        if not msg.get("is_error"):
            return True
    return False


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    # Unknown responses require readback, not a blind duplicate terminal mutation.
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` has no confirmed successful terminal board result in this session. "
        "A requested, denied, failed or ambiguous call is not proof of handoff. "
        "First call `kanban_show` to read back the card. If it has already left your "
        "owned running state (including todo, review, ready, scheduled, blocked or done), "
        "do not repeat the transition or close a card handed to a reviewer. "
        "Only if it is still running and owned by this run, continue below.\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work is done "
        "and needs no review, `kanban_request_review(summary=...)` if it is a code "
        "change that needs same-card review, OR `kanban_block(reason=...)` if you are "
        "blocked. Reviewers approve with `kanban_complete` or send the card back with "
        "`kanban_request_changes(reason=...)`.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "session_called_kanban_terminal"]
