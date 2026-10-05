"""Regression: a requested handoff is not a successful handoff (stdlib runner)."""
import json
import os
import unittest
from unittest.mock import patch

from agent.kanban_stop import build_kanban_stop_nudge, session_called_kanban_terminal
from agent.delegation_context import _DELEGATED_CHILD_CONTEXT, non_dispatcher_owned_context


def transcript(name="kanban_complete", result=None, call_id="call1", result_id="call1"):
    rows = [{"role": "assistant", "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": "{}"}}]}]
    if result is not None:
        rows.append({"role": "tool", "tool_call_id": result_id, "content": json.dumps(result)})
    return rows


class HandoffResults(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"HERMES_KANBAN_TASK": "t_owned", "HERMES_KANBAN_STOP_NUDGE": "1", "HERMES_DELEGATED_CHILD_CONTEXT": ""})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_denied_failed_ambiguous_and_missing_are_not_handoffs(self):
        for result in [None, {"ok": False, "error": "denied"}, {"success": False, "error": "failed"}, {}, "done", {"ok": "true"}, {"ok": True, "error": "denied"}]:
            with self.subTest(result=result):
                rows = transcript(result=result)
                self.assertFalse(session_called_kanban_terminal(rows))
                self.assertIsNotNone(build_kanban_stop_nudge(messages=rows))

    def test_real_success_shapes_for_all_terminal_tools(self):
        for name, status in [("kanban_complete", None), ("kanban_block", "blocked"), ("kanban_schedule", "scheduled"), ("kanban_request_review", "review"), ("kanban_request_changes", "ready")]:
            result = {"ok": True, "task_id": "t_owned", "run_id": "run1"}
            if status:
                result["status"] = status
            with self.subTest(name=name):
                rows = transcript(name, result)
                self.assertTrue(session_called_kanban_terminal(rows))
                self.assertIsNone(build_kanban_stop_nudge(messages=rows))

    def test_wrong_call_id_and_wrong_task_do_not_suppress(self):
        for rows in [transcript(result={"ok": True}, result_id="different"), transcript(result={"ok": True, "task_id": "t_other"}), transcript("kanban_heartbeat", {"ok": True})]:
            self.assertFalse(session_called_kanban_terminal(rows))

    def test_still_running_is_not_a_handoff(self):
        self.assertFalse(session_called_kanban_terminal(transcript(result={"ok": True, "status": "running"})))

    def test_named_tool_results_can_survive_compaction(self):
        rows = [{"role": "tool", "name": "kanban_request_review", "content": json.dumps({"ok": True, "task_id": "t_owned", "status": "review"})}]
        self.assertTrue(session_called_kanban_terminal(rows))

    def test_budget_and_child_exclusions(self):
        rows = transcript(result={"ok": False})
        self.assertIsNotNone(build_kanban_stop_nudge(messages=rows, attempts=1))
        self.assertIsNone(build_kanban_stop_nudge(messages=rows, attempts=2))
        token = _DELEGATED_CHILD_CONTEXT.set(True)
        try:
            self.assertIsNone(build_kanban_stop_nudge(messages=rows))
        finally:
            _DELEGATED_CHILD_CONTEXT.reset(token)
        with non_dispatcher_owned_context():
            self.assertIsNone(build_kanban_stop_nudge(messages=rows))

    def test_ambiguous_response_requires_readback_not_blind_retry(self):
        text = build_kanban_stop_nudge(messages=transcript(result="uncertain"))
        assert text is not None
        self.assertIn("kanban_show", text)
        self.assertIn("still", text)

    def test_failed_then_successful_retry(self):
        self.assertTrue(session_called_kanban_terminal(transcript(result={"ok": False}) + transcript(result={"ok": True}, call_id="call2", result_id="call2")))
