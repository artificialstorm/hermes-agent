"""No-provider dispatcher checks for persisted automatic-claim ceilings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def test_dispatch_reports_claim_side_exhaustion_and_dry_run_does_not_spawn(
    tmp_path, monkeypatch, all_assignees_spawnable,
):
    """A prior infrastructure refusal used a claim, even if it was not a failure."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    with kbc.connect(tmp_path / "attempts.db") as conn:
        tid = kb.create_task(conn, title="exhausted claim", assignee="a", max_retries=1)
        assert kb.claim_task(conn, tid) is not None
        assert not kbd._record_task_failure(
            conn, tid, "host cannot place scoped worker", outcome="spawn_failed",
            release_claim=True, end_run=True, infrastructure=True,
        )
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "ready"
        assert [r.outcome for r in kb.list_runs(conn, tid)] == ["spawn_failed"]

        def no_provider_spawn(*_args, **_kwargs):
            raise AssertionError("exhausted task must never launch a model worker")

        dry = kbd.dispatch_once(
            conn, dry_run=True, spawn_fn=no_provider_spawn, max_in_progress=2,
        )
        assert dry.spawned == []
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "ready"
        assert len(kb.list_runs(conn, tid)) == 1

        actual = kbd.dispatch_once(
            conn, spawn_fn=no_provider_spawn, max_in_progress=2,
        )
        assert actual.spawned == []
        assert tid in actual.auto_blocked
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "blocked"
        assert [r.outcome for r in kb.list_runs(conn, tid)] == ["spawn_failed"]
        assert len([e for e in kb.list_events(conn, tid) if e.kind == "gave_up"]) == 1


def test_competing_connections_cannot_claim_past_the_same_budget(tmp_path):
    """Distinct dispatcher connections serialize the last claim and its exhausted block."""
    db_path = tmp_path / "competing.db"
    with kbc.connect_closing(db_path) as conn:
        tid = kb.create_task(conn, title="competing claims", assignee="a", max_retries=1)

        def simultaneous_claims():
            barrier = Barrier(3)

            def one_claim():
                with kbc.connect_closing(db_path) as competitor:
                    barrier.wait(timeout=5)
                    return kb.claim_task(competitor, tid) is not None

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(one_claim) for _ in range(2)]
                barrier.wait(timeout=5)
                return [f.result(timeout=10) for f in futures]

        assert sorted(simultaneous_claims()) == [False, True]
        assert len(kb.list_runs(conn, tid)) == 1
        assert not kbd._record_task_failure(
            conn, tid, "host cannot place scoped worker", outcome="spawn_failed",
            release_claim=True, end_run=True, infrastructure=True,
        )
        assert simultaneous_claims() == [False, False]
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "blocked"
        assert len(kb.list_runs(conn, tid)) == 1
        assert len([e for e in kb.list_events(conn, tid) if e.kind == "gave_up"]) == 1
