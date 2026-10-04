"""No-provider regression for automatic Kanban claim budgets on legacy boards."""

from __future__ import annotations

import json
import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _pre_ceiling_connection(path):
    schema = kb.SCHEMA_SQL.replace(
        "    automatic_attempts INTEGER NOT NULL DEFAULT 0,\n", "",
    )
    assert schema != kb.SCHEMA_SQL
    old = sqlite3.connect(path)
    old.executescript(schema)
    return old


def _drift_event_ids(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE task_events;
        CREATE TABLE task_events (
            id TEXT PRIMARY KEY, task_id TEXT NOT NULL, run_id TEXT,
            kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL
        );
    """)


def _drift_run_ids(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE task_runs;
        CREATE TABLE task_runs (
            id TEXT PRIMARY KEY, task_id TEXT NOT NULL, status TEXT NOT NULL,
            outcome TEXT, started_at INTEGER NOT NULL, ended_at INTEGER
        );
    """)


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_migration_preserves_spent_claims_in_current_phase(tmp_path, lane):
    """A pre-ceiling quota run cannot become a free extra dispatch on upgrade."""
    db_path = tmp_path / "pre-ceiling.db"
    with _pre_ceiling_connection(db_path) as old:
        old.execute(
            "INSERT INTO tasks (id, title, status, created_at, max_retries) "
            "VALUES ('old', 'quota task', ?, 1, 1)", (lane,),
        )
        if lane == "review":
            prior_run = old.execute(
                "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
                "VALUES ('old', 'review_requested', 'review_requested', 2, 3)",
            ).lastrowid
            old.execute(
                "INSERT INTO task_events (task_id, run_id, kind, created_at) "
                "VALUES ('old', ?, 'claimed', 2)", (prior_run,),
            )
            old.execute(
                "INSERT INTO task_events (task_id, run_id, kind, created_at) "
                "VALUES ('old', ?, 'review_requested', 3)", (prior_run,),
            )
        run_id = old.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
            "VALUES ('old', 'rate_limited', 'rate_limited', 4, 5)",
        ).lastrowid
        old.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES ('old', ?, 'claimed', ?, 4)",
            (run_id, json.dumps({"source_status": lane})),
        )
    old.close()

    with kbc.connect(db_path) as migrated:
        attempts = migrated.execute(
            "SELECT automatic_attempts FROM tasks WHERE id='old'",
        ).fetchone()["automatic_attempts"]
        assert attempts == 1
        claim = kb.claim_review_task if lane == "review" else kb.claim_task
        old_runs = len(kb.list_runs(migrated, "old"))
        assert claim(migrated, "old") is None
        task = kb.get_task(migrated, "old")
        assert task is not None and task.status == "blocked"
        assert len(kb.list_runs(migrated, "old")) == old_runs
        assert kb.unblock_task(migrated, "old")
        task = kb.get_task(migrated, "old")
        assert task is not None and task.status == lane
        assert claim(migrated, "old") is not None
        assert len(kb.list_runs(migrated, "old")) == old_runs + 1


def test_migration_charges_inflight_claim_without_legacy_claim_event(tmp_path):
    """A legacy running worker with only run evidence already used a claim."""
    db_path = tmp_path / "pre-ceiling-inflight.db"
    with _pre_ceiling_connection(db_path) as old:
        old.execute(
            "INSERT INTO tasks (id, title, status, created_at, max_retries, claim_lock) "
            "VALUES ('running', 'old worker', 'running', 1, 1, 'host:worker')",
        )
        run_id = old.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, started_at) "
            "VALUES ('running', 'running', 'host:worker', 2)",
        ).lastrowid
        old.execute(
            "UPDATE tasks SET current_run_id=? WHERE id='running'", (run_id,),
        )
    old.close()

    with kbc.connect(db_path) as migrated:
        row = migrated.execute(
            "SELECT status, current_run_id, automatic_attempts "
            "FROM tasks WHERE id='running'",
        ).fetchone()
        assert (row["status"], row["current_run_id"], row["automatic_attempts"]) == (
            "running", run_id, 1,
        )
        assert len(kb.list_runs(migrated, "running")) == 1


def test_migration_charges_nonnumeric_legacy_claim_without_runs(tmp_path):
    """A TEXT claim ID must not grant a fresh budget on upgrade."""
    db_path = tmp_path / "text-claim.db"
    with _pre_ceiling_connection(db_path) as old:
        _drift_event_ids(old)
        old.execute(
            "INSERT INTO tasks (id, title, status, created_at, max_retries) "
            "VALUES ('old', 'legacy task', 'ready', 1, 1)",
        )
        old.execute(
            "INSERT INTO task_events (id, task_id, kind, created_at) "
            "VALUES ('claim-old', 'old', 'claimed', 2)",
        )
    with kbc.connect(db_path) as migrated:
        attempts = migrated.execute(
            "SELECT automatic_attempts FROM tasks WHERE id='old'",
        ).fetchone()["automatic_attempts"]
        assert attempts == 1
        assert kb.list_runs(migrated, "old") == []
        assert kb.claim_task(migrated, "old") is None
        task = kb.get_task(migrated, "old")
        assert task is not None and task.status == "blocked"


def test_migration_resets_after_null_legacy_event_id(tmp_path):
    """A nullable TEXT phase marker separates old claims from the current one."""
    db_path = tmp_path / "null-reset.db"
    with _pre_ceiling_connection(db_path) as old:
        _drift_event_ids(old)
        old.execute(
            "INSERT INTO tasks (id, title, status, created_at, max_retries) "
            "VALUES ('old', 'legacy review', 'review', 1, 1)",
        )
        for event_id, kind, stamp in (
            ("before", "claimed", 2),
            (None, "review_requested", 3),
            ("after", "claimed", 4),
        ):
            old.execute(
                "INSERT INTO task_events (id, task_id, kind, created_at) "
                "VALUES (?, 'old', ?, ?)",
                (event_id, kind, stamp),
            )
    with kbc.connect(db_path) as migrated:
        assert migrated.execute(
            "SELECT automatic_attempts FROM tasks WHERE id='old'",
        ).fetchone()[0] == 1
        assert kb.claim_review_task(migrated, "old") is None
        task = kb.get_task(migrated, "old")
        assert task is not None and task.status == "blocked"
        events = migrated.execute(
            "SELECT kind FROM task_events WHERE task_id='old' ORDER BY id",
        ).fetchall()
        assert [event["kind"] for event in events[:3]] == [
            "claimed", "review_requested", "claimed",
        ]


def test_migration_does_not_charge_run_before_same_second_unblock(tmp_path):
    """An explicit unblock starts a fresh phase even within one clock tick."""
    db_path = tmp_path / "same-second-unblock.db"
    with _pre_ceiling_connection(db_path) as old:
        old.execute(
            "INSERT INTO tasks (id, title, status, created_at, max_retries) "
            "VALUES ('old', 'resumed task', 'ready', 1, 1)",
        )
        run_id = old.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
            "VALUES ('old', 'rate_limited', 'rate_limited', 50, 50)",
        ).lastrowid
        old.execute(
            "INSERT INTO task_events (task_id, run_id, kind, created_at) "
            "VALUES ('old', ?, 'claimed', 50)", (run_id,),
        )
        old.execute(
            "INSERT INTO task_events (task_id, run_id, kind, created_at) "
            "VALUES ('old', ?, 'gave_up', 50)", (run_id,),
        )
        old.execute(
            "INSERT INTO task_events (task_id, kind, created_at) "
            "VALUES ('old', 'unblocked', 50)",
        )
    with kbc.connect(db_path) as migrated:
        assert migrated.execute(
            "SELECT automatic_attempts FROM tasks WHERE id='old'",
        ).fetchone()[0] == 0
        assert kb.claim_task(migrated, "old") is not None
        assert len(kb.list_runs(migrated, "old")) == 2


def test_migration_counts_text_run_ids_after_same_second_review_reset(tmp_path):
    """Legacy review claims remain charged after the TEXT-ID tables rebuild."""
    db_path = tmp_path / "text-runs-review.db"
    with _pre_ceiling_connection(db_path) as old:
        _drift_event_ids(old)
        _drift_run_ids(old)
        old.execute(
            "INSERT INTO tasks (id, title, status, created_at, max_retries) "
            "VALUES ('old', 'review task', 'review', 1, 1)",
        )
        old.execute(
            "INSERT INTO task_runs (id, task_id, status, outcome, started_at, ended_at) "
            "VALUES ('implementer', 'old', 'review_requested', 'review_requested', 3, 3)",
        )
        old.execute(
            "INSERT INTO task_events (id, task_id, run_id, kind, created_at) "
            "VALUES ('before', 'old', 'implementer', 'claimed', 3)",
        )
        old.execute(
            "INSERT INTO task_events (id, task_id, run_id, kind, created_at) "
            "VALUES ('reset', 'old', 'implementer', 'review_requested', 3)",
        )
        old.execute(
            "INSERT INTO task_runs (id, task_id, status, outcome, started_at, ended_at) "
            "VALUES ('reviewer', 'old', 'rate_limited', 'rate_limited', 3, 4)",
        )
        old.execute(
            "INSERT INTO task_events (id, task_id, run_id, kind, created_at) "
            "VALUES ('after', 'old', 'reviewer', 'claimed', 3)",
        )
    with kbc.connect(db_path) as migrated:
        assert migrated.execute(
            "SELECT automatic_attempts FROM tasks WHERE id='old'",
        ).fetchone()[0] == 1
        assert len(kb.list_runs(migrated, "old")) == 2
        assert kb.claim_review_task(migrated, "old") is None
        task = kb.get_task(migrated, "old")
        assert task is not None and task.status == "blocked"
