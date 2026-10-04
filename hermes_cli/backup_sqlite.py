"""WAL-safe SQLite snapshots. Direct execution needs only the standard library.

Desktop invokes this file before stopping its backend, even when application
imports cannot load. Full and quick backups use the same SQLite copy operation.
"""
import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class _SQLiteBackupTimeout(RuntimeError):
    """Raised when a SQLite snapshot exceeds its busy or total deadline."""


def _close_quietly(conn: Optional[sqlite3.Connection]) -> None:
    if conn is not None:
        with suppress(Exception):
            conn.close()


def _safe_copy_db(
    src: Path, dst: Path, *, timeout_seconds: float = 10.0,
    max_duration_seconds: Optional[float] = None,
) -> bool:
    """Copy a SQLite database with the backup() API (WAL-safe consistent snapshot).

    WAL sources are pinned at the first read; later commits stay in the live
    source, not this snapshot. The read transaction can retain WAL frames until
    close. Non-WAL sources retain incremental locking to avoid blocking writers.
    No total deadline applies unless max_duration_seconds is explicitly supplied.
    That deadline is checked between SQLite calls/backup steps, not an I/O
    interrupt. Fails closed: copying only the main file would lose WAL data.
    """
    # This must be outside destination cleanup: an alias is not staging that
    # we own. stat follows symlinks and compares filesystem identity, covering
    # lexical aliases, case aliases on case-insensitive volumes, and hardlinks
    # without lowercasing paths or probing the filesystem with writes.
    # Callers must keep source/destination path bindings stable during the copy;
    # this preflight is not protection against hostile concurrent path swaps.
    try:
        source_stat = src.stat()
        try:
            destination_stat = dst.stat()
        except FileNotFoundError:
            destination_stat = None
        if destination_stat is not None and os.path.samestat(source_stat, destination_stat):
            logger.warning("SQLite safe copy refused source/destination identity for %s", src)
            return False
    except (OSError, ValueError) as exc:
        # Missing source or uncertain identity: no chmod, open, or unlink.
        logger.warning("SQLite safe copy could not establish file identity for %s: %s", src, exc)
        return False

    conn = backup_conn = None
    try:
        started = time.monotonic()
        duration = None if max_duration_seconds is None else max(0.0, max_duration_seconds)
        deadline = None if duration is None else started + duration
        # sqlite3.connect() creates a missing destination with the process
        # umask, which is commonly 0022 (0644).  Snapshot databases contain
        # session and tool state, so create the inode owner-only before SQLite
        # writes its first byte.  O_NOFOLLOW also refuses a planted symlink on
        # platforms that support it.  Tighten an existing internal staging
        # file as well (NamedTemporaryFile callers already create it 0600).
        if os.name != "nt":
            open_flags = os.O_WRONLY | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                open_flags |= os.O_NOFOLLOW
            secure_fd = os.open(dst, open_flags, 0o600)
            try:
                os.fchmod(secure_fd, 0o600)
            finally:
                os.close(secure_fd)
        # timeout=0.0 disables sqlite3's implicit busy wait so the progress callback owns the
        # full locked-source deadline instead of adding the default timeout before each callback.
        conn = sqlite3.connect(f"{src.resolve().as_uri()}?mode=ro", uri=True, timeout=0.0)
        backup_conn = sqlite3.connect(str(dst), timeout=0.0)
        busy_deadline = started + max(0.0, timeout_seconds)

        def _check_backup_progress(status: int, _remaining: int, _total: int) -> None:
            nonlocal busy_deadline
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise _SQLiteBackupTimeout(f"database copy exceeded {duration:g} seconds")
            if status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                if now >= busy_deadline:
                    raise _SQLiteBackupTimeout(f"database remained locked for {timeout_seconds:g} seconds")
            else:
                busy_deadline = now + max(0.0, timeout_seconds)

        # BEGIN alone is deferred: the read is required to establish a snapshot.
        # Keep it only in WAL mode, where readers do not block writers. Checking
        # journal_mode inside the read transaction also prevents a mode-change race.
        while True:
            try:
                conn.execute("BEGIN")
                conn.execute("SELECT count(*) FROM sqlite_schema").fetchone()
                if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                    conn.rollback()
                break
            except sqlite3.OperationalError as exc:
                conn.rollback()
                code = getattr(exc, "sqlite_errorcode", 0) & 0xff
                if code not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                    raise
                _check_backup_progress(code, 0, 0)
                time.sleep(0.1)

        conn.backup(backup_conn, pages=256, progress=_check_backup_progress, sleep=0.1)
        return True
    except Exception as exc:
        logger.warning("SQLite safe copy failed for %s: %s", src, exc)
        # Windows won't remove the partial destination while SQLite still has it open.
        _close_quietly(backup_conn)
        backup_conn = None
        with suppress(OSError):
            dst.unlink(missing_ok=True)
        return False
    finally:
        _close_quietly(backup_conn)
        _close_quietly(conn)


def preflight_state_db(home: Path) -> dict:
    """Publish an emergency snapshot; do not prune recovery files on failure."""
    source = home / "state.db"
    if not source.exists():
        return {"path": None, "message": "state.db not found (fresh install?)"}
    prefix = "state.db.pre-update-emergency-"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    destination = home / f"{prefix}{stamp}-{os.getpid()}.bak"
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".partial", dir=home)
    os.close(fd)
    staged = Path(name)
    try:
        if not _safe_copy_db(source, staged):
            raise RuntimeError("SQLite safe copy failed; previous emergency snapshots were retained")
        connection = sqlite3.connect(str(staged))
        try:
            result = connection.execute("PRAGMA quick_check").fetchall()
            if result != [("ok",)]:
                raise RuntimeError(f"SQLite snapshot integrity check failed: {result}")
        finally:
            connection.close()
        size = staged.stat().st_size
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
    for old in sorted(home.glob(f"{prefix}*.bak"), reverse=True)[2:]:
        try:
            old.unlink()
        except OSError as exc:
            logger.warning("Could not prune emergency snapshot %s: %s", old, exc)
    return {"path": str(destination), "bytes": size}


if __name__ == "__main__":
    print(json.dumps(preflight_state_db(Path(sys.argv[1]))))
