"""Real SQLite regression; only synthetic databases, bounded writer handshakes."""
import json
from pathlib import Path
import queue
import sqlite3
import threading
import time
import pytest

from hermes_cli import backup_sqlite as subject
CONNECT = sqlite3.connect


def fixture_db(path, wal=True):
    c = CONNECT(path)
    if wal:
        c.execute('PRAGMA journal_mode=WAL')
    c.execute('CREATE TABLE payload (id INTEGER PRIMARY KEY, body BLOB)')
    c.executemany('INSERT INTO payload(body) VALUES (zeroblob(3000))', [()] * 1200)
    c.execute('CREATE TABLE epoch (value INTEGER)')
    c.execute('INSERT INTO epoch VALUES (0)')
    c.commit()
    if wal:
        c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        c.execute('UPDATE epoch SET value=1')
        c.commit()  # snapshot must include this WAL-only commit
    return c


def test_sustained_writer_snapshot(tmp_path, monkeypatch):
    src, dst = tmp_path / 'src.db', tmp_path / 'dst.db'
    keeper = fixture_db(src)
    requests, replies = queue.Queue(), queue.Queue()
    writes, durations, progress, connections = [], [], [], []
    def worker():
        c = CONNECT(src, timeout=0)
        try:
            while requests.get() is not None:
                start = time.monotonic()
                try:
                    c.execute('UPDATE epoch SET value=value+1')
                    c.commit()
                    writes.append(1)
                    durations.append(time.monotonic()-start)
                    replies.put(None)
                except Exception as exc:
                    c.rollback()
                    replies.put(exc)
        finally:
            c.close()
    thread = threading.Thread(target=worker)
    thread.start()
    wal_before = Path(str(src)+'-wal').stat().st_size
    class Observed(sqlite3.Connection):
        def backup(self, target, **kwargs):
            callback = kwargs['progress']
            def observe(status, remaining, total):
                progress.append([status, remaining, total])
                callback(status, remaining, total)
                if status == sqlite3.SQLITE_OK and len(progress) % 2 == 0:
                    requests.put(True)
                    error = replies.get(timeout=2)
                    assert error is None, repr(error)
                if len(progress) >= 40:
                    raise RuntimeError('harness ceiling: 40 backup steps without completion')
            kwargs['progress'] = observe
            return super().backup(target, **kwargs)
    def connect(*args, **kwargs):
        c = CONNECT(*args, factory=Observed, **kwargs)
        connections.append(c)
        return c
    monkeypatch.setattr(subject.sqlite3, 'connect', connect)
    start = time.monotonic()
    try:
        ok = subject._safe_copy_db(src, dst, timeout_seconds=.1)
        elapsed = time.monotonic()-start
    finally:
        requests.put(None)
        thread.join(timeout=3)
    wal_after = Path(str(src)+'-wal').stat().st_size
    live_epoch = keeper.execute('SELECT value FROM epoch').fetchone()[0]
    checkpoint = keeper.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
    keeper.close()
    assert not thread.is_alive()
    closed = []
    for c in connections:
        try:
            c.execute('SELECT 1')
            closed.append(False)
        except sqlite3.ProgrammingError:
            closed.append(True)
    report = dict(ok=ok, elapsed=elapsed, steps=len(progress), progress=progress,
                  backtracks=sum(b[1]>a[1] for a,b in zip(progress,progress[1:])),
                  writes=len(writes), max_writer_seconds=max(durations,default=0),
                  wal_before=wal_before, wal_after=wal_after, checkpoint=checkpoint,
                  live_epoch=live_epoch, connections_closed=closed)
    if ok:
        c = CONNECT(dst)
        report['snapshot_epoch'] = c.execute('SELECT value FROM epoch').fetchone()[0]
        report['integrity'] = c.execute('PRAGMA integrity_check').fetchone()[0]
        c.close()
    print(json.dumps(report))
    assert ok
    assert report['backtracks'] == 0
    assert writes and live_epoch > report['snapshot_epoch'] == 1
    assert report['integrity'] == 'ok'
    assert all(closed) and checkpoint == (0,0,0)
    assert dst.stat().st_mode & 0o777 == 0o600


def test_unspecified_total_duration_preserves_legacy_default(tmp_path, monkeypatch):
    import inspect
    assert inspect.signature(subject._safe_copy_db).parameters['max_duration_seconds'].default is None
    src, dst = tmp_path/'src.db', tmp_path/'dst.db'
    keeper = fixture_db(src)
    clock = iter([0.0] + [1000.0] * 20)
    monkeypatch.setattr(subject.time, 'monotonic', lambda: next(clock))
    try:
        assert subject._safe_copy_db(src, dst)
    finally:
        keeper.close()


def test_absolute_deadline_during_successful_steps(tmp_path, monkeypatch):
    src, dst = tmp_path/'src.db', tmp_path/'dst.db'
    keeper = fixture_db(src)
    class Slow(sqlite3.Connection):
        def backup(self, target, **kwargs):
            callback = kwargs['progress']
            def slow(*args):
                time.sleep(.03)
                callback(*args)
            kwargs['progress'] = slow
            return super().backup(target, **kwargs)
    monkeypatch.setattr(subject.sqlite3, 'connect', lambda *a, **kw: CONNECT(*a, factory=Slow, **kw))
    try:
        start = time.monotonic()
        assert not subject._safe_copy_db(src,dst,max_duration_seconds=.01)
        assert time.monotonic()-start < 1
        assert not dst.exists()
        assert keeper.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone() == (0,0,0)
    finally:
        keeper.close()


def test_busy_source_deadline(tmp_path):
    src, dst = tmp_path/'src.db', tmp_path/'dst.db'
    c = fixture_db(src,wal=False)
    c.execute('BEGIN EXCLUSIVE')
    try:
        start=time.monotonic()
        assert not subject._safe_copy_db(src,dst,timeout_seconds=.1)
        assert .09 <= time.monotonic()-start < 1
        assert not dst.exists()
    finally:
        c.rollback()
        c.close()


def test_rollback_journal_does_not_pin_writer(tmp_path,monkeypatch):
    src,dst=tmp_path/'src.db',tmp_path/'dst.db'
    c=fixture_db(src,wal=False)
    commits=[]
    class Observe(sqlite3.Connection):
        def backup(self,target,**kwargs):
            callback=kwargs['progress']
            def check(*args):
                callback(*args)
                if not commits:
                    c.execute('UPDATE epoch SET value=2')
                    c.commit()
                    commits.append(True)
            kwargs['progress']=check
            return super().backup(target,**kwargs)
    monkeypatch.setattr(subject.sqlite3,'connect',lambda *a,**kw: CONNECT(*a,factory=Observe,**kw))
    c.execute('PRAGMA busy_timeout=0')
    try:
        assert subject._safe_copy_db(src,dst)
        assert commits
    finally:
        c.close()


def test_symlink_target_not_written(tmp_path):
    src=tmp_path/'src.db'
    c=fixture_db(src); c.close()
    sentinel=tmp_path/'sentinel'; sentinel.write_bytes(b'untouched')
    dst=tmp_path/'dst.db'; dst.symlink_to(sentinel)
    assert not subject._safe_copy_db(src,dst)
    assert sentinel.read_bytes()==b'untouched'
