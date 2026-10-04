import importlib.util, os, sqlite3, stat, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from hermes_cli import backup_sqlite as m

class Identity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def test_aliases(self):
        for wal in (False,True):
            for duration in (None,0):
                for alias in ('exact','dotdot','symlink','hardlink','parent_symlink'):
                    with self.subTest(wal=wal,duration=duration,alias=alias), tempfile.TemporaryDirectory(dir=self.root) as td:
                        root=Path(td); src=root/'s.db'; dst=src
                        c=sqlite3.connect(src)
                        if wal: c.execute('pragma journal_mode=wal')
                        c.execute('create table t(x)'); c.execute('insert into t values(7)'); c.commit()
                        src.chmod(0o640)
                        if alias=='dotdot': (root/'child').mkdir(); dst=root/'child/../s.db'
                        if alias=='symlink': dst=root/'link'; dst.symlink_to(src)
                        if alias=='hardlink': dst=root/'hard'; os.link(src,dst)
                        if alias=='parent_symlink': (root/'parent').symlink_to(root,target_is_directory=True); dst=root/'parent/s.db'
                        before=src.read_bytes(); mode=stat.S_IMODE(src.stat().st_mode); inode=src.stat().st_ino
                        ok=m._safe_copy_db(src,dst,timeout_seconds=.001,max_duration_seconds=duration)
                        try:
                            self.assertTrue(src.exists(),'SOURCE DELETED')
                            self.assertFalse(ok)
                            self.assertEqual(src.read_bytes(),before)
                            self.assertEqual(stat.S_IMODE(src.stat().st_mode),mode)
                            self.assertEqual(src.stat().st_ino,inode)
                            self.assertTrue(dst.exists())
                            c.execute('insert into t values(8)'); c.commit()
                            self.assertEqual(c.execute('pragma integrity_check').fetchone(),('ok',))
                            self.assertEqual(c.execute('select x from t').fetchall(),[(7,),(8,)])
                        finally: c.close()
    def test_invalid_and_missing(self):
        with tempfile.TemporaryDirectory(dir=self.root) as td:
            root=Path(td); src=root/'s'; dst=root/'d'
            self.assertFalse(m._safe_copy_db(src,src,max_duration_seconds=0)); self.assertFalse(src.exists())
            src.write_bytes(b'malformed'); src.chmod(0o640)
            for alias in ('exact','hard','sym'):
                dst=src if alias=='exact' else root/alias
                if alias=='hard': os.link(src,dst)
                if alias=='sym': dst.symlink_to(src)
                self.assertFalse(m._safe_copy_db(src,dst,max_duration_seconds=0))
                self.assertEqual(src.read_bytes(),b'malformed'); self.assertEqual(src.stat().st_mode&0o777,0o640)
                self.assertTrue(dst.exists())
    def test_identity_failure_no_effects(self):
        with tempfile.TemporaryDirectory(dir=self.root) as td:
            src=Path(td)/'s'; dst=Path(td)/'d'; src.write_bytes(b'source'); dst.write_bytes(b'destination')
            with patch.object(Path,'stat',side_effect=PermissionError('identity unavailable')):
                self.assertFalse(m._safe_copy_db(src,dst))
            self.assertEqual(src.read_bytes(),b'source'); self.assertEqual(dst.read_bytes(),b'destination')
