"""Independent acceptance assertions, exercised against unmodified candidate code."""
import contextvars
import json
import stat
from pathlib import Path

import pytest
from hermes_cli.config import atomic_config_write
from tools import skill_manager_tool as smt
from tools.registry import registry

CAP = 512

def entry(size=CAP, name='probe'):
    prefix = f'---\nname: {name}\ndescription: Use when reviewing size limits.\nplatforms: [macos, linux]\n---\n# Probe\nTOKEN\n'
    assert size >= len(prefix)
    return prefix + 'e' * (size - len(prefix))

def call(**payload):
    tool = registry.get_entry('skill_manage')
    assert tool is not None
    return json.loads(tool.handler(payload))

@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(smt, 'SKILLS_DIR', smt._SKILLS_DIR_AT_IMPORT)
    from hermes_constants import get_hermes_home
    monkeypatch.setattr('agent.skill_utils.get_all_skills_dirs', lambda: [get_hermes_home() / 'skills'])
    atomic_config_write(tmp_path / 'config.yaml', {'skills': {'max_entry_chars': CAP}})
    return tmp_path

def seed(home, size=CAP, name='probe'):
    target = home / 'skills' / name / 'SKILL.md'
    target.parent.mkdir(parents=True)
    target.write_text(entry(size, name), encoding='utf-8')
    return target

@pytest.mark.platforms("any")
@pytest.mark.require_symlinks
@pytest.mark.parametrize('referent', ['../SKILL.md', '../skill.md'])
def test_removing_supporting_symlink_must_not_erase_entry_from_projection(home, referent):
    target = seed(home)
    alias = target.parent / 'references' / 'alias.md'
    alias.parent.mkdir()
    alias.symlink_to(referent)
    if not alias.exists():
        pytest.skip('requires a case-insensitive filesystem')
    before = target.read_bytes(), target.stat().st_mtime_ns
    result = call(operations=[
        {'name': 'probe', 'action': 'patch', 'old_string': 'TOKEN', 'new_string': 'TOO-LONG-TOKEN'},
        {'name': 'probe', 'action': 'remove_file', 'file_path': 'references/alias.md'},
    ])
    observed = {'result': result, 'final_chars': len(target.read_text()), 'cap': CAP,
                'alias_exists': alias.is_symlink()}
    assert not result['success'], observed
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    assert alias.is_symlink()

@pytest.mark.platforms("any")
@pytest.mark.require_symlinks
@pytest.mark.parametrize('referent', ['../SKILL.md', '../skill.md'])
@pytest.mark.parametrize('route', ['registry', 'final-write'])
def test_symlink_write_growth_is_denied_without_delete(home, referent, route):
    target = seed(home)
    alias = target.parent / 'references' / 'alias.md'
    alias.parent.mkdir()
    alias.symlink_to(referent)
    if not alias.exists():
        pytest.skip('requires a case-insensitive filesystem')
    assert alias.samefile(target)
    if route == 'registry':
        result = call(operations=[{'name': 'probe', 'action': 'patch', 'file_path': 'references/alias.md',
                                   'old_string': 'TOKEN', 'new_string': 'TOO-LONG-TOKEN'}])
    else:
        result = smt._guarded_write('probe', target.parent, alias.resolve(), 'patch',
                                    'references/alias.md', entry(CAP + 1))
    assert result and not result['success'], result
    assert target.read_text() == entry()

def test_contextvar_is_reset_after_success_failure_and_staging(home):
    target = seed(home, CAP + 100)
    assert not smt._batch_entry_targets.get()
    assert call(operations=[{'name': 'probe', 'action': 'patch', 'old_string': 'TOKEN', 'new_string': 'T'}])['success']
    assert not smt._batch_entry_targets.get()
    assert not call(operations=[{'name': 'probe', 'action': 'patch', 'old_string': 'T\n', 'new_string': 'TOO-LONG\n'}])['success']
    assert not smt._batch_entry_targets.get()
    atomic_config_write(home / 'config.yaml', {'skills': {'max_entry_chars': CAP, 'write_approval': True}})
    result = call(operations=[{'name': 'probe', 'action': 'patch', 'content': entry(CAP)}])
    assert result.get('staged'), result
    assert len(target.read_text()) > CAP
    assert not smt._batch_entry_targets.get()
    assert not contextvars.Context().run(smt._batch_entry_targets.get)

@pytest.mark.platforms("posix")
def test_batch_preserves_permissions_platform_metadata_and_security_gate(home, monkeypatch):
    target = seed(home, CAP + 100)
    target.chmod(0o640)
    result = call(operations=[{'name': 'probe', 'action': 'patch', 'old_string': 'TOKEN', 'new_string': 'T'}])
    assert result['success'], result
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert b'platforms: [macos, linux]' in target.read_bytes()
    shortened = target.read_bytes()
    monkeypatch.setattr(smt, '_security_scan_skill', lambda skill_dir: 'controlled scan refusal')
    result = call(operations=[{'name': 'probe', 'action': 'patch', 'old_string': 'T\n', 'new_string': '\n'}])
    assert not result['success'] and 'controlled scan refusal' in result['error'], result
    assert target.read_bytes() == shortened
    assert not smt._batch_entry_targets.get()

def test_empty_batch_delete_mixed_and_duplicate_destructive_contracts(home):
    target = seed(home)
    cases = [[], [{'name': 'probe', 'action': 'delete'}, {'name': 'probe', 'action': 'create', 'content': entry()}],
             [{'name': 'probe', 'action': 'patch', 'old_string': 'TOKEN', 'new_string': 'T'},
              {'name': 'probe', 'action': 'write_file', 'file_path': 'SKILL.md', 'file_content': entry()}]]
    for ops in cases:
        result = call(operations=ops)
        assert result['success'] is False and isinstance(result['error'], str), result
        assert target.read_text() == entry()

def test_flat_oversize_is_not_staged(home):
    atomic_config_write(home / 'config.yaml', {'skills': {'max_entry_chars': CAP, 'write_approval': True}})
    result = call(action='create', name='probe', content=entry(CAP + 1))
    assert not result['success'] and not result.get('staged'), result

def test_removing_entry_is_unchanged_with_invalid_limit(home):
    target = seed(home)
    atomic_config_write(home / 'config.yaml', {'skills': {'max_entry_chars': False}})
    result = call(operations=[{'name': 'probe', 'action': 'remove_file', 'file_path': 'SKILL.md'}])
    assert result['success'], result
    assert not target.exists()

def test_preflight_unreadable_create_retains_json_error_contract(home):
    target = seed(home)
    target.write_bytes(b'\xff')
    result = call(operations=[{'name': 'probe', 'action': 'create', 'content': entry()}])
    assert not result['success'], result
    assert target.read_bytes() == b'\xff'


from concurrent.futures import ThreadPoolExecutor

def test_patch_projection_matches_actual_text_read_between_operations(home):
    target = seed(home)
    before = target.read_bytes(), target.stat().st_mtime_ns
    result = call(operations=[
        {'name': 'probe', 'action': 'patch', 'old_string': 'TOKEN', 'new_string': 'ABCDEFGHIJK\r'},
        {'name': 'probe', 'action': 'patch', 'old_string': 'ABCDEFGHIJK\r', 'new_string': r'\r\r\r\r'},
    ])
    from tools.fuzzy_match import fuzzy_find_and_replace
    transient = entry().replace('TOKEN', 'ABCDEFGHIJK\r')
    projected = fuzzy_find_and_replace(transient, 'ABCDEFGHIJK\r', r'\r\r\r\r', False)
    actual = fuzzy_find_and_replace(transient.replace('\r', '\n'), 'ABCDEFGHIJK\r', r'\r\r\r\r', False)
    observed = {'result': result, 'cap': CAP, 'final_chars': len(target.read_text()),
                'projected_chars': len(projected[0]), 'actual_chars': len(actual[0]),
                'projection_strategy': projected[2], 'actual_strategy': actual[2]}
    assert not result['success'], observed
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_context_local_profile_a_b_a_limits_and_batch_target_reset(home):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    other = home / 'other-profile'
    atomic_config_write(other / 'config.yaml', {'skills': {'max_entry_chars': CAP * 2}})
    for path, name, succeeds in [(home, 'a', False), (other, 'b', True), (home, 'c', False)]:
        token = set_hermes_home_override(path)
        try:
            result = call(operations=[{'action': 'create', 'name': name, 'content': entry(CAP + 1, name)}])
            assert result['success'] is succeeds, result
            assert (path / 'skills' / name / 'SKILL.md').exists() is succeeds
            assert not smt._batch_entry_targets.get()
        finally:
            reset_hermes_home_override(token)


def test_nested_lock_batch_and_op_remain_reentrant_and_serialize(home):
    target = seed(home, CAP + 100)
    def shrink(ch):
        return call(operations=[{'action': 'patch', 'name': 'probe', 'old_string': 'eeee', 'new_string': ch, 'replace_all': True}])
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(shrink, ch) for ch in ('z', 'y')]
        results = [future.result(timeout=10) for future in futures]
    assert sum(r['success'] for r in results) == 1, results
    assert len(target.read_text()) < CAP
    assert not smt._batch_entry_targets.get()


def test_unwrapped_native_batch_has_json_error_for_existing_invalid_utf8(home):
    target = seed(home)
    target.write_bytes(b'\xff')
    result = json.loads(smt._skill_manage_batch([{'action': 'create', 'name': 'probe', 'content': entry()}]))
    assert not result['success'] and 'already exists' in result['error'], result


def test_unwrapped_native_batch_still_allows_deleting_entry_with_invalid_limit(home):
    target = seed(home)
    atomic_config_write(home / 'config.yaml', {'skills': {'max_entry_chars': False}})
    result = json.loads(smt._skill_manage_batch([{'action': 'remove_file', 'name': 'probe', 'file_path': 'SKILL.md'}]))
    assert result['success'], result
    assert not target.exists()


@pytest.mark.parametrize("replacement", ["A\rB", "A\r\nB", "\ufeff---"])
def test_projection_normalizes_native_intermediate_read(home, replacement):
    target = seed(home, CAP - 100)
    old = "---\nname" if replacement.startswith("\ufeff") else "TOKEN"
    if replacement.startswith("\ufeff"):
        replacement += "\nname"
    text = target.read_text().replace(old, replacement, 1)
    from io import StringIO
    normalized = StringIO(text.removeprefix("\ufeff"), newline=None).read()
    # A following patch must use the native read of the prior write, including BOM stripping.
    result = call(operations=[
        {"name": "probe", "action": "patch", "old_string": old, "new_string": replacement},
        {"name": "probe", "action": "patch", "old_string": normalized, "new_string": entry(CAP - 101)},
    ])
    assert result['success'], result
    assert target.read_text(encoding='utf-8-sig') == entry(CAP - 101)


@pytest.mark.platforms("any")
@pytest.mark.require_symlinks
@pytest.mark.parametrize('referent', ['../SKILL.md', '../skill.md'])
def test_within_limit_patch_and_alias_unlink_succeeds(home, referent):
    target = seed(home)
    alias = target.parent / 'references/alias.md'
    alias.parent.mkdir()
    alias.symlink_to(referent)
    if not alias.exists():
        pytest.skip('requires a case-insensitive filesystem')
    result = call(operations=[
        {'name': 'probe', 'action': 'patch', 'file_path': 'references/alias.md',
         'old_string': 'TOKEN', 'new_string': 'LONGER-TOKEN'},
        {'name': 'probe', 'action': 'patch', 'old_string': 'LONGER-TOKEN', 'new_string': 'T'},
    ])
    assert result['success'], result
    assert target.read_text() == entry().replace('TOKEN', 'T')
    # Native batch rules forbid removing a path patched earlier in the same batch.
    result = call(operations=[
        {'name': 'probe', 'action': 'patch', 'old_string': 'T\n', 'new_string': '\n'},
        {'name': 'probe', 'action': 'remove_file', 'file_path': 'references/alias.md'},
    ])
    assert result['success'], result
    assert target.read_text() == entry().replace('TOKEN', '')
    assert not alias.is_symlink()


@pytest.mark.parametrize('shape', ['edit', 'rewrite', 'targeted', 'explicit', 'write_file'])
def test_flat_oversize_updates_not_staged(home, shape):
    target = seed(home)
    atomic_config_write(home / 'config.yaml', {'skills': {'max_entry_chars': CAP, 'write_approval': True}})
    args = {'edit': dict(action='edit', content=entry(CAP + 1)),
            'rewrite': dict(action='patch', content=entry(CAP + 1)),
            'targeted': dict(action='patch', old_string='TOKEN', new_string='LONGER'),
            'explicit': dict(action='patch', file_path='SKILL.md', old_string='TOKEN', new_string='LONGER'),
            'write_file': dict(action='write_file', file_path='SKILL.md', file_content=entry(CAP + 1))}[shape]
    before = target.read_bytes(), target.stat().st_mtime_ns
    result = call(name='probe', **args)
    assert not result['success'] and not result.get('staged'), result
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    assert not list((home / 'pending').glob('**/*'))


def test_flat_staging_and_replay_preserve_approval_and_recheck_cap(home):
    payload = dict(action='create', name='probe', content=entry(CAP))
    atomic_config_write(home / 'config.yaml', {'skills': {'max_entry_chars': CAP, 'write_approval': True}})
    assert call(**payload).get('staged')
    assert not (home / 'skills/probe/SKILL.md').exists()
    atomic_config_write(home / 'config.yaml', {'skills': {'max_entry_chars': CAP - 1, 'write_approval': True}})
    result = json.loads(smt.apply_skill_pending(payload))
    assert not result['success'], result
    assert not (home / 'skills/probe/SKILL.md').exists()


@pytest.mark.parametrize('flat', [False, True])
def test_unreadable_patch_fails_closed_without_exception_or_write(home, flat):
    target = seed(home)
    target.write_bytes(b'\xff')
    before = target.read_bytes(), target.stat().st_mtime_ns
    op = dict(action='patch', name='probe', old_string='TOKEN', new_string='T')
    result = call(**op) if flat else call(operations=[op])
    assert not result['success'] and isinstance(result['error'], str), result
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_first_patch_does_not_normalize_an_already_read_bom_twice(home, monkeypatch):
    target = seed(home)
    target.write_text('\ufeff\ufeff' + entry())
    expected = target.read_text(encoding='utf-8-sig')
    from tools import fuzzy_match
    original = fuzzy_match.fuzzy_find_and_replace
    seen = []
    def capture(content, *args, **kwargs):
        seen.append(content)
        return original(content, *args, **kwargs)
    monkeypatch.setattr(fuzzy_match, 'fuzzy_find_and_replace', capture)
    result = call(operations=[dict(action='patch', name='probe', old_string='TOKEN', new_string='T')])
    assert result['success'], result
    assert seen == [expected, expected]
