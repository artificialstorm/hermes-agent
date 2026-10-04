"""Managed-entry ingress: native guards, ShellFileOperations and real local writes."""
import json
import os
import subprocess
from pathlib import Path
import pytest
from hermes_cli.config import atomic_config_write
from tools import file_tools
from tools.file_operations import ShellFileOperations
from tools.environments.local import LocalEnvironment

pytestmark = pytest.mark.platforms("posix")

@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    class LocalTestEnvironment(LocalEnvironment):
        def __init__(self):
            self.cwd = str(tmp_path)
        def execute(self, command, cwd=None, timeout=None, **kwargs):
            result = subprocess.run(command, shell=True, executable='/bin/sh', cwd=cwd or self.cwd,
                                    text=True, capture_output=True, timeout=timeout or 20,
                                    env=os.environ.copy(), input=kwargs.get('stdin_data'))
            return {'output': result.stdout + result.stderr, 'returncode': result.returncode}
    ops = ShellFileOperations(LocalTestEnvironment(), cwd=str(tmp_path))
    monkeypatch.setattr(file_tools, '_get_file_ops', lambda task_id='default': ops)
    atomic_config_write(tmp_path / 'config.yaml', {'skills': {'max_entry_chars': 512, 'write_approval': True}})
    return tmp_path, ops


def seed(local):
    home, _ = local
    target = home / 'skills' / 'probe' / 'SKILL.md'
    target.parent.mkdir(parents=True)
    target.write_text('TOKEN\n' + 'e' * 506)
    return target


def write(target, mode, task='guard-correction'):
    if mode == 'write':
        read = json.loads(file_tools.read_file_tool(str(target), task_id=task))
        assert not read.get('error'), read
        return json.loads(file_tools.write_file_tool(str(target), 'TOKEN\n' + 'e' * 556, task_id=task))
    if mode == 'replace':
        return json.loads(file_tools.patch_tool(path=str(target), old_string='TOKEN', new_string='LONGER-TOKEN', task_id=task))
    return json.loads(file_tools.patch_tool(mode='patch', patch=f'*** Begin Patch\n*** Update File: {target}\n@@\n-TOKEN\n+LONGER-TOKEN\n*** End Patch', task_id=task))


@pytest.mark.parametrize('mode', ['write', 'replace', 'v4a'])
@pytest.mark.parametrize('spelling', ['canonical', 'case-alias'])
def test_native_writer_refuses_managed_entry_before_write(local, mode, spelling):
    target = seed(local)
    if spelling == 'case-alias':
        alias = local[0] / 'SKILLS' / 'probe' / 'SKILL.md'
        if not alias.exists():
            pytest.skip('requires a case-insensitive filesystem')
        assert alias.samefile(target)
        target = alias
    before = target.read_bytes(), target.stat().st_mtime_ns
    result = write(target, mode)
    assert 'skill_manage' in result.get('error', ''), result
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    assert not list((local[0] / 'pending').glob('**/*'))


@pytest.mark.parametrize('mode', ['write', 'replace', 'v4a'])
@pytest.mark.parametrize('location', ['ordinary.txt', 'project/SKILL.md', 'skills/probe/references/notes.md', 'SKILLS/separate/SKILL.md'])
def test_unrelated_files_remain_writable(local, mode, location):
    seed(local)
    target = local[0] / location
    if location.startswith('SKILLS/') and (local[0] / 'SKILLS').exists():
        pytest.skip('requires distinct case-sensitive directories')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('TOKEN\n')
    result = write(target, mode)
    assert not result.get('error'), result
    assert target.read_text() != 'TOKEN\n'


@pytest.mark.parametrize('mode', ['write', 'replace', 'v4a'])
@pytest.mark.require_symlinks
def test_alias_to_managed_entry_is_refused(local, mode):
    target = seed(local)
    alias = local[0] / 'alias.txt'
    alias.symlink_to(target)
    result = write(alias, mode)
    assert 'skill_manage' in result.get('error', ''), result
    assert target.read_text().startswith('TOKEN\n')


@pytest.mark.parametrize('spelling', ['canonical', 'case-alias'])
def test_new_create_dir_entry_is_refused(local, spelling):
    home, _ = local
    external = home / 'fleet'
    if spelling == 'case-alias':
        anchor = home / 'Anchor'
        anchor.mkdir()
        alias = home / 'ANCHOR'
        if not alias.exists():
            pytest.skip('requires a case-insensitive filesystem')
        external = anchor / 'fleet'
    atomic_config_write(home / 'config.yaml', {'skills': {'max_entry_chars': 512, 'write_approval': True, 'create_dir': str(external)}})
    target = (alias / 'fleet' if spelling == 'case-alias' else external) / 'category/new/SKILL.md'
    result = json.loads(file_tools.write_file_tool(str(target), 'new skill'))
    assert 'skill_manage' in result.get('error', ''), result
    assert not target.exists()


@pytest.mark.parametrize('configured,spelled', [
    ('fleet', 'FLEET'), ('fleet/nested', 'fleet/NESTED'), ('skills', 'SKILLS'),
])
@pytest.mark.parametrize('route', ['write', 'batch'])
def test_missing_root_case_ambiguity_fails_closed(local, configured, spelled, route):
    home, _ = local
    root = home / configured
    assert not root.exists()
    atomic_config_write(home / 'config.yaml', {'skills': {
        'max_entry_chars': 512, 'write_approval': True, 'create_dir': str(root)}})
    target = home / spelled / 'probe/SKILL.md'
    ordinary = home / 'ordinary.txt'
    ordinary.write_text('TOKEN\n')
    if route == 'write':
        result = json.loads(file_tools.write_file_tool(str(target), 'unapproved entry'))
    else:
        result = json.loads(file_tools.patch_tool(mode='patch', patch=(
            f'*** Begin Patch\n*** Update File: {ordinary}\n@@\n-TOKEN\n+CHANGED\n'
            f'*** Add File: {target}\n+unapproved entry\n*** End Patch')))
    assert 'skill_manage' in result.get('error', ''), result
    assert not target.exists()
    assert not root.exists()
    assert ordinary.read_text() == 'TOKEN\n'


@pytest.mark.require_symlinks
def test_missing_root_alias_through_existing_symlink_fails_closed(local):
    home, _ = local
    anchor = home / 'anchor'
    anchor.mkdir()
    alias = home / 'alias'
    alias.symlink_to(anchor, target_is_directory=True)
    root = anchor / 'fleet/nested'
    atomic_config_write(home / 'config.yaml', {'skills': {
        'max_entry_chars': 512, 'write_approval': True, 'create_dir': str(root)}})
    target = alias / 'FLEET/NESTED/probe/SKILL.md'
    result = json.loads(file_tools.write_file_tool(str(target), 'unapproved entry'))
    assert 'skill_manage' in result.get('error', ''), result
    assert not root.exists()
    assert not target.exists()


@pytest.mark.parametrize('spelled,blocked', [('fleet', True), ('elsewhere', False), ('fleet-other', False)])
def test_missing_root_exact_and_unrelated_native_controls(local, spelled, blocked):
    home, _ = local
    atomic_config_write(home / 'config.yaml', {'skills': {
        'max_entry_chars': 512, 'write_approval': True, 'create_dir': str(home / 'fleet')}})
    target = home / spelled / 'probe/SKILL.md'
    result = json.loads(file_tools.write_file_tool(str(target), 'entry'))
    assert ('skill_manage' in result.get('error', '')) is blocked, result
    assert target.exists() is not blocked
    if not blocked:
        assert target.read_text() == 'entry'


def test_external_managed_entry_is_refused(local):
    home, _ = local
    external = home / 'fleet'
    target = external / 'category/probe/SKILL.md'
    target.parent.mkdir(parents=True)
    target.write_text('TOKEN\n')
    atomic_config_write(home / 'config.yaml', {'skills': {'max_entry_chars': 512, 'write_approval': True, 'external_dirs': [str(external)]}})
    assert 'skill_manage' in write(target, 'replace').get('error', '')
    assert target.read_text() == 'TOKEN\n'


def test_multifile_patch_refuses_before_ordinary_sibling_write(local):
    target = seed(local)
    ordinary = local[0] / 'ordinary.txt'
    ordinary.write_text('TOKEN\n')
    patch = '*** Begin Patch\n' + ''.join(f'*** Update File: {p}\n@@\n-TOKEN\n+LONGER-TOKEN\n' for p in (ordinary, target)) + '*** End Patch'
    result = json.loads(file_tools.patch_tool(mode='patch', patch=patch))
    assert 'skill_manage' in result.get('error', ''), result
    assert ordinary.read_text() == 'TOKEN\n'
    assert target.read_text().startswith('TOKEN\n')


def test_active_profile_binding_a_b_a(local):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tools import file_tools_write_guards as guards
    target = seed(local)
    other = local[0] / 'profiles/other'
    other.mkdir(parents=True)
    for home, blocked in [(local[0], True), (other, False), (local[0], True)]:
        token = set_hermes_home_override(home)
        try:
            err = guards._check_managed_skill_entry_write(str(target))
            assert bool(err) is blocked, err
        finally:
            reset_hermes_home_override(token)


def test_remote_namespace_does_not_use_host_skill_policy(local, monkeypatch):
    from tools import file_tools_write_guards as guards
    target = seed(local)
    local[1].env = object()  # Nonlocal backend: identical text names a different filesystem.
    monkeypatch.setattr(guards, '_resolve_path_for_task', lambda *a, **k: pytest.fail('must not stat remote paths on host'))
    assert guards._check_managed_skill_entry_write(str(target)) is None


def test_unrelated_instruction_gate_still_applies(local, monkeypatch):
    from tools import file_tools_write_guards as guards
    project = local[0].parent / 'ordinary-project'
    target = project / 'AGENTS.md'
    monkeypatch.setattr(guards, '_request_protected_instruction_approval', lambda *a, **k: 'instruction approval refused')
    result = json.loads(file_tools.write_file_tool(str(target), 'steers behavior'))
    assert result.get('error') == 'instruction approval refused', result
    assert not target.exists()


@pytest.mark.parametrize('cross_profile', [False, True])
def test_new_profile_entry_and_cross_profile_flag_cannot_bypass(local, cross_profile):
    target = local[0] / 'skills/category/new/SKILL.md'
    result = json.loads(file_tools.write_file_tool(str(target), 'TOKEN', cross_profile=cross_profile))
    assert 'skill_manage' in result.get('error', ''), result
    assert not target.exists()


@pytest.mark.parametrize('operation', ['delete', 'move-in', 'move-out'])
def test_v4a_entry_mutations_route_through_skill_manage(local, operation):
    target = seed(local)
    ordinary = local[0] / 'ordinary.txt'
    ordinary.write_text('TOKEN\n')
    header = {'delete': f'*** Delete File: {target}',
              'move-in': f'*** Move File: {ordinary} -> {target}',
              'move-out': f'*** Move File: {target} -> {ordinary}'}[operation]
    result = json.loads(file_tools.patch_tool(mode='patch', patch=f'*** Begin Patch\n{header}\n*** End Patch'))
    assert 'skill_manage' in result.get('error', ''), result
    assert target.read_text().startswith('TOKEN\n')
    assert ordinary.read_text() == 'TOKEN\n'
