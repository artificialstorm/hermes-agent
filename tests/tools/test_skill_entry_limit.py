"""Profile-scoped SKILL.md limits through the native registry and atomic runner."""
import json
from pathlib import Path

import pytest

from hermes_cli.config import atomic_config_replace, atomic_config_write
from tools import skill_manager_tool as smt
from tools.registry import registry


CAP = 512


def entry(size, name="probe"):
    prefix = f"---\nname: {name}\ndescription: Use when testing limits.\nplatforms: [macos, linux]\n---\n# Probe\nTOKEN\n"
    return prefix + "é" * (size - len(prefix))


def call(**payload):
    tool = registry.get_entry("skill_manage")
    assert tool is not None
    return json.loads(tool.handler(payload))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(smt, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [tmp_path / "skills"])
    atomic_config_write(tmp_path / "config.yaml", {"skills": {"max_entry_chars": CAP}})
    return tmp_path


def seed(home, size=CAP + 100, name="probe"):
    target = home / "skills" / name / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text(entry(size, name), encoding="utf-8")
    return target


def test_exact_character_boundary_and_above_rejected_before_create(home):
    result = call(action="create", name="edge", content=entry(CAP, "edge"))
    assert result["success"], result
    target = home / "skills" / "edge" / "SKILL.md"
    assert target.read_text() == entry(CAP, "edge")
    result = call(action="create", name="large", content=entry(CAP + 1, "large"))
    assert not result["success"], result
    assert "skills.max_entry_chars" in result["error"]
    assert "narrow" in result["error"] and "on demand" in result["error"]
    assert not (home / "skills" / "large").exists()


@pytest.mark.parametrize("shape", ["edit", "rewrite", "patch", "explicit-patch", "write_file"])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_legacy_only_strict_shrink_allowed(home, shape, delta):
    target = seed(home)
    before = target.read_bytes()
    old = target.read_text()
    new = entry(len(old) + delta).replace("TOKEN", "ALTER")
    if shape == "edit":
        args = {"action": "edit", "content": new}
    elif shape == "rewrite":
        args = {"action": "patch", "content": new}
    elif shape in {"patch", "explicit-patch"}:
        args = {"action": "patch", "old_string": old, "new_string": new}
        if shape == "explicit-patch":
            args["file_path"] = "SKILL.md"
    else:
        args = {"action": "write_file", "file_path": "SKILL.md", "file_content": new}
    result = call(name="probe", **args)
    assert result["success"] is (delta < 0), result
    assert target.read_bytes() == (new.encode() if delta < 0 else before)
    assert "platforms: [macos, linux]" in target.read_text()


def test_supporting_files_keep_existing_limits_and_do_not_load_references(home, monkeypatch):
    target = seed(home)
    reference = "r" * (CAP + 200)
    assert call(action="write_file", name="probe", file_path="references/depth.md",
                file_content=reference)["success"]
    assert call(action="patch", name="probe", file_path="references/depth.md",
                old_string=reference, new_string=reference + "more")["success"]
    before = target.read_bytes()
    assert call(action="write_file", name="probe", file_path="references/huge.md",
                file_content="x" * (smt.MAX_SKILL_FILE_BYTES + 1))["success"] is False
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        assert path.name != "depth.md", "entry checks must not read supporting references"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    assert call(action="patch", name="probe", content=entry(CAP))["success"]
    assert target.read_bytes() != before


@pytest.mark.parametrize("size", [CAP, CAP + 100])
def test_batch_checks_final_size_not_transient_growth(home, size):
    target = seed(home, size)
    result = call(operations=[
        {"action": "patch", "name": "probe", "old_string": "TOKEN", "new_string": "LONGER-TOKEN"},
        {"action": "patch", "name": "probe", "old_string": "LONGER-TOKEN", "new_string": "X"},
    ])
    assert result["success"], result
    assert len(target.read_text()) == size - 4


def test_create_then_patch_uses_final_state(home):
    text = entry(CAP + 1)
    result = call(operations=[
        {"action": "create", "name": "probe", "content": text},
        {"action": "patch", "name": "probe", "old_string": "TOKEN", "new_string": "X"},
    ])
    assert result["success"], result
    assert (home / "skills/probe/SKILL.md").read_text() == text.replace("TOKEN", "X")


@pytest.mark.parametrize("ending", ["TOKEN", "TOKEN-plus"])
def test_batch_legacy_shrink_then_regrow_compares_original(home, ending, monkeypatch):
    target = seed(home)
    before = target.read_bytes()
    writes = []
    real_write = smt.atomic_write_text
    def record_write(*args, **kwargs):
        writes.append(args[0])
        return real_write(*args, **kwargs)
    monkeypatch.setattr(smt, "atomic_write_text", record_write)
    result = call(operations=[
        {"action": "write_file", "name": "probe", "file_path": "references/new.md", "file_content": "new"},
        {"action": "patch", "name": "probe", "old_string": "TOKEN", "new_string": "X"},
        {"action": "patch", "name": "probe", "old_string": "X", "new_string": ending},
    ])
    assert not result["success"], result
    assert not writes, "size refusal must happen before any skill file write"
    assert target.read_bytes() == before
    assert not (target.parent / "references/new.md").exists()


def test_batch_cross_skill_size_failure_leaves_every_target_untouched(home):
    a = seed(home, CAP, "a")
    b = seed(home, CAP, "b")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (a, b)}
    result = call(operations=[
        {"action": "patch", "name": "a", "old_string": "TOKEN", "new_string": "X"},
        {"action": "patch", "name": "b", "old_string": "TOKEN", "new_string": "TOO-LONG"},
    ])
    assert not result["success"], result
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (a, b)} == before


def test_final_state_preflight_does_not_break_runtime_rollback(home):
    target = seed(home)
    before = target.read_bytes()
    result = call(operations=[
        {"action": "patch", "name": "probe", "old_string": "TOKEN", "new_string": "X"},
        {"action": "write_file", "name": "probe", "file_path": "bad/ref.md", "file_content": "fail"},
    ])
    assert not result["success"] and result["failed_index"] == 1, result
    assert target.read_bytes() == before


@pytest.mark.parametrize("value", [0, -1, True, False, 513.0, "512", None, [], {}, 100001])
def test_invalid_explicit_limit_fails_closed(home, value):
    atomic_config_write(home / "config.yaml", {"skills": {"max_entry_chars": value}})
    result = call(action="create", name="probe", content=entry(200))
    assert not result["success"], result
    assert "skills.max_entry_chars" in result["error"] and "integer" in result["error"]
    assert not (home / "skills/probe").exists()


def test_malformed_yaml_never_falls_back_to_lax_limit(home):
    (home / "config.yaml").write_text("skills:\n  max_entry_chars: [\n")
    result = call(action="create", name="probe", content=entry(200))
    assert not result["success"], result
    assert "config" in result["error"]
    assert not (home / "skills/probe").exists()


def test_unconfigured_hard_ceiling_is_preserved(home):
    # This fixture deliberately removes the explicit cap. Upstream ordinary
    # writes reject omission so real callers cannot silently lose settings.
    atomic_config_replace(home / "config.yaml", {"skills": {}})
    assert call(action="create", name="probe", content=entry(smt.MAX_SKILL_CONTENT_CHARS))["success"]
    assert not call(action="create", name="too-big", content=entry(smt.MAX_SKILL_CONTENT_CHARS + 1))["success"]


def test_invalid_limit_does_not_block_deletion(home):
    target = seed(home)
    atomic_config_write(home / "config.yaml", {"skills": {"max_entry_chars": False}})
    result = call(operations=[{"action": "delete", "name": "probe"}])
    assert result["success"], result
    assert not target.exists()


def test_real_profile_config_is_resolved_per_call(home, monkeypatch):
    other = home / "other"
    other.mkdir()
    atomic_config_write(other / "config.yaml", {"skills": {"max_entry_chars": CAP * 2}})
    monkeypatch.setattr(smt, "SKILLS_DIR", smt._SKILLS_DIR_AT_IMPORT)
    from hermes_constants import get_hermes_home
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [get_hermes_home() / "skills"])
    for location, name, expected in [(home, "a", False), (other, "b", True), (home, "c", False)]:
        monkeypatch.setenv("HERMES_HOME", str(location))
        result = call(action="create", name=name, content=entry(CAP + 1, name))
        assert result["success"] is expected, result


def test_staged_replay_rechecks_current_limit(home):
    atomic_config_write(home / "config.yaml", {"skills": {"max_entry_chars": CAP * 2, "write_approval": True}})
    payload = {"operations": [{"action": "create", "name": "probe", "content": entry(CAP + 1)}]}
    staged = call(**payload)
    assert staged.get("staged"), staged
    assert not (home / "skills/probe").exists()
    atomic_config_write(home / "config.yaml", {"skills": {"max_entry_chars": CAP, "write_approval": True}})
    result = json.loads(smt.apply_skill_pending(payload))
    assert not result["success"], result
    assert not (home / "skills/probe").exists()
