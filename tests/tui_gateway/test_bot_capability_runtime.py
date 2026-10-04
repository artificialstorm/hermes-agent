"""Capability refresh changes tools, not the conversation's selected runtime."""

import copy
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml


@pytest.mark.parametrize("service_tier", ["priority", ""])
def test_bot_capability_refresh_preserves_session_runtime(tmp_path, monkeypatch, service_tier):
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from tui_gateway import server

    home = get_hermes_home()
    config = {"model": {"default": "gpt-6.1-sol", "provider": "openai"},
              "agent": {"reasoning_effort": "low", "service_tier": "priority"}}
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    config_before = config_path.read_bytes()
    picked = {"model": "gpt-6-astra-900k", "provider": "openai-codex",
              "base_url": "https://selected.invalid/v1", "api_key": "test-only",
              "api_mode": "codex_responses"}
    reasoning = {"enabled": True, "effort": "high"}
    history = [{"role": "user", "content": "Preserve every detail."},
               {"role": "assistant", "content": "Conversation continues."}]
    history_before = copy.deepcopy(history)
    key = "bot-capability-runtime"
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(key, "desktop", model=picked["model"])
    for message in history:
        db.append_message(key, message["role"], message["content"])
    stored_before = db.get_messages(key)
    old = SimpleNamespace(**picked, reasoning_config=reasoning, service_tier=service_tier,
                          _session_db=db, _owns_session_db=True, _session_title_hint="Bot Chat")
    session = {"agent": old, "session_key": key, "source": "desktop", "cwd": str(tmp_path),
               "bot_caps_seen": "before", "model_override": picked,
               "create_reasoning_override": reasoning, "create_service_tier_override": service_tier,
               "history": history, "history_version": 7}

    # Only credential acquisition and the agent constructor are substituted: the real
    # capability -> rebuild -> factory -> model-resolution chain remains under test.
    def runtime_provider(*, requested=None, **_kwargs):
        return {"provider": requested, "api_key": "test-only",
                "base_url": "https://profile.invalid/v1", "api_mode": "chat_completions"}

    def agent_constructor(**kwargs):
        return SimpleNamespace(**kwargs, _session_db=kwargs["session_db"], _owns_session_db=False)

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", runtime_provider)
    monkeypatch.setattr("run_agent.AIAgent", agent_constructor)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda _platform: [])
    monkeypatch.setattr("tools.bot_mode_probe.capability_fingerprint", lambda _home: "after")
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    try:
        server._sync_bot_capabilities("sid-bot", session)
        new = session["agent"]
        assert new is not old
        assert (new.model, new.provider) == (picked["model"], picked["provider"])
        assert (new.base_url, new.api_key, new.api_mode) == tuple(
            picked[k] for k in ("base_url", "api_key", "api_mode"))
        assert new.reasoning_config == reasoning
        assert new.service_tier == service_tier
        assert new.session_id == session["session_key"] == key
        assert new._session_db is db
        assert new._owns_session_db and not old._owns_session_db
        assert session["history"] is history
        assert history == history_before and session["history_version"] == 7
        assert db.get_messages(key) == stored_before
        assert config_path.read_bytes() == config_before
        server._sync_agent_model_with_config("sid-bot", session)
        assert session["agent"] is new
        assert (new.model, new.provider) == (picked["model"], picked["provider"])
        server._sync_bot_capabilities("sid-bot", session)
        assert session["agent"] is new, "unchanged capabilities must not rebuild again"
    finally:
        db.close()
