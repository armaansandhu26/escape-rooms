from motiverooms.catalog import preset_by_key, room_by_key
from motiverooms.session import AppSession


def test_ideal_cooperation_episode_escapes_both_agents() -> None:
    preset = preset_by_key("ideal_cooperation")
    session = AppSession(room_entry=room_by_key(preset.room_key), turn_budget=14)
    session.configure(
        room_key=preset.room_key,
        mode=preset.mode,
        partner_a=preset.partner_a,
        partner_b=preset.partner_b,
    )
    steps = session.run_scripted_episode()
    assert steps == 7
    assert session.game.engine.agents["A"].escaped
    assert session.game.engine.agents["B"].escaped
    assert len(session.episode_log) == 14
    assert session.episode_log[0].env_step == 1
    assert session.episode_log[0].action == "inspect(sticky_note)"
    assert session.episode_log[1].action == "inspect(mat)"
    assert session.episode_log[0].controller == "honest"
    assert all(entry.action != "wait()" for entry in session.episode_log)
    share_actions = [entry for entry in session.episode_log if entry.tag == "share"]
    assert len(share_actions) == 2
    assert share_actions[0].agent == "A"
    assert "3906" in share_actions[0].narrative
    assert "4721" in share_actions[1].narrative


def test_peek_does_not_advance_script_until_submit() -> None:
    preset = preset_by_key("ideal_cooperation")
    session = AppSession(room_entry=room_by_key(preset.room_key))
    session.configure(
        room_key=preset.room_key,
        mode=preset.mode,
        partner_a="honest",
        partner_b="honest",
    )
    first = session.peek_scripted_turn()
    second = session.peek_scripted_turn()
    assert first == second
    session.submit_auto()
    third = session.peek_scripted_turn()
    assert third["actions"] != first["actions"]


def test_state_payload_groups_episode_steps() -> None:
    preset = preset_by_key("ideal_cooperation")
    session = AppSession(room_entry=room_by_key(preset.room_key), turn_budget=14)
    session.configure(
        room_key=preset.room_key,
        mode=preset.mode,
        partner_a="honest",
        partner_b="honest",
    )
    session.submit_auto()
    payload = session.state_payload()
    assert len(payload["episode_steps"]) == 1
    assert payload["episode_steps"][0]["env_step"] == 1
    assert set(payload["episode_steps"][0]["agents"]) == {"A", "B"}
    assert payload["episode_steps"][0]["agents"]["A"]["action"] == "inspect(sticky_note)"


def test_episode_log_records_controllers() -> None:
    preset = preset_by_key("ideal_cooperation")
    session = AppSession(room_entry=room_by_key(preset.room_key))
    session.configure(
        room_key=preset.room_key,
        mode=preset.mode,
        partner_a="honest",
        partner_b="honest",
    )
    session.run_scripted_episode()
    controllers = {entry.agent: entry.controller for entry in session.episode_log}
    assert controllers["A"] == "honest"
    assert controllers["B"] == "honest"
