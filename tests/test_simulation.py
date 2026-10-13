from motiverooms.catalog import preset_by_key, room_by_key
from motiverooms.session import AppSession


def test_ideal_cooperation_episode_escapes_both_agents() -> None:
    preset = preset_by_key("ideal_cooperation")
    session = AppSession(room_entry=room_by_key(preset.room_key), turn_budget=12)
    session.configure(
        room_key=preset.room_key,
        mode=preset.mode,
        partner_a=preset.partner_a,
        partner_b=preset.partner_b,
    )
    steps = session.run_scripted_episode()
    assert steps == 10
    assert session.game.engine.agents["A"].escaped
    assert session.game.engine.agents["B"].escaped
    assert len(session.episode_log) == 10
    assert session.episode_log[0].action == "inspect(sticky_note)"
    assert session.episode_log[0].controller == "honest"


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
    assert third["action"] != first["action"]


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
