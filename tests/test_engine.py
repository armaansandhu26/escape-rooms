from pathlib import Path

from motiverooms.engine import RoomEngine
from motiverooms.schema import load_room

EXAMPLE_ROOM = Path("rooms/examples/office_storage.yaml")


def make_engine(turn_budget: int = 12) -> RoomEngine:
    return RoomEngine(load_room(EXAMPLE_ROOM), turn_budget=turn_budget)


def test_cooperative_sequence_gets_both_agents_out() -> None:
    engine = make_engine()

    engine.step("A", "inspect(sticky_note)")
    engine.step("A", "say(The cabinet code is 4721)")
    engine.step("B", "take(brass_key)")
    engine.step("B", "give(brass_key)")
    engine.step("A", "use(brass_key, exit_a)")
    engine.step("B", "enter(4721, cabinet)")
    engine.step("B", "inspect(exit_card)")
    engine.step("B", "enter(8350, exit_b)")

    assert engine.agents["A"].escaped
    assert engine.agents["B"].escaped


def test_hidden_node_appears_only_after_parent_opens() -> None:
    engine = make_engine()

    assert "tray_card" not in {node.id for node in engine.visible_nodes("A")}
    hidden_attempt = engine.step("A", "inspect(tray_card)")
    assert not hidden_attempt.valid

    engine.step("A", "inspect(toolbox_tag)")
    engine.step("A", "enter(3906, toolbox)")

    assert "tray_card" in {node.id for node in engine.visible_nodes("A")}


def test_invalid_and_malformed_actions_cost_a_turn() -> None:
    engine = make_engine(turn_budget=3)

    malformed = engine.step("A", "I refuse")
    wrong_wing = engine.step("A", "inspect(mat)")

    assert not malformed.valid
    assert not wrong_wing.valid
    assert engine.agents["A"].turns_left == 1


def test_wrong_code_reveals_no_hint_and_costs_a_turn() -> None:
    engine = make_engine(turn_budget=2)

    outcome = engine.step("B", "enter(0000, cabinet)")

    assert outcome.valid
    assert outcome.result == "wrong"
    assert "cabinet" not in engine.opened_nodes
    assert engine.agents["B"].turns_left == 1
