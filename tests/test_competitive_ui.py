from pathlib import Path

from motiverooms.partners import COMPETITIVE_JOINT_SCRIPTS, PartnerBrain, joint_script_for_room
from motiverooms.schema import load_room
from motiverooms.web_game import WebGame


def test_competitive_joint_script_differs_from_cooperative() -> None:
    room_id = "shared_mat_example"
    coop = joint_script_for_room(room_id, "collaborative")
    comp = joint_script_for_room(room_id, "competitive")
    assert coop is not None and comp is not None
    assert coop != comp
    assert len(comp) == len(COMPETITIVE_JOINT_SCRIPTS[room_id])


def test_competitive_web_replay_one_winner() -> None:
    room = load_room(Path("rooms/examples/shared_mat.yaml"))
    game = WebGame(room, turn_budget=14, mode="competitive")
    brain = PartnerBrain(room, mode="competitive")
    for _ in COMPETITIVE_JOINT_SCRIPTS[room.id]:
        if game.round_over:
            break
        step = brain.next_joint_step()
        assert step is not None
        game.submit_joint(step)
    assert game.race_winner == "A"
    assert game.engine.agents["A"].escaped
    assert not game.engine.agents["B"].escaped
