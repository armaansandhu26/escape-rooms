from pathlib import Path

from motiverooms.env import MotiveRoomsEnv

SHARED = Path("rooms/examples/shared_mat.yaml")


def test_joint_step_env() -> None:
    env = MotiveRoomsEnv.from_yaml(SHARED, turn_budget=8)
    assert set(env.observe_all()) == {"A", "B"}
    outcomes = env.step_joint(
        {"A": "inspect(sticky_note)", "B": "inspect(mat)"},
    )
    assert outcomes["A"].valid and outcomes["B"].valid
    assert env.env_step == 1
