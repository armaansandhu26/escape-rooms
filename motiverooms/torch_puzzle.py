"""Explicit torch affordances and a small single-agent dependency diagnostic."""
from __future__ import annotations

from dataclasses import replace
import random

from motiverooms.connected_generator import ConnectedEpisode
from motiverooms.schema import Item, Node, Requirement, Room


TORCH_STATION_DESCRIPTION = (
    "A fixed torch assembly station containing an unpowered UV torch. "
    "This mechanism cannot be picked up. Carry batteries and use them on this "
    "station; then take the working uv_torch revealed here."
)
TORCH_DRAWER_DESCRIPTION = (
    "A desk drawer with a small keyhole. Its contents include a fixed torch "
    "assembly station; only objects listed as available items can be taken."
)


def clarify_torch_affordance(episode: ConnectedEpisode) -> ConnectedEpisode:
    """Change descriptions only; preserve placements, requirements and solutions."""
    descriptions = {"unpowered_torch": TORCH_STATION_DESCRIPTION,
                    "desk_drawer": TORCH_DRAWER_DESCRIPTION}
    nodes = tuple(replace(node, desc=descriptions[node.id]) if node.id in descriptions else node
                  for node in episode.room.nodes)
    return ConnectedEpisode(
        replace(episode.room, id=episode.room.id + "_station", nodes=nodes),
        dict(episode.metadata, torch_affordance="explicit_station"),
    )


def generate_torch_episode(seed: int) -> ConnectedEpisode:
    """Six necessary actions, one worker, no navigation or controller overhead.

    Seeds vary code values, battery source and object ordering. All prerequisites
    are visible initially except the output torch and the painting's inscription.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = random.Random(f"torch-dependency-v1:{seed}")
    code = f"{rng.randrange(10000):04d}"
    source = rng.choice(("shelf", "workbench"))
    items = {item: Item(item, name) for item, name in (
        ("batteries", "torch batteries"), ("uv_torch", "working UV torch"),
    )}
    nodes = [
        Node(source, "shared", "spot", "A surface holding a pack of batteries.", items=("batteries",)),
        Node("unpowered_torch", "shared", "container", TORCH_STATION_DESCRIPTION,
             requires=Requirement(items=("batteries",)), contents=("uv_torch",)),
        Node("painting", "shared", "container",
             "A painting with fluorescent markings. Use a working UV torch to reveal its inscription.",
             requires=Requirement(items=("uv_torch",)), contents=("painting_inscription",)),
        Node("painting_inscription", "shared", "clue", "The final exit's code.",
             hidden=True, code="exit_code"),
        Node("final_exit", "shared", "door", "The exit has a four-digit code lock.",
             requires=Requirement(code="exit_code")),
    ]
    rng.shuffle(nodes)
    room = Room(id=f"torch_dependency_{seed}", wings={"A": "workroom"},
                codes={"exit_code": code}, items=items, nodes=tuple(nodes),
                layout="shared", space="workroom")
    actions = ["take(batteries)", "use(batteries, unpowered_torch)", "take(uv_torch)",
               "use(uv_torch, painting)", "inspect(painting_inscription)", f"enter({code}, final_exit)"]
    metadata = dict(generator="torch_dependency", generator_version=1, seed=seed, num_rooms=1,
                    family="torch_dependency", profile="isolated", split="generated", handoffs=[],
                    torch_affordance="explicit_station", reference=[{"A": action} for action in actions],
                    reference_actions=actions, reference_joint_steps=len(actions),
                    recommended_turn_budget=12, oracle={"A": {"solo_turns": len(actions)}},
                    note="Isolated tool/dependency diagnostic; no navigation, search or cooperation test.")
    return ConnectedEpisode(room, metadata)
