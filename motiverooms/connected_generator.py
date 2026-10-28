"""Seeded, solvable episodes spanning a configurable chain of physical rooms.

Generation is constructive: prerequisites for stage i are placed only in rooms
0..i. A reference route proves reachability without an exponential search. The
reference and all code values in metadata are evaluator-only.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from motiverooms.schema import Node, Passage, Requirement, Room, Item


@dataclass(frozen=True)
class ConnectedRoomsConfig:
    num_rooms: int = 3

    def __post_init__(self) -> None:
        if type(self.num_rooms) is not int or self.num_rooms < 1:
            raise ValueError("num_rooms must be a positive integer")


@dataclass(frozen=True)
class ConnectedEpisode:
    room: Room
    metadata: dict


ROOM_NAMES = ("Office", "Storage", "Workshop", "Archive", "Study", "Laboratory")
FURNITURE = ("desk", "bookshelf", "noticeboard", "cabinet", "workbench", "side table")


def generate_connected_episode(seed: int, config: ConnectedRoomsConfig | None = None) -> ConnectedEpisode:
    """Generate one reproducible map; a fresh seed selects a fresh episode."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    config = config or ConnectedRoomsConfig()
    rng = random.Random(f"connected-v1:{seed}")
    count = config.num_rooms
    locations = {f"room_{i + 1}": f"{ROOM_NAMES[i % len(ROOM_NAMES)]} {i + 1}" for i in range(count)}
    location_ids = tuple(locations)
    values = list(range(10000))
    rng.shuffle(values)
    codes: dict[str, str] = {}
    items: dict[str, Item] = {}
    nodes: list[Node] = []
    passages: list[Passage] = []
    actions: list[str] = []
    current = 0

    def allocate_code(name: str) -> str:
        if not values:
            raise ValueError("Episode exhausts the unique four-digit code space")
        codes[name] = f"{values.pop():04d}"
        return name

    def travel_to(index: int) -> None:
        nonlocal current
        while current != index:
            current += 1 if index > current else -1
            actions.append(f"move({location_ids[current]})")

    for stage in range(count):
        # Stage i unlocks room i+1; the last stage unlocks the actual exit.
        number = stage + 1
        clue_location = rng.randrange(number)
        box_location = rng.randrange(number)
        target_id = f"passage_{number}" if number < count else "final_exit"
        box_id, clue_id = f"box_{number}", f"clue_{number}"
        box_code = allocate_code(f"box_code_{number}")
        box_surface, clue_surface = rng.sample(FURNITURE, 2)
        nodes.append(Node(clue_id, "shared", "clue",
                          f"A card on the {clue_surface}; it gives the code for {box_id}.",
                          code=box_code, location=location_ids[clue_location]))
        if rng.choice((True, False)):
            secret = allocate_code(f"lock_code_{number}")
            payload = f"lock_card_{number}"
            nodes.append(Node(payload, "shared", "clue", f"The code for {target_id}.",
                              hidden=True, code=secret, location=location_ids[box_location]))
            requirement = Requirement(code=secret)
            acquire = f"inspect({payload})"
            unlock = f"enter({codes[secret]}, {target_id})"
        else:
            payload = f"key_{number}"
            items[payload] = Item(payload, f"key for {target_id}")
            requirement = Requirement(items=(payload,))
            acquire = f"take({payload})"
            unlock = f"use({payload}, {target_id})"
        nodes.append(Node(box_id, "shared", "container",
                          f"A locked box on the {box_surface}. Its contents unlock {target_id}.",
                          requires=Requirement(code=box_code), contents=(payload,),
                          location=location_ids[box_location]))
        if number < count:
            passages.append(Passage(target_id, (location_ids[stage], location_ids[stage + 1]),
                                    "A connecting doorway.", requirement))
        else:
            nodes.append(Node(target_id, "shared", "door", "The final exit from the building.",
                              requires=requirement, location=location_ids[stage]))

        # At this point all earlier passages are unlocked. Prerequisites are
        # reachable even when a seed puts the card and box in different rooms.
        travel_to(clue_location)
        actions.append(f"inspect({clue_id})")
        travel_to(box_location)
        actions.extend((f"enter({codes[box_code]}, {box_id})", acquire))
        travel_to(stage)
        actions.append(unlock)
        if number < count:
            travel_to(stage + 1)

    rng.shuffle(nodes)
    room = Room(id=f"connected_{count}_{seed}", wings={"A": "building", "B": "building"},
                codes=codes, items=items, nodes=tuple(nodes), layout="shared", space="building",
                locations=locations, passages=tuple(passages),
                start_locations={"A": location_ids[0], "B": location_ids[0]})
    budget = len(actions) + max(4, count * 2)
    metadata = dict(generator="connected_rooms", generator_version=1, seed=seed,
                    num_rooms=count, topology="chain", family="connected_rooms", profile="shared",
                    split="generated", handoffs=[], reference=[{"A": action, "B": "wait()"} for action in actions],
                    reference_actions=actions, reference_joint_steps=len(actions),
                    recommended_turn_budget=budget,
                    note="Constructive reference is a feasibility witness, not an optimal solo or joint solution.")
    return ConnectedEpisode(room, metadata)
