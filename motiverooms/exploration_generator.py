"""Open-room exploration puzzles with concealed objects and distributed clues.

Only the evaluator receives the assembled code and constructive reference route.
Search and item dependencies use ordinary engine actions, with no puzzle-specific
handlers or privileged hints in the agents' observations.
"""
from __future__ import annotations

from collections import deque
import random

from motiverooms.connected_generator import ConnectedEpisode, ConnectedRoomsConfig
from motiverooms.schema import Item, Node, Passage, Requirement, Room
from motiverooms.varied_exploration import generate_varied_exploration_episode
from motiverooms.torch_puzzle import clarify_torch_affordance


def generate_exploration_episode(
    seed: int, config: ConnectedRoomsConfig | None = None, *, version: int = 2,
    torch_affordance: str | None = None,
) -> ConnectedEpisode:
    """Generate varied puzzles by default; version 1 reproduces the original pilot."""
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("Exploration generator version must be 1 or 2")
    affordance = torch_affordance if torch_affordance is not None else (
        "explicit_station" if version == 2 else "legacy")
    if not isinstance(affordance, str) or affordance not in {"legacy", "explicit_station"}:
        raise ValueError("torch_affordance must be legacy or explicit_station")
    if version == 2:
        episode = generate_varied_exploration_episode(seed, config)
    else:
        episode = _generate_classic_episode(seed, config)
    return clarify_torch_affordance(episode) if affordance == "explicit_station" else episode


def _generate_classic_episode(seed: int, config: ConnectedRoomsConfig | None) -> ConnectedEpisode:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    count = (config or ConnectedRoomsConfig()).num_rooms
    rng = random.Random(f"exploration-v1:{seed}")
    names = ("Office", "Storage", "Lounge", "Archive", "Study", "Workshop")
    locations = {f"room_{i + 1}": f"{names[i % len(names)]} {i + 1}" for i in range(count)}
    office = "room_1"
    destinations = list(locations)[1:3]
    rng.shuffle(destinations)
    storage = destinations[0] if destinations else office
    lounge = destinations[1] if len(destinations) > 1 else storage
    if count > 2:
        locations[storage] = f"Storage {storage.removeprefix('room_')}"
        locations[lounge] = f"Lounge {lounge.removeprefix('room_')}"
    elif count == 2:
        locations[storage] = "Storage / Lounge 2"
    items = {i: Item(i, name) for i, name in (
        ("desk_key", "small desk key"), ("batteries", "torch batteries"),
        ("uv_torch", "working UV torch"), ("brass_token", "brass token"))}
    symbols = ["sun", "moon", "star"] + [f"symbol_{i}" for i in range(4, count + 1)]
    fragments = {symbol: f"{rng.randrange(100):02d}" for symbol in symbols}
    order = rng.sample(symbols, len(symbols))
    exit_code = "".join(fragments[s] for s in order)
    nodes: list[Node] = []

    def add(id, kind, desc, location, **kwargs):
        nodes.append(Node(id, "shared", kind, desc, location=location, **kwargs))

    def fragment(id, symbol, location, hidden=False):
        add(id, "clue", "A small symbol-marked inscription.", location,
            hidden=hidden, text=f"{symbol}: {fragments[symbol]}")

    add("doormat", "search", "A loose doormat with a raised corner; inspect to look underneath.", office,
        contents=("desk_key",))
    add("desk_drawer", "container", "A desk drawer with a small keyhole.", office,
        requires=Requirement(items=("desk_key",)), contents=("unpowered_torch",))
    add("unpowered_torch", "container", "An unpowered UV torch. Use batteries here, then take the working torch.",
        office, hidden=True, requires=Requirement(items=("batteries",)), contents=("uv_torch",))
    add("tool_cabinet", "search", "An unlocked tool cabinet; inspect to search its compartments.", storage,
        contents=("batteries",))
    add("painting", "container", "A painting with faint fluorescent markings; illuminate it using a working UV torch.",
        lounge, requires=Requirement(items=("uv_torch",)), contents=("painting_inscription",))
    fragment("painting_inscription", "moon", lounge, hidden=True)
    add("sofa", "search", "A sofa with loose cushions; inspect to search between them.", lounge,
        contents=("brass_token",))
    add("display_case", "container", "A display case with a brass-token-shaped slot.", lounge,
        requires=Requirement(items=("brass_token",)), contents=("case_inscription",))
    fragment("case_inscription", "sun", lounge, hidden=True)
    add("shipping_receipt", "clue", "A shipping receipt bearing a handwritten symbol and two digits.", storage,
        text=f"star: {fragments['star']}")
    add("wall_plaque", "clue", "A wall plaque explaining the exit keypad's symbol sequence.", lounge,
        text=f"For the exit, concatenate the two-digit inscriptions in this order: {' → '.join(order)}. Preserve leading zeros.")
    add("final_exit", "door", "The building's final exit. Combine the symbol inscriptions in the order on the wall plaque.",
        office, requires=Requirement(code="exit_code"))
    for i in range(4, count + 1):
        fragment(f"inscription_{i}", f"symbol_{i}", f"room_{i}")
    for location in locations:
        add(f"books_{location}", "scenery", "A shelf of ordinary books; inspect for details.", location,
            text="Ordinary books with no inscriptions, loose objects or hidden compartments.")
        add(f"calendar_{location}", "scenery", "A faded wall calendar; inspect for details.", location,
            text="A faded calendar with no marked dates or puzzle symbols.")

    pairs = [(f"room_{i}", f"room_{i + 1}") for i in range(1, count)]
    if count > 2:
        pairs.append((f"room_{count}", office))
    passages = tuple(Passage(f"passage_{i}", pair, "An open doorway.")
                     for i, pair in enumerate(pairs, 1))
    adjacency = {location: [] for location in locations}
    for a, b in pairs:
        adjacency[a].append(b)
        adjacency[b].append(a)
    actions: list[str] = []
    current = office

    def visit(destination):
        nonlocal current
        pending = deque([(current, [])])
        visited = {current}
        while pending:
            at, route = pending.popleft()
            if at == destination:
                actions.extend(f"move({step})" for step in route)
                current = destination
                return
            for neighbor in adjacency[at]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append((neighbor, route + [neighbor]))
        raise ValueError("Generated rooms are disconnected")

    actions.extend(("inspect(doormat)", "take(desk_key)", "use(desk_key, desk_drawer)"))
    visit(storage)
    actions.extend(("inspect(tool_cabinet)", "take(batteries)", "inspect(shipping_receipt)"))
    visit(office)
    actions.extend(("use(batteries, unpowered_torch)", "take(uv_torch)"))
    visit(lounge)
    actions.extend(("use(uv_torch, painting)", "inspect(painting_inscription)",
                    "inspect(sofa)", "take(brass_token)", "use(brass_token, display_case)",
                    "inspect(case_inscription)", "inspect(wall_plaque)"))
    for i in range(4, count + 1):
        visit(f"room_{i}")
        actions.append(f"inspect(inscription_{i})")
    visit(office)
    actions.append(f"enter({exit_code}, final_exit)")
    rng.shuffle(nodes)
    room = Room(id=f"exploration_{count}_{seed}", wings={"A": "building", "B": "building"},
                codes={"exit_code": exit_code}, items=items, nodes=tuple(nodes), layout="shared", space="building",
                locations=locations, passages=passages, start_locations={"A": office, "B": office})
    metadata = dict(generator="exploration_rooms", generator_version=1, seed=seed, num_rooms=count,
                    topology="open_ring" if count > 2 else "open_chain", family="exploration_rooms",
                    profile="shared", split="generated", handoffs=[],
                    reference=[{"A": action, "B": "wait()"} for action in actions], reference_actions=actions,
                    reference_joint_steps=len(actions), recommended_turn_budget=len(actions) + max(16, count * 4),
                    note="Reference demonstrates feasibility, not an optimal solution. No multi-worker dependency is forced.")
    return ConnectedEpisode(room, metadata)
