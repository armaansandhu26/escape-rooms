"""Composable exploration puzzles with seeded placements and clue-to-lock wiring.

Fixtures are searched, portable items are taken, and clues are read privately.
The reference follows generated dependencies rather than guessing fixed locations.
All routing annotations and solutions belong to the evaluator, never prompts.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import random

from motiverooms.connected_generator import ConnectedEpisode, ConnectedRoomsConfig, ROOM_NAMES
from motiverooms.schema import Item, Node, Passage, Requirement, Room


@dataclass(frozen=True)
class SearchFixture:
    id: str
    description: str


# Each fixture can hide items or readable clue nodes using the same engine rules.
SEARCH_FIXTURES = (
    SearchFixture("doormat", "A loose doormat; inspect underneath it."),
    SearchFixture("bookshelf", "A bookshelf with gaps between the books; inspect to search."),
    SearchFixture("calendar", "A desk calendar with a pocket behind its pages; inspect to search."),
    SearchFixture("tool_cabinet", "An unlocked tool cabinet; inspect its compartments."),
    SearchFixture("sofa", "A sofa with loose cushions; inspect between them."),
    SearchFixture("noticeboard", "A noticeboard with overlapping papers; inspect behind them."),
)


class ReferenceRoute:
    """Build a feasibility witness through the actual open map, not an oracle."""

    def __init__(self, locations: dict[str, str], passages: tuple[Passage, ...]):
        self.actions: list[str] = []
        self.current = next(iter(locations))
        self.adjacency: dict[str, list[str]] = {location: [] for location in locations}
        for passage in passages:
            a, b = passage.locations
            self.adjacency[a].append(b)
            self.adjacency[b].append(a)

    def visit(self, destination: str) -> None:
        pending = deque([(self.current, [])])
        visited = {self.current}
        while pending:
            at, route = pending.popleft()
            if at == destination:
                self.actions.extend(f"move({step})" for step in route)
                self.current = destination
                return
            for neighbor in self.adjacency[at]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append((neighbor, route + [neighbor]))
        raise ValueError("Generated exploration map is disconnected")

    def act(self, node: Node, *actions: str) -> None:
        assert node.location is not None
        self.visit(node.location)
        self.actions.extend(actions)


def generate_varied_exploration_episode(
    seed: int, config: ConnectedRoomsConfig | None = None,
) -> ConnectedEpisode:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    count = (config or ConnectedRoomsConfig()).num_rooms
    rng = random.Random(f"exploration-v2:{seed}")
    names = ("Office", "Lounge", "Storage") + ROOM_NAMES[3:]
    locations = {f"room_{i + 1}": f"{names[i % len(names)]} {i + 1}" for i in range(count)}
    location_ids = list(locations)
    office = location_ids[0]

    # Deal visible objects across all rooms, rather than concentrating puzzle
    # objects in the first three. Children inherit their parent's placement.
    placement_order = rng.sample(location_ids, count)
    placement_index = 0
    nodes: dict[str, Node] = {}

    def add(id: str, kind: str, desc: str, *, location: str | None = None, **kwargs) -> Node:
        nonlocal placement_index
        if location is None:
            location = placement_order[placement_index % count]
            placement_index += 1
        node = Node(id, "shared", kind, desc, location=location, **kwargs)
        nodes[id] = node
        return node

    def hide(parent_id: str, child_id: str) -> None:
        parent = nodes[parent_id]
        nodes[parent_id] = replace(parent, contents=parent.contents + (child_id,))

    fixtures = list(SEARCH_FIXTURES)
    rng.shuffle(fixtures)
    for fixture in fixtures:
        add(fixture.id, "search", fixture.description)
    # More rooms add real searchable surfaces, not just longer movement paths.
    for index in range(len(fixtures), max(len(fixtures), count * 2)):
        fixture = rng.choice(SEARCH_FIXTURES)
        add(f"{fixture.id}_{index + 1}", "search", fixture.description)
    sources = list(nodes)
    key_source, token_source, receipt_source, calendar_source = rng.sample(sources, 4)

    items = {id: Item(id, name) for id, name in (
        ("desk_key", "small desk key"), ("batteries", "torch batteries"),
        ("uv_torch", "working UV torch"), ("brass_token", "brass token"),
    )}
    hide(key_source, "desk_key")
    hide(token_source, "brass_token")

    # Codes have unique values, but their targets and physical clue sources vary.
    lock_codes = rng.sample(range(10000), 2)
    codes = {f"mechanism_code_{i + 1}": f"{value:04d}" for i, value in enumerate(lock_codes)}
    targets = rng.sample(["supply_locker", "document_safe"], 2)
    clue_sources = dict(zip(("shipping_receipt", "calendar_annotation"), (receipt_source, calendar_source)))
    code_clues = {}
    for index, (clue_id, source) in enumerate(clue_sources.items(), 1):
        target = targets[index - 1]
        code_id = f"mechanism_code_{index}"
        description = ("A shipping receipt" if clue_id == "shipping_receipt" else "An annotated calendar page")
        add(clue_id, "clue", f"{description} labelled 'Code for {target}'.",
            location=nodes[source].location, hidden=True, code=code_id)
        hide(source, clue_id)
        code_clues[target] = (clue_id, code_id, source)

    add("desk_drawer", "container", "A desk drawer with a small keyhole.",
        requires=Requirement(items=("desk_key",)), contents=("unpowered_torch",))
    add("unpowered_torch", "container", "An unpowered UV torch. Use batteries here, then take the working torch.",
        location=nodes["desk_drawer"].location, hidden=True,
        requires=Requirement(items=("batteries",)), contents=("uv_torch",))
    add("supply_locker", "container", "A supply locker with a numeric keypad.",
        requires=Requirement(code=code_clues["supply_locker"][1]), contents=("batteries",))
    add("document_safe", "container", "A document safe with a numeric keypad.",
        requires=Requirement(code=code_clues["document_safe"][1]), contents=("safe_inscription",))
    add("painting", "container", "A painting with fluorescent markings; illuminate it using a working UV torch.",
        requires=Requirement(items=("uv_torch",)), contents=("painting_inscription",))
    add("display_case", "container", "A display case with a brass-token-shaped slot.",
        requires=Requirement(items=("brass_token",)), contents=("case_inscription",))

    symbols = ["sun", "moon", "star"] + [f"symbol_{i}" for i in range(4, count + 1)]
    fragments = {symbol: f"{rng.randrange(100):02d}" for symbol in symbols}
    symbol_sources = rng.sample(symbols[:3], 3)
    for parent, child, symbol in zip(
        ("painting", "display_case", "document_safe"),
        ("painting_inscription", "case_inscription", "safe_inscription"), symbol_sources,
    ):
        add(child, "clue", "A symbol-marked inscription.", location=nodes[parent].location,
            hidden=True, text=f"{symbol}: {fragments[symbol]}")
    extra_sources = {}
    for i in range(4, count + 1):
        location = f"room_{i}"
        source = rng.choice([id for id in sources if nodes[id].location == location])
        child = f"inscription_{i}"
        add(child, "clue", "A symbol-marked note.", location=location, hidden=True,
            text=f"symbol_{i}: {fragments[f'symbol_{i}']}")
        hide(source, child)
        extra_sources[child] = source
    order = rng.sample(symbols, len(symbols))
    codes["exit_code"] = "".join(fragments[symbol] for symbol in order)
    add("wall_plaque", "clue", "A plaque explaining the final exit's symbol sequence.",
        text=f"For the exit, concatenate the two-digit inscriptions in this order: {' → '.join(order)}. Preserve leading zeros.")
    add("final_exit", "door", "The final exit. Combine the inscriptions in the wall plaque's order.",
        location=office, requires=Requirement(code="exit_code"))

    pairs = list(zip(location_ids, location_ids[1:]))
    if count > 2:
        pairs.append((location_ids[-1], office))
    passages = tuple(Passage(f"passage_{i}", pair, "An open doorway.") for i, pair in enumerate(pairs, 1))
    route = ReferenceRoute(locations, passages)

    def acquire(source: str, item: str) -> None:
        route.act(nodes[source], f"inspect({source})", f"take({item})")

    def unlock(target: str) -> None:
        clue, code, source = code_clues[target]
        route.act(nodes[source], f"inspect({source})", f"inspect({clue})")
        route.act(nodes[target], f"enter({codes[code]}, {target})")

    acquire(key_source, "desk_key")
    route.act(nodes["desk_drawer"], "use(desk_key, desk_drawer)")
    unlock("supply_locker")
    route.actions.append("take(batteries)")
    route.act(nodes["unpowered_torch"], "use(batteries, unpowered_torch)", "take(uv_torch)")
    route.act(nodes["painting"], "use(uv_torch, painting)", "inspect(painting_inscription)")
    acquire(token_source, "brass_token")
    route.act(nodes["display_case"], "use(brass_token, display_case)", "inspect(case_inscription)")
    unlock("document_safe")
    route.actions.append("inspect(safe_inscription)")
    for child, source in extra_sources.items():
        route.act(nodes[source], f"inspect({source})", f"inspect({child})")
    route.act(nodes["wall_plaque"], "inspect(wall_plaque)")
    route.act(nodes["final_exit"], f"enter({codes['exit_code']}, final_exit)")

    node_list = list(nodes.values())
    rng.shuffle(node_list)
    room = Room(id=f"exploration_v2_{count}_{seed}", wings={"A": "building", "B": "building"},
                codes=codes, items=items, nodes=tuple(node_list), layout="shared", space="building",
                locations=locations, passages=passages, start_locations={"A": office, "B": office})
    metadata = dict(generator="exploration_rooms", generator_version=2, seed=seed, num_rooms=count,
                    topology="open_ring" if count > 2 else "open_chain", family="exploration_rooms",
                    profile="varied", split="generated", handoffs=[],
                    reference=[{"A": action, "B": "wait()"} for action in route.actions],
                    reference_actions=route.actions, reference_joint_steps=len(route.actions),
                    recommended_turn_budget=len(route.actions) + max(16, count * 4),
                    item_sources={"desk_key": key_source, "brass_token": token_source, "batteries": "supply_locker"},
                    code_targets={clue: target for target, (clue, _, _) in code_clues.items()},
                    note="Constructive feasibility witness, not an optimal route or a forced cooperation requirement.")
    return ConnectedEpisode(room, metadata)
