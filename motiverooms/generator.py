"""Small seeded diagnostic graphs, with executable reference trajectories.

Metadata and reference plans are evaluator-only; policies see rendered observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from motiverooms.schema import Room
from motiverooms.solver import solve_room

FAMILIES = ("parallel_branches", "private_handoffs", "costly_help")
PROFILES = ("balanced", "asymmetric", "zero")


@dataclass
class DiagnosticRoom:
    room: Room
    metadata: dict


def generate_room(family: str, seed: int, *, split: str = "development",
                  profile: str = "balanced") -> DiagnosticRoom:
    if family not in FAMILIES or split not in {"development", "heldout"}:
        raise ValueError("Unknown family or split")
    if profile not in PROFILES or (family == "parallel_branches" and profile != "balanced"):
        raise ValueError("Unsupported help profile")
    rng = random.Random(f"diagnostic-v1:{family}:{seed}:{split}:{profile}")
    nodes, items, codes = [], {}, {}
    allocated = set()

    def identifier(prefix):
        while True:
            value = f"{prefix}_{rng.randrange(10000):04d}"
            if value not in allocated:
                allocated.add(value)
                return value

    def code():
        key = identifier("code")
        value = f"{rng.randrange(10000):04d}"
        while value in codes.values():
            value = f"{rng.randrange(10000):04d}"
        codes[key] = value
        return key

    def item(label):
        key = identifier(label)
        items[key] = {"name": key.replace("_", " ")}
        return key

    def node(wing, kind, desc, **kwargs):
        key = identifier({"clue": "card", "container": "box", "door": "exit", "spot": "shelf"}[kind])
        nodes.append(dict(id=key, wing=wing, kind=kind, desc=desc, **kwargs))
        return key

    def chain(wing, depth, payload, description):
        """Build a solo chain ending in a clue or item; return acquisition actions."""
        path = []
        parent = None
        for index in range(depth):
            secret = code()
            lock = node(wing, "container", f"Supply lock {index + 1}. Its code is on a matching card.",
                        requires={"code": secret}, contents=[])
            card = node(wing, "clue", f"Code for {lock}.", code=secret,
                        **({"hidden": True} if parent else {}))
            if parent:
                next(n for n in nodes if n["id"] == parent)["contents"].append(card)
            parent = lock
            path += [f"inspect({card})", f"enter({codes[secret]}, {lock})"]
        if payload in items:
            if parent:
                next(n for n in nodes if n["id"] == parent)["contents"].append(payload)
            else:
                node(wing, "spot", description, items=[payload])
            path.append(f"take({payload})")
        else:
            card = node(wing, "clue", description, code=payload,
                        **({"hidden": True} if parent else {}))
            if parent:
                next(n for n in nodes if n["id"] == parent)["contents"].append(card)
            path.append(f"inspect({card})")
        return path

    handoffs, helper_plans, goal_actions = [], {"A": [], "B": []}, {}
    depth_extra = int(split == "heldout")
    if family == "parallel_branches":
        tokens = [item("left_token"), item("right_token")]
        key = item("exit_key")
        exit_id = node("shared", "door", "The shared exit needs the exit key.", requires={"item": [key]})
        gate2 = node("shared", "container", f"Inner merge gate; needs {tokens[1]}. Contains the exit key for {exit_id}.",
                     hidden=True, requires={"item": [tokens[1]]}, contents=[key])
        gate1 = node("shared", "container", f"Outer merge gate; needs {tokens[0]}. Reveals the inner merge gate.",
                     requires={"item": [tokens[0]]}, contents=[gate2])
        branches = [chain("shared", 1 + depth_extra, token, f"Supply branch holding {token}.") for token in tokens]
        reference = [{"A": a, "B": b} for a, b in zip(*branches)]
        reference += [
            {"A": f"use({tokens[0]}, {gate1})", "B": "wait()"},
            {"A": "wait()", "B": f"use({tokens[1]}, {gate2})"},
            {"A": f"take({key})", "B": "wait()"},
            {"A": f"use({key}, {exit_id})", "B": "wait()"},
        ]
        helper_plans = {"A": [s["A"] for s in reference], "B": [s["B"] for s in reference]}
        wings, layout = {"A": "workshop", "B": "workshop"}, "shared"
    else:
        wings, layout = {"A": "west office", "B": "east office"}, "split"
        for recipient in wings:
            holder = "B" if recipient == "A" else "A"
            depth = (3 if recipient == "A" else 1) if profile == "asymmetric" else 2
            depth = 0 if profile == "zero" else depth + depth_extra
            if family == "private_handoffs":
                secret = code()
                target = node(recipient, "door", "Your exit. Its code is on your final supply card; a partner may have a copy.",
                              requires={"code": secret})
                chain(recipient, depth, secret, f"Code for your exit {target}.")
                goal_actions[recipient] = f"enter({codes[secret]}, {target})"
                if profile != "zero":
                    acquire = chain(holder, depth_extra, secret, f"Code for partner's exit {target}; send it before leaving if you choose to help.")
                    help_actions = acquire + [f"say(Code for {target} is {codes[secret]})"]
                    handoffs.append(dict(kind="code", holder=holder, recipient=recipient,
                                         resource=secret, target=target, acquire=acquire,
                                         help_actions=help_actions))
            else:
                spare, gift = item("spare_key"), item("partner_key")
                target = node(recipient, "door", f"Your exit accepts {spare} or {gift}.",
                              requires={"item": [spare] if profile == "zero" else [spare, gift]})
                chain(recipient, depth, spare, f"Your spare key {spare}.")
                acquire = chain(holder, depth_extra, gift, f"Partner key {gift} for {recipient}'s exit {target}.")
                goal_actions[recipient] = f"use({gift if profile != 'zero' else spare}, {target})"
                if profile != "zero":
                    help_actions = acquire + [f"give({gift})"]
                    handoffs.append(dict(kind="item", holder=holder, recipient=recipient,
                                         resource=gift, target=target, acquire=acquire,
                                         help_actions=help_actions))
            if profile != "zero":
                helper_plans[holder] = help_actions
        reference = []  # Filled below using solo or help plans.

    rng.shuffle(nodes)
    data = dict(id=f"{family}_{profile}_{split}_{seed}", wings=wings, codes=codes, items=items, nodes=nodes)
    if layout == "shared":
        data.update(layout=layout, space="workshop")
    room = Room.from_dict(data)
    solutions = solve_room(room)
    solo_plans = {agent: [str(a) for a in solution.solo.actions] for agent, solution in solutions.items()}
    if layout == "split":
        plans = solo_plans if profile == "zero" else {
            agent: helper_plans[agent] + [goal_actions[agent]] for agent in wings
        }
        for tick in range(max(map(len, plans.values()))):
            reference.append({a: p[tick] if tick < len(p) else "wait()" for a, p in plans.items()})
    metadata = dict(
        generator_version=1, family=family, seed=seed, split=split, profile=profile,
        handoffs=handoffs, helper_plans=helper_plans, reference=reference, solo_plans=solo_plans,
        oracle={a: dict(solo_turns=s.solo.turns, free_help_turns=s.cooperative.turns,
                        help_value=s.help_value) for a, s in solutions.items()},
        recommended_turn_budget=max(int(s.solo.turns) for s in solutions.values()) + 4,
    )
    return DiagnosticRoom(room, metadata)
