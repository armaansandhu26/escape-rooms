"""Transparent, suite-specific scripted baselines (not learned policies).

Plans know dependency structure. Code actions follow clue inspections; these
baselines are instrumentation controls, not independent tests of puzzle reasoning.
"""
from motiverooms.actions import parse_action


class ScriptedPolicy:
    def __init__(self, agent, env, metadata, kind="silent"):
        if kind not in {"silent", "honest", "deceptive", "reference", "idle"}:
            raise ValueError(f"Unknown scripted policy: {kind}")
        self.agent, self.env, self.kind = agent, env, kind
        self.index = 0
        self.tried_messages = set()
        self.handoffs = metadata.get("handoffs", [])
        self.helper_count = len(metadata["helper_plans"][agent])
        self.policy_version = 1 if kind in {"reference", "idle"} else 2
        self.sent_actions = set()
        self.requested_items = set()
        self.helper_plan = list(metadata["helper_plans"][agent]) if kind in {"honest", "deceptive"} else []
        solo = metadata["solo_plans"][agent]
        if kind == "reference":
            self.plan = [step.get(agent, "wait()") for step in metadata["reference"]]
        elif kind in {"honest", "deceptive"}:
            self.plan = list(metadata["helper_plans"][agent]) + list(solo)
        else:
            self.plan = list(solo)

    def choose_action(self, observation):
        del observation
        if self.kind == "idle":
            return "wait()", "wait()"
        if self.kind == "reference":
            step = self.env.env_step
            line = self.plan[step] if step < len(self.plan) else "wait()"
            return line, line
        engine = self.env.engine
        # A scripted partner can use a received item or an explicitly targeted
        # code claim, including a false claim; failed claims are not retried.
        helper_action = self._next_feasible(self.helper_plan)
        if helper_action is not None:
            return helper_action, helper_action
        if self.kind in {"honest", "deceptive"}:
            for h in self.handoffs:
                if h["recipient"] != self.agent or h["target"] in engine.opened_nodes:
                    continue
                if h["kind"] == "item" and h["resource"] in engine.agents[self.agent].bag:
                    line = f"use({h['resource']}, {h['target']})"
                    return line, line
                if h["kind"] == "code":
                    import re
                    for message in reversed(engine.messages):
                        match = re.search(r"\bis (\d{4})\b", message.text)
                        if message.sender != self.agent and h["target"] in message.text and match:
                            line = f"enter({match.group(1)}, {h['target']})"
                            if line not in self.tried_messages:
                                self.tried_messages.add(line)
                                return line, line
        line = self._next_feasible(self.plan)
        if line is not None:
            return line, line
        if self.kind in {"honest", "deceptive"} and engine.room.is_shared:
            # Request use of a missing item only after exhausting work this seat
            # can do. Never inspect the partner's private bag or current action.
            for node in engine.visible_nodes(self.agent):
                if node.id in engine.opened_nodes or not node.requires:
                    continue
                for item in node.requires.items:
                    request = (item, node.id)
                    if item in engine.taken_items and item not in engine.agents[self.agent].bag and request not in self.requested_items:
                        self.requested_items.add(request)
                        line = f"say(Please use {item} on {node.id}; I am not carrying it.)"
                        return line, line
        return "wait()", "wait()"

    def _next_feasible(self, plan):
        """Recheck a dependency plan against this seat's current resources.

        A blocked action is retained for a later tick. Other feasible work can
        proceed; unavailable items are never assumed to be in this seat's bag.
        """
        engine = self.env.engine
        state = engine.agents[self.agent]
        visible = {n.id for n in engine.visible_nodes(self.agent)}
        for line in plan:
            action = parse_action(line)
            if action.verb == "wait":
                continue  # Golden-trace scheduling waits aren't dependency work.
            if action.verb in {"say", "write_note"}:
                if line in self.sent_actions:
                    continue
                # A prewritten message may name a code only after this agent
                # acquired it; a hidden helper source cannot be bypassed.
                import re
                match = re.search(r"\bis (\d{4})\b", action.args[0])
                if match and match.group(1) not in state.read_codes.values():
                    continue
            if action.verb in {"inspect", "enter", "use"} and action.args[-1] not in visible:
                continue
            if action.verb in {"enter", "use"} and action.args[-1] in engine.opened_nodes:
                continue
            if action.verb == "inspect" and action.args[0] in state.read_codes:
                continue
            if action.verb == "enter" and action.args[0] not in state.read_codes.values():
                continue
            if action.verb in {"use", "give"} and action.args[0] not in state.bag:
                continue
            if action.verb == "take":
                if action.args[0] in engine.taken_items or engine._available_item_source(self.agent, action.args[0]) is None:
                    continue
            original_line = line
            if self.kind == "deceptive" and action.verb == "say":
                import re
                line = re.sub(r"(\d{4})(?=\)$)", lambda m: f"{(int(m[1]) + 1) % 10000:04d}", line)
            if action.verb in {"say", "write_note"}:
                self.sent_actions.add(original_line)
            return line
        return None
