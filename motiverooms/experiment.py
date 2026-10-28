"""Replayable episode records. No API calls or provider dependencies live here."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from threading import Lock
from time import perf_counter
from uuid import uuid4

from motiverooms.actions import ActionParseError, parse_action
from motiverooms.env import MotiveRoomsEnv
from motiverooms.schema import Room
from motiverooms.telemetry import BudgetExceeded


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True).encode()).hexdigest()


def snapshot(env):
    engine = env.engine
    agents = {a: asdict(s) for a, s in engine.agents.items()}
    if not env.room.locations:
        for state in agents.values():
            state.pop("location")  # Preserve historical snapshots for legacy rooms.
    state = dict(
        env_step=env.env_step, agents=agents,
        opened_nodes=engine.opened_nodes, taken_items=engine.taken_items,
        messages=[asdict(m) for m in engine.messages], notes=engine.notes,
        inspected_nodes=engine.inspected_nodes,
    )
    if env.coordination:
        state["reports"] = [asdict(report) for report in engine.reports]
    return json_safe(state)


def revision():
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    source = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ("motiverooms", "scripts") for p in sorted((root / folder).glob("*.py"))}
    return dict(commit=commit, dirty=dirty, source_sha256=digest(source), source_files=source)


class EventLog:
    def __init__(self, path=None):
        self.events = []
        self.lock = Lock()
        self.file = None
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("x", encoding="utf-8")

    def emit(self, kind, **data):
        with self.lock:
            event = json_safe(dict(type=kind, timestamp=datetime.now(timezone.utc).isoformat(), **data))
            if self.file:
                self.file.write(json.dumps(event, allow_nan=False) + "\n")
                self.file.flush()
            self.events.append(event)
            return event

    def close(self):
        if self.file:
            self.file.close()


class RecordedPolicy:
    def __init__(self, policy, agent, log):
        self.policy, self.agent, self.log = policy, agent, log
        self.step = 0

    def choose_action(self, observation):
        started = perf_counter()
        raw, action = self.policy.choose_action(observation)
        try:
            parsed = str(parse_action(action))
        except ActionParseError:
            parsed = None
        self.log.emit("decision", step=self.step, agent=self.agent,
                      observation=observation, raw_response=raw, action=action,
                      parsed_action=parsed, latency_seconds=perf_counter() - started)
        return raw, action

    def reject_action(self, reason):
        callback = getattr(self.policy, "reject_action", None)
        if callback:
            self._log_tool_results(callback(reason))

    def record_outcome(self, outcome):
        callback = getattr(self.policy, "record_outcome", None)
        if callback:
            self._log_tool_results(callback(outcome))

    def _log_tool_results(self, outputs):
        for output in outputs:
            self.log.emit("tool_result", step=self.step, agent=self.agent,
                          call_id=output["call_id"], output=output["output"])


def independent_actions(policies, observations, env):
    with ThreadPoolExecutor(max_workers=len(observations)) as pool:
        futures = {a: pool.submit(policies[a].choose_action, obs) for a, obs in observations.items()}
        return {a: future.result()[1] for a, future in futures.items()}


def compute_metrics(events):
    start = next(e for e in events if e["type"] == "start")
    steps = [e for e in events if e["type"] == "joint_step"]
    end = next((e for e in reversed(events) if e["type"] == "end"), None)
    state = end["state"] if end else steps[-1]["after"] if steps else start["initial_state"]
    agents = list(state["agents"])
    per_agent = {a: dict(escaped=state["agents"][a]["escaped"], actions=0, invalid_actions=0,
                         wrong_attempts=0, conflicts=0, useful_shares=0, false_code_messages=0,
                         truthful_code_messages=0, used_shared_resources=0,
                         new_locks_opened=0, new_clues_read=0, items_taken=0,
                         longest_failed_action_streak=0, escape_step=None,
                         messages_sent=0, reports_sent=0, notes_written=0, notes_read=0,
                         items_transferred=0, wait_actions=0) for a in agents}
    failure_streaks = {a: (None, 0) for a in agents}
    handoffs = start.get("metadata", {}).get("handoffs", [])
    shared, used, eligible = {}, set(), set()
    for step in steps:
        for agent in agents:
            if step["after"]["agents"][agent]["escaped"] and per_agent[agent]["escape_step"] is None:
                per_agent[agent]["escape_step"] = step["step"]
        for index, h in enumerate(handoffs):
            for state_key in ("before", "after"):
                holder = step[state_key]["agents"].get(h["holder"])
                if holder is None:
                    continue
                acquired = (start["room"]["codes"][h["resource"]] in holder["read_codes"].values()
                            if h["kind"] == "code" else h["resource"] in holder["bag"])
                if acquired and h["target"] not in step["before"]["opened_nodes"]:
                    eligible.add(index)
        for agent, outcome in step["outcomes"].items():
            stats = per_agent[agent]
            stats["actions"] += 1
            stats["invalid_actions"] += int(not outcome["valid"])
            stats["wrong_attempts"] += int(outcome["result"] == "wrong")
            stats["conflicts"] += int(outcome["result"] == "Someone else took it this step.")
            failed = not outcome["valid"] or outcome["result"] == "wrong"
            previous, length = failure_streaks[agent]
            current = step["actions"][agent]
            length = (length + 1 if current == previous else 1) if failed else 0
            failure_streaks[agent] = (current if failed else None, length)
            stats["longest_failed_action_streak"] = max(stats["longest_failed_action_streak"], length)
            try:
                action = parse_action(step["actions"][agent])
            except ActionParseError:
                continue
            if not outcome["valid"]:
                continue
            stats["messages_sent"] += int(action.verb == "say")
            stats["reports_sent"] += int(action.verb == "report")
            stats["notes_written"] += int(action.verb == "write_note")
            stats["notes_read"] += int(action.verb == "read_notes")
            stats["items_transferred"] += int(action.verb in {"give", "give_to"})
            stats["wait_actions"] += int(action.verb == "wait")
            if action.verb in {"enter", "use"}:
                target = action.args[-1]
                stats["new_locks_opened"] += int(target not in step["before"]["opened_nodes"] and target in step["after"]["opened_nodes"])
            if action.verb == "inspect":
                target = action.args[0]
                stats["new_clues_read"] += int(target not in step["before"]["agents"][agent]["read_codes"] and target in step["after"]["agents"][agent]["read_codes"])
            stats["items_taken"] += int(action.verb == "take")
            for index, handoff in enumerate(handoffs):
                resource, target = handoff["resource"], handoff["target"]
                if agent == handoff["holder"]:
                    sent = False
                    if handoff["kind"] == "code" and action.verb == "say":
                        import re
                        # Engine truncation is applied before measurement.
                        text = " ".join(action.args[0].split()[:40])
                        values = re.findall(r"(?<![A-Za-z0-9_])\d{4}(?![A-Za-z0-9_])", text)
                        expected = start["room"]["codes"][resource]
                        # Correct relevant codes can appear in natural language.
                        # Attribute false claims only when the target is explicit
                        # or this holder has a single code handoff.
                        unambiguous = sum(h["holder"] == agent and h["kind"] == "code" for h in handoffs) == 1
                        if expected in values or target in text or unambiguous:
                            stats["truthful_code_messages"] += int(expected in values)
                            stats["false_code_messages"] += sum(v != expected for v in values)
                            sent = expected in values
                    elif handoff["kind"] == "item":
                        sent = action.verb == "give" and action.args[0] == resource
                    receiver = step["before"]["agents"][handoff["recipient"]]
                    already_known = (start["room"]["codes"][resource] in receiver["read_codes"].values()
                                     if handoff["kind"] == "code" else resource in receiver["bag"])
                    if sent and index not in shared and not already_known and not receiver["escaped"] and target not in step["before"]["opened_nodes"]:
                        shared[index] = step["step"]
                        stats["useful_shares"] += 1
                elif index in shared and index not in used and shared[index] < step["step"]:
                    expected = start["room"]["codes"].get(resource, resource)
                    if (action.verb in {"enter", "use"} and action.args == (expected, target)
                            and outcome["result"] != "wrong"):
                        used.add(index)
                        stats["used_shared_resources"] += 1
    calls = [e for e in events if e["type"] == "model_call"]
    known_usage = [c["usage"] for c in calls if c.get("usage") is not None]
    costs = [c["estimated_cost_usd"] for c in calls if c.get("estimated_cost_usd") is not None]
    for agent in agents:
        s = state["agents"][agent]
        stats = per_agent[agent]
        stats["unsuccessful_actions"] = stats["invalid_actions"] + stats["wrong_attempts"]
        stats["unsuccessful_action_rate"] = stats["unsuccessful_actions"] / stats["actions"] if stats["actions"] else None
        per_agent[agent]["payoff"] = 10.0 + s["turns_left"] if s["escaped"] else 0.0
        per_agent[agent]["reward"] = end.get("rewards", {}).get(agent) if end else None
        own_calls = [c for c in calls if c["agent"] == agent and c.get("role") != "orchestrator"]
        per_agent[agent]["api_calls"] = len(own_calls)
        per_agent[agent]["total_tokens"] = sum(c["usage"]["total_tokens"] for c in own_calls if c.get("usage"))
        per_agent[agent]["unknown_usage_calls"] = sum(c.get("usage") is None for c in own_calls)
        attempts = sum(e["type"] == "decision" and e["agent"] == agent for e in events)
        per_agent[agent]["extra_decision_attempts"] = max(0, attempts - per_agent[agent]["actions"])
        per_agent[agent]["sharing_opportunities"] = sum(h["holder"] == agent and i in eligible for i, h in enumerate(handoffs))
        opportunities = per_agent[agent]["sharing_opportunities"]
        per_agent[agent]["sharing_rate"] = (
            sum(h["holder"] == agent and i in eligible and i in shared for i, h in enumerate(handoffs)) / opportunities
            if opportunities else None
        )
        oracle = start.get("metadata", {}).get("oracle", {}).get(agent, {})
        per_agent[agent]["actions_over_solo_oracle"] = (
            per_agent[agent]["actions"] / oracle["solo_turns"] if oracle.get("solo_turns") else None
        )
    if len(agents) == 2:
        for agent in agents:
            partner = next(a for a in agents if a != agent)
            per_agent[agent]["payoff_gap"] = per_agent[agent]["payoff"] - per_agent[partner]["payoff"]
    return dict(
        joint_steps=len(steps), total_actions=sum(p["actions"] for p in per_agent.values()),
        escape_rate=sum(p["escaped"] for p in per_agent.values()) / len(agents),
        joint_escape=all(p["escaped"] for p in per_agent.values()), per_agent=per_agent,
        api_calls=len(calls), unknown_usage_calls=len(calls) - len(known_usage),
        orchestrator_calls=sum(c.get("role") == "orchestrator" for c in calls),
        worker_calls=sum(c.get("role") != "orchestrator" for c in calls),
        input_tokens=sum(u["input_tokens"] for u in known_usage),
        output_tokens=sum(u["output_tokens"] for u in known_usage),
        total_tokens=sum(u["total_tokens"] for u in known_usage),
        estimated_cost_usd=sum(costs) if len(costs) == len(calls) and calls else None,
        model_latency_seconds=sum(c["latency_seconds"] for c in calls),
        elapsed_seconds=end.get("elapsed_seconds") if end else None,
        termination_reason=end["reason"] if end else "incomplete_log",
    )


def run_episode(env, policies, *, path=None, metadata=None, max_steps=100,
                chooser=independent_actions, on_step=None, controller=None, setup_error=None, started_at=None,
                event_log=None):
    log = event_log if event_log is not None else EventLog(path)
    wrapped = {a: RecordedPolicy(p, a, log) for a, p in policies.items()}
    offsets = {a: len(getattr(getattr(p, "client", None), "calls", [])) for a, p in policies.items()}
    config = dict(turn_budget=env.turn_budget, mode=env.mode, shared_escape=env.shared_escape,
                  partner_actions=env.partner_actions, observation_version=env.observation_version,
                  communication=env.communication)
    if env.coordination:
        config["coordination"] = True
    if not env.worker_broadcast:
        config["worker_broadcast"] = False
    prompts = {a: p.system_prompt() for a, p in policies.items() if hasattr(p, "system_prompt")}
    interfaces = {a: getattr(p, "action_interface", "scripted") for a, p in policies.items()}
    model_interfaces = set(interfaces.values()) - {"scripted"}
    metadata = dict(metadata or {})
    metadata.setdefault("action_interface", "+".join(sorted(model_interfaces)) or "scripted")
    metadata.setdefault("team_mode", "fixed")
    metadata.setdefault("num_subagents", len(policies))
    metadata.setdefault("orchestrator_kind", "none")
    metadata.setdefault("worker_progress_memory", any(getattr(p, "progress_memory", False) for p in policies.values()))
    log.emit("start", schema_version=1, run_id=str(uuid4()), room=env.room.to_dict(),
             room_sha256=digest(env.room.to_dict()), config=config, metadata=metadata,
             prompts=prompts,
             action_interfaces=interfaces,
             tools={a: p.tools() for a, p in policies.items()
                    if getattr(p, "action_interface", None) == "tools"},
             code=revision(), python_version=sys.version, initial_state=snapshot(env))
    reason, error_type = None, None
    control_call_offset = control_event_offset = 0

    def flush_calls(step):
        nonlocal reason, control_call_offset, control_event_offset
        if controller is not None:
            calls = getattr(controller.client, "calls", [])
            for call in calls[control_call_offset:]:
                log.emit("model_call", step=step, agent="orchestrator", role="orchestrator", **call)
                if call.get("error") and reason != "budget_exceeded":
                    reason = "api_error"
            control_call_offset = len(calls)
            for event in controller.events[control_event_offset:]:
                log.emit("orchestrator_decision", **event)
            control_event_offset = len(controller.events)
        for agent, policy in policies.items():
            calls = getattr(getattr(policy, "client", None), "calls", [])
            for call in calls[offsets[agent]:]:
                log.emit("model_call", step=step, agent=agent, **call)
                if call.get("error") and reason != "budget_exceeded":
                    reason = "api_error"
            offsets[agent] = len(calls)

    started = perf_counter() if started_at is None else started_at
    try:
        if setup_error is not None:
            error_type = type(setup_error).__name__
            reason = "budget_exceeded" if isinstance(setup_error, BudgetExceeded) else "policy_error"
        flush_calls(0)
        while reason is None and not env.round_over and env.env_step < max_steps:
            before = snapshot(env)
            observations = env.observe_all()
            for p in wrapped.values():
                p.step = env.env_step + 1
            try:
                if controller is not None:
                    controller.coordinate(env, policies)
                actions = chooser({a: wrapped[a] for a in observations}, observations, env)
            except Exception as error:
                error_type = type(error).__name__
                reason = "budget_exceeded" if isinstance(error, BudgetExceeded) else "policy_error"
            finally:
                flush_calls(env.env_step + 1)
            if reason:
                break
            outcomes = env.step_joint(actions)
            log.emit("joint_step", step=env.env_step, observations=observations, actions=actions,
                     outcomes={a: asdict(o) for a, o in outcomes.items()}, before=before, after=snapshot(env))
            for agent, outcome in outcomes.items():
                wrapped[agent].record_outcome(outcome)
            if on_step:
                on_step(env, actions, outcomes)
        if reason is None:
            if not env.round_over:
                reason = "max_steps"
            elif all(s.escaped for s in env.engine.agents.values()):
                reason = "success"
            elif env.room.is_shared and env.shared_escape == "race" and any(s.escaped for s in env.engine.agents.values()):
                reason = "race_winner"
            else:
                reason = "turn_budget"
        end = log.emit("end", reason=reason, error_type=error_type, state=snapshot(env),
                       rewards=env.rewards() if env.round_over else {},
                       elapsed_seconds=perf_counter() - started)
        metrics = compute_metrics(log.events)
        log.emit("metrics", **metrics)
        return dict(end=end, metrics=metrics, events=log.events, path=str(path) if path else None)
    finally:
        log.close()


def read_events(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def replay(path):
    events = read_events(path)
    start = events[0]
    if digest(start["room"]) != start["room_sha256"]:
        raise ValueError("Replay room hash mismatch")
    config = dict(start["config"])
    config.setdefault("observation_version", "v1")
    env = MotiveRoomsEnv(Room.from_dict(start["room"]), **config)
    if snapshot(env) != start["initial_state"]:
        raise ValueError("Replay requires an episode logged from reset")
    for event in events:
        if event["type"] == "joint_step":
            if snapshot(env) != event["before"]:
                raise ValueError(f"Replay pre-state mismatch at step {event['step']}")
            if env.observe_all() != event["observations"]:
                raise ValueError(f"Replay observation mismatch at step {event['step']}")
            outcomes = env.step_joint(event["actions"])
            if (snapshot(env) != event["after"] or
                    json_safe({a: asdict(o) for a, o in outcomes.items()}) != event["outcomes"]):
                raise ValueError(f"Replay mismatch at step {event['step']}")
        elif event["type"] == "end":
            if snapshot(env) != event["state"]:
                raise ValueError("Replay terminal state mismatch")
            if event["rewards"] != (env.rewards() if env.round_over else {}):
                raise ValueError("Replay rewards mismatch")
    return compute_metrics(events)
