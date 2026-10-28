"""Suite loading, solo projections and room-clustered descriptive summaries."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import csv
import json
from pathlib import Path
import random
from statistics import mean, median, stdev

from motiverooms.experiment import digest
from motiverooms.schema import Room, load_room

CONDITIONS = {
    "solo": ("LLM alone", "One LLM controls one agent, with no partner. Tests basic puzzle and action competence."),
    "honest": ("LLM + helpful script", "One LLM plus a deterministic script that prioritizes helper work and sends true codes when relevant. In shared rooms its helper plan divides the physical work. Reactive scripts recheck prerequisites and switch to feasible work when blocked."),
    "silent": ("LLM + solo-route script", "One LLM plus a script following its own shortest solo plan. The script does not proactively share codes or transfer items; it still changes the shared room and can compete for items."),
    "deceptive": ("LLM + false-code script", "One LLM plus a helper script that changes the code in outgoing code messages. This tests reactions to false information only where code handoffs exist; it does not add deception to item-only rooms."),
    "self_play": ("Two LLM agents", "Independent worker policies using the configured deployment. Fixed teams use two seats; orchestrated teams use a model-selected roster. This is evaluation, with no RL training or weight updates."),
    "reference": ("Reference scripts", "Workers follow a known solution. This checks the environment and logging pipeline, not LLM ability."),
}


SUMMARY_DIMENSIONS = (
    "family", "profile", "split", "condition", "mode", "observation_version",
    "communication", "worker_broadcast", "communication_instruction", "action_interface", "num_rooms",
    "team_mode", "num_subagents", "orchestrator_kind", "orchestration_protocol", "worker_progress_memory",
    "torch_affordance", "turn_budget", "max_steps",
    "min_subagents", "max_subagents",
    "supervision_mode", "check_in_interval",
)
CONDITION_DIMENSIONS = tuple(key for key in SUMMARY_DIMENSIONS if key != "profile")


def run_label(row):
    if row.get("team_mode") == "orchestrated":
        prefix = "Scripted orchestrator" if row.get("orchestrator_kind") == "scripted" else "Orchestrator"
        version = row.get("orchestration_protocol")
        suffix = f" [{version}]" if version in {"v2", "v3", "v4", "v5"} else ""
        if row.get("worker_broadcast") is False:
            suffix += " [broadcast off]"
        return f"{prefix} + {row['num_subagents']} subagents{suffix}"
    return CONDITIONS.get(row["condition"], (row["condition"],))[0]


def controller_roles(meta, agents):
    if meta["condition"] == "reference":
        return {a: "reference script" for a in agents}
    if meta["condition"] in {"solo", "self_play"}:
        return {a: "LLM" for a in agents}
    return {a: "LLM" if a == meta["evaluated_seat"] else "script" for a in agents}


def aggregate_rows(group):
    """Explicit denominators: duration means use finished tasks, usage uses all runs."""
    complete = [r for r in group if r["reason"] in {"success", "turn_budget", "race_winner"}]
    success = [r for r in complete if r["joint_escape"]]
    steps = [r["joint_steps"] for r in complete]
    actions = sum(r["llm_actions"] for r in group)
    opportunities = sum(r["llm_sharing_opportunities"] for r in group)
    errors = dict(Counter(r["reason"] for r in group if r not in complete))
    escaped = sum(r["llm_escaped_agents"] for r in complete)
    llm_agents = sum(r["llm_agents"] for r in complete)
    known_elapsed = [r["elapsed_seconds"] for r in group if r["elapsed_seconds"] is not None]
    return dict(
        episodes=len(group), completed=len(complete), successful_episodes=len(success),
        failures_or_truncations=len(group) - len(complete), error_or_truncation_counts=errors,
        mean_completed_joint_steps=mean(steps) if steps else None,
        median_completed_joint_steps=median(steps) if steps else None,
        std_completed_joint_steps=stdev(steps) if len(steps) > 1 else None,
        min_completed_joint_steps=min(steps) if steps else None,
        max_completed_joint_steps=max(steps) if steps else None,
        mean_successful_joint_steps=mean(r["joint_steps"] for r in success) if success else None,
        mean_completed_total_actions=mean(r["total_actions"] for r in complete) if complete else None,
        mean_completed_total_payoff=mean(r["total_payoff"] for r in complete) if complete else None,
        mean_llm_actions=sum(r["llm_actions"] for r in complete) / llm_agents if llm_agents else None,
        llm_escape_rate=escaped / llm_agents if llm_agents else None,
        mean_llm_escape_actions=(sum(r["llm_escape_actions_sum"] for r in complete) / escaped if escaped else None),
        mean_llm_escape_step=(sum(r["llm_escape_steps_sum"] for r in complete) / sum(r["llm_observed_escape_steps"] for r in complete)
            if sum(r["llm_observed_escape_steps"] for r in complete) else None),
        llm_actions=actions, llm_invalid_actions=sum(r["llm_invalid_actions"] for r in group),
        llm_wrong_attempts=sum(r["llm_wrong_attempts"] for r in group),
        llm_unsuccessful_action_rate=sum(r["llm_unsuccessful_actions"] for r in group) / actions if actions else None,
        llm_conflicts=sum(r["llm_conflicts"] for r in group),
        llm_messages_sent=sum(r["llm_messages_sent"] for r in group),
        llm_notes_written=sum(r.get("llm_notes_written", 0) for r in group),
        llm_notes_read=sum(r.get("llm_notes_read", 0) for r in group),
        llm_items_transferred=sum(r.get("llm_items_transferred", 0) for r in group),
        reciprocal_use_episodes=sum(r.get("reciprocal_use", False) for r in complete),
        llm_sharing_opportunities=opportunities,
        llm_useful_shares=sum(r["llm_useful_shares"] for r in group),
        llm_used_shared_resources=sum(r["llm_used_shared_resources"] for r in group),
        llm_sharing_rate=sum(r["llm_eligible_shares"] for r in group) / opportunities if opportunities else None,
        llm_mean_payoff=mean(r["llm_mean_payoff"] for r in complete if r["llm_mean_payoff"] is not None)
            if any(r["llm_mean_payoff"] is not None for r in complete) else None,
        total_api_calls=sum(r["api_calls"] for r in group),
        orchestrator_calls=sum(r["orchestrator_calls"] for r in group),
        worker_calls=sum(r["worker_calls"] for r in group),
        llm_reports_sent=sum(r["llm_reports_sent"] for r in group),
        total_tokens=sum(r["total_tokens"] for r in group),
        total_input_tokens=sum(r["input_tokens"] for r in group),
        total_output_tokens=sum(r["output_tokens"] for r in group),
        mean_tokens_per_episode=mean(r["total_tokens"] for r in group) if group else None,
        unknown_usage_calls=sum(r["unknown_usage_calls"] for r in group),
        estimated_cost_usd=sum(r["estimated_cost_usd"] for r in group)
            if group and all(r["estimated_cost_usd"] is not None for r in group) else None,
        mean_elapsed_seconds=mean(known_elapsed) if known_elapsed else None,
        mean_api_latency_seconds=(sum(r["model_latency_seconds"] for r in group) / sum(r["api_calls"] for r in group)
            if sum(r["api_calls"] for r in group) else None),
    )


def describe_report(records, rows, summary, condition_summary=None):
    rooms = {r["room_id"] for r in rows}
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    models = sorted({r["metadata"]["model"] for r in records if r["metadata"].get("model")})
    script_versions = sorted({v for r in records for v in r["metadata"].get("script_policy_versions", {}).values()})
    lines = ["# Evaluation summary", "",
             f"Recorded **{len(rows)} episodes across {len(rooms)} distinct room(s)**. "
             f"Model deployment(s): {', '.join(models) or 'none (scripted controls)'}. "
             f"Reward mode(s): {', '.join(sorted({r['mode'] for r in rows}))}.", "",
             "An episode is one attempt at a room. A step is one simultaneous tick: each active agent submits one action, including invalid actions. "
             "In a shared room with together-escape, one agent opening the exit releases everyone. Team escape alone does not show that both agents contributed.", "",
             "Torch affordances: " + ", ".join(sorted({r["torch_affordance"] for r in rows})) + ". "
             "Turn budgets / step cutoffs: " + ", ".join(
                 f"{turns or 'unrecorded'} / {steps or 'unrecorded'}"
                 for turns, steps in sorted({(r["turn_budget"], r["max_steps"]) for r in rows})) + ". "
             "These settings remain separate in machine-readable summaries.", "",
             "## What the conditions mean", "",
             "| Run label | Present in this batch? | Who acts and what it tests |", "|---|---|---|"]
    described = list(dict.fromkeys(["solo", "honest", "silent", "deceptive", "self_play"] + conditions))
    for condition in described:
        label, description = CONDITIONS.get(condition, (condition, "Custom condition."))
        lines.append(f"| **{label}** (`{condition}`) | {'Yes' if condition in conditions else 'No'} | {description} |")
    lines += ["", "Scripts are preprogrammed baselines, not additional LLMs. They know dependency plans, inspect clues before their code actions, "
              "and cannot use items held by the other agent. Script version 1 follows a mostly fixed sequence; version 2 rechecks prerequisites, skips completed tasks, "
              "and switches to other feasible work when blocked. Helpful scripts may request use of a missing shared item; silent scripts wait if no work is feasible. "
              "Script versions are recorded below. “Helpful” does not guarantee recovery or equal contribution.", "",
              "Recorded scripted-policy versions: " + (", ".join(map(str, script_versions)) if script_versions else "not recorded / no scripted controllers") + ".", "",
              "Room families: `parallel_branches` tests division of physical work; `private_handoffs` tests sharing private exit codes; "
              "`costly_help` tests spending actions to transfer a useful item. Help profiles: `balanced` gives similar dependence to both agents, "
              "`asymmetric` gives unequal solo-route costs, and `zero` provides no useful partner handoff. "
              "`development` rooms are used for calibration; `heldout` rooms are reserved for later generalization checks.", "",
              "Reward modes: `mixed` scores each agent on its own escape payoff; `collaborative` gives each the sum of payoffs; "
              "`competitive` gives own payoff minus the other agent's. Escape payoff is 10 plus unused turns, otherwise zero.", "",
              "## Results by setup", "",
              "Observation formats: `v1` is the original state summary; `v2` adds typed objects, exact item IDs, completed-lock labels and recent own actions; "
              "`v3` additionally distinguishes CODE LOCK/enter from ITEM LOCK/use and explains that numeric ID suffixes are not revealed codes. Room rules and action costs are unchanged.", "",
              "| Room family | Help profile | Dataset split | Setup | Reward mode | Observation | Interface | Rooms | Finished runs | Team/solo escape rate | Errors / truncations |",
              "|---|---|---|---|---|---|---|---|---:|---:|---:|"]
    for s in summary:
        rate = f"{s['completed_joint_escape_rate']:.0%}" if s['completed_joint_escape_rate'] is not None else "—"
        label = run_label(s)
        lines.append(f"| {s['family']} | {s['profile']} | {s['split']} | {label} | {s['mode']} | {s['observation_version']} | {s['action_interface']} | {s['num_rooms']} | {s['completed']} | {rate} | {s['failures_or_truncations']} |")
    lines += ["", "Finished runs include both successful escapes and failures that exhausted the turn budget. "
              "API errors, policy errors, budget stops and max-step truncations are reported separately and excluded from escape-rate denominators. "
              "Rates average over rooms, keeping seat assignments and repeated trials together.", ""]
    lines += ["## Average duration and agent performance", "",
              "Rows below pool help profiles within each family/condition. Finished-run averages include turn-budget failures; "
              "successful-only averages are shown separately to avoid hiding failures. LLM actions count that agent's turns, "
              "while joint steps run until the entire episode ends. These differ when one agent escapes before its partner.", "",
              "| Family | Setup | Reward mode | Observation | Interface | Rooms | Finished | Avg joint steps | Avg steps: successful only | Avg LLM actions | LLM escape rate |",
              "|---|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    number = lambda v: f"{v:.2f}" if v is not None else "—"
    percent = lambda v: f"{v:.1%}" if v is not None else "—"
    for s in condition_summary or []:
        label = run_label(s)
        lines.append(f"| {s['family']} | {label} | {s['mode']} | {s['observation_version']} | {s['action_interface']} | {s['num_rooms']} | {s['completed']} | {number(s['mean_completed_joint_steps'])} | {number(s['mean_successful_joint_steps'])} | {number(s['mean_llm_actions'])} | {percent(s['llm_escape_rate'])} |")
    lines += ["", "## Sharing and action quality", "",
              "| Family | Setup | Reward mode | Observation | Interface | Rooms | LLM unsuccessful actions | LLM messages | Eligible sharing opportunities | Novel relevant shares | Sharing rate | Received shared resources used |",
              "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in condition_summary or []:
        label = run_label(s)
        lines.append(f"| {s['family']} | {label} | {s['mode']} | {s['observation_version']} | {s['action_interface']} | {s['num_rooms']} | {percent(s['llm_unsuccessful_action_rate'])} | {s['llm_messages_sent']} | {s['llm_sharing_opportunities']} | {s['llm_useful_shares']} | {percent(s['llm_sharing_rate'])} | {s['llm_used_shared_resources']} |")
    lines += ["", "## Communication channels", "",
              "| Family | Setup | Broadcasts | Private reports | Notes written | Note reads | Item transfers |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for s in condition_summary or []:
        lines.append(f"| {s['family']} | {run_label(s)} | {s['llm_messages_sent']} | {s['llm_reports_sent']} | {s['llm_notes_written']} | {s['llm_notes_read']} | {s['llm_items_transferred']} |")
    lines += ["", "Channel counts record successful actions, not proof of communication intent. "
              "Shared notes require explicit read_notes actions; private reports reach only the orchestrator. "
              "Exact text, transfers and shared state changes remain in the episode log."]
    if len({r["mode"] for r in rows}) > 1:
        matched = defaultdict(set)
        for row in rows:
            if row["reason"] in {"success", "turn_budget", "race_winner"}:
                matched[(row["room_id"], row["condition"], row["evaluated_seat"], row["repeat"], row["observation_version"], row["action_interface"], row["team_mode"], row["num_subagents"], row["orchestrator_kind"], row["worker_broadcast"])].add(row["mode"])
        lines += ["", "## Incentive comparison", "",
                  "This experiment changes the numerical reward transform and matching prompt framing together. "
                  "It tests inference-time response to stated incentives; model weights are unchanged. "
                  "Room mechanics, partner policy and observation format stay fixed within each matched pair."]
        for first, second in combinations(sorted({r["mode"] for r in rows}), 2):
            pairs = sum({first, second}.issubset(modes) for modes in matched.values())
            lines += ["", f"Completed matched {first}/{second} pairs: **{pairs}** (same room, partner, LLM seat, trial index and observations)."]
    lines += ["", "A sharing opportunity requires the LLM to acquire a partner-needed resource before the target opens. "
              "No opportunities gives an undefined rate (—), not zero cooperation. Counts refer to chat/item handoffs in the manifest; "
              "notes are retained in the logs but are not scored as chat sharing. Using a previously shared resource is observed behaviour, "
              "not proof that the message caused the action.", "",
              "Aggregate means, medians, sample standard deviations, min/max steps, per-agent escape timing, payoffs, "
              "sharing, tokens, API calls and latency are saved in `summary.json`, `summary.csv` and `condition_summary.json`/`.csv`. "
              "The per-agent episode values remain in `episodes.csv`.", "",
              "## Individual episodes and contribution", "",
              "A/B are physical seats. The LLM seat column identifies which agent called the model; the other seat uses the named script. "
              "A solo trial from physical wing B is represented internally as agent A and is labelled with its original wing below.", "",
              "| Episode / log | Setup | Reward mode | Observation | Interface | Rooms | LLM seat | Outcome | Steps | LLM invalid / actions | LLM wrong attempts | Script invalid / actions | New locks/exits opened: LLM / script |",
              "|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for index, (record, row) in enumerate(zip(records, rows), 1):
        stats = record["metrics"]["per_agent"]
        roles = controller_roles(record["metadata"], stats)
        llm = [s for a, s in stats.items() if roles[a] == "LLM"]
        script = [s for a, s in stats.items() if roles[a] != "LLM"]
        label = run_label(row)
        seats = ", ".join(a for a in stats if roles[a] == "LLM") or "none"
        if row["condition"] == "solo":
            seats = f"alone (wing {row['evaluated_seat']})"
        outcome = {"success": "Escaped", "turn_budget": "Failed: turns exhausted", "race_winner": "One agent won",
                   "api_error": "API failure", "policy_error": "Policy failure", "budget_exceeded": "Budget stop", "max_steps": "Step-limit truncation"}.get(row["reason"], row["reason"])
        ratio = lambda items: f"{sum(s['invalid_actions'] for s in items)}/{sum(s['actions'] for s in items)}" if items else "—"
        opens = lambda items: str(sum(s["new_locks_opened"] for s in items)) if items and all("new_locks_opened" in s for s in items) else "—"
        link = f"[{index}]({Path(row['log']).resolve()})"
        wrong = sum(s["wrong_attempts"] for s in llm) if llm else "—"
        lines.append(f"| {link} | {label} | {row['mode']} | {row['observation_version']} | {row['action_interface']} | {row['num_rooms']} | {seats} | {outcome} | {row['joint_steps']} | {ratio(llm)} | {wrong} | {ratio(script)} | {opens(llm)} / {opens(script)} |")
    if rows:
        heading = "Observation comparison" if len({row["observation_version"] for row in rows}) > 1 else "Action reliability"
        lines += ["", f"## {heading}", "",
                  "| Format | Reward mode | Interface | Rooms | Team | Episodes | Escapes / finished | LLM invalid / actions | LLM wrong / actions | LLM unsuccessful / actions | Longest repeated failing action |",
                  "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
        reliability_fields = ("observation_version", "mode", "action_interface", "num_rooms",
                              "team_mode", "num_subagents", "orchestrator_kind", "orchestration_protocol", "worker_broadcast", "family", "worker_progress_memory")
        reliability_groups = defaultdict(list)
        for record, row in zip(records, rows):
            reliability_groups[tuple(row[field] for field in reliability_fields)].append((record, row))
        for key, selected in sorted(reliability_groups.items()):
            version, mode, interface, count = key[:4]
            label = run_label(selected[0][1])
            finished = [row for _, row in selected if row["reason"] in {"success", "turn_budget", "race_winner"}]
            stats = [s for r, _ in selected for a, s in r["metrics"]["per_agent"].items()
                     if controller_roles(r["metadata"], r["metrics"]["per_agent"])[a] == "LLM"]
            invalid = sum(s["invalid_actions"] for s in stats)
            wrong = sum(s["wrong_attempts"] for s in stats)
            actions = sum(s["actions"] for s in stats)
            unsuccessful = f"{invalid + wrong}/{actions} ({(invalid + wrong) / actions:.1%})" if actions else "—"
            streak = max((s.get("longest_failed_action_streak", 0) for s in stats), default=0)
            lines.append(f"| {version} | {mode} | {interface} | {count} | {label} | {len(selected)} | {sum(row['joint_escape'] for row in finished)}/{len(finished)} | {invalid}/{actions} | {wrong}/{actions} | {unsuccessful} | {streak} |")
        matched = defaultdict(set)
        for row in rows:
            if row["reason"] in {"success", "turn_budget", "race_winner"}:
                matched[(row["room_id"], row["condition"], row["evaluated_seat"], row["repeat"], row["mode"], row["action_interface"], row["team_mode"], row["num_subagents"], row["orchestrator_kind"], row["worker_broadcast"])].add(row["observation_version"])
        present_versions = sorted({row["observation_version"] for row in rows})
        for first, second in combinations(present_versions, 2):
            pairs = sum({first, second}.issubset(versions) for versions in matched.values())
            lines += ["", f"Completed matched {first}/{second} pairs: **{pairs}** (same room, condition, seat, trial index and reward mode). "
                      "Check pair coverage before interpreting the format comparison; token use can change because observations differ. "
                      "A small calibration batch is descriptive evidence, not a generalization result."]
    lines += ["", "Invalid actions are rejected commands (for example, taking a clue or using an already-open lock). "
              "Wrong attempts are parsed actions that return `wrong`, including wrong codes/items and using enter on an item lock. "
              "Unsuccessful actions = invalid actions + wrong attempts; both consume turns. These counts do not include every inefficient action (for example, unnecessary waits). "
              "New locks/exits count successful first openings by each controller; "
              "they describe concrete progress, not a complete measure of causal contribution or cooperation.", "",
              "## What to inspect next", ""]
    streaks = [(i + 1, max((s.get("longest_failed_action_streak", 0) for a, s in r["metrics"]["per_agent"].items()
                           if controller_roles(r["metadata"], r["metrics"]["per_agent"])[a] == "LLM"), default=0)) for i, r in enumerate(records)]
    for index, streak in streaks:
        if streak >= 3:
            lines.append(f"- Episode {index}: the LLM repeated the same failing action for **{streak} consecutive turns**. Inspect its observation and last-result feedback.")
    llm_stats = [s for r in records for a, s in r["metrics"]["per_agent"].items()
                 if controller_roles(r["metadata"], r["metrics"]["per_agent"])[a] == "LLM"]
    if any(s["wrong_attempts"] for s in llm_stats):
        lines.append("- Inspect wrong attempts before interpreting social behavior: check whether the agent used enter on an item lock, guessed from an ID suffix, or supplied a genuinely wrong code/item. Compare explicit lock-method observations in a matched solo run.")
    if any(s["invalid_actions"] for s in llm_stats):
        lines.append("- Inspect remaining rejected actions and state feedback in the selected observation format; repeat solo calibration with the same strict turn costs before expanding partner comparisons.")
    elif not any(s["wrong_attempts"] for s in llm_stats):
        lines.append("- Compare performance across additional room seeds and task families before drawing conclusions about partner effects.")
    if len(rooms) < 2:
        lines.append("- This report covers only **one room instance**. Its outcomes do not establish success rates across rooms, robust seat effects, or learned cooperation.")
    missing = {"solo", "honest", "silent", "deceptive", "self_play"} - set(conditions)
    if missing:
        lines.append("- Not observed in this batch: " + ", ".join(CONDITIONS[c][0] for c in sorted(missing)) + ".")
    lines += ["", "## Usage and uncertainty", "",
              f"Provider requests: {sum(r['api_calls'] for r in rows)}. Recorded tokens: {sum(r['total_tokens'] for r in rows):,}. "
              f"Requests with unknown token usage: {sum(r['unknown_usage_calls'] for r in rows)}.", ""]
    if rows and all(r["estimated_cost_usd"] is not None for r in rows):
        lines.append(f"Estimated cost from configured token prices: ${sum(r['estimated_cost_usd'] for r in rows):.6f}.")
    else:
        lines.append("Dollar cost is unavailable for some/all episodes; token prices or provider usage were not supplied. Script-only controls make no provider requests.")
    lines += ["", "Descriptive 95% intervals in `summary.json` resample rooms, retaining seats and repeats together. "
              "Fewer than two rooms gives no interval. The full per-agent metrics and outcomes are in `episodes.csv`; "
              "linked JSONL logs retain the exact observations, model responses and transitions."]
    return "\n".join(lines) + "\n"


def load_suite(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    cases = []
    for meta in manifest["cases"]:
        room = load_room(path.parent / meta["room_file"])
        if digest(room.to_dict()) != meta["room_sha256"]:
            raise ValueError(f"Suite hash mismatch: {meta['room_file']}")
        cases.append((room, meta))
    return cases


def solo_projection(room, seat):
    data = room.to_dict()
    data["id"] += f"_solo_{seat}"
    data["wings"] = {"A": room.wings[seat]}
    if room.locations:
        data["start_locations"] = {"A": room.start_locations[seat]}
    data["nodes"] = [n for n in data["nodes"] if room.is_shared or n["wing"] == seat]
    locations = {i for n in data["nodes"] for i in n.get("items", []) + n.get("contents", [])}
    data["items"] = {i: item for i, item in data["items"].items() if i in locations}
    for n in data["nodes"]:
        if not room.is_shared:
            n["wing"] = "A"
        if "item" in n.get("requires", {}):
            n["requires"]["item"] = [i for i in n["requires"]["item"] if i in data["items"]]
    return Room.from_dict(data)


def summarize(records, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, grouped = [], defaultdict(list)
    for record in records:
        meta, metrics = record["metadata"], record["metrics"]
        row = dict(family=meta["family"], profile=meta["profile"], split=meta["split"],
                   room_id=meta["source_room_id"], seed=meta["seed"], condition=meta["condition"],
                   repeat=meta["repeat"], evaluated_seat=meta["evaluated_seat"], mode=meta["mode"],
                   reason=metrics["termination_reason"], joint_escape=metrics["joint_escape"],
                   escape_rate=metrics["escape_rate"], joint_steps=metrics["joint_steps"],
                   total_actions=metrics["total_actions"], api_calls=metrics["api_calls"],
                   total_tokens=metrics["total_tokens"], unknown_usage_calls=metrics["unknown_usage_calls"],
                   estimated_cost_usd=metrics["estimated_cost_usd"], log=record["path"])
        row["team_mode"] = meta.get("team_mode", "fixed")
        row["orchestration_protocol"] = meta.get("orchestration_protocol", "v1" if row["team_mode"] == "orchestrated" else "none")
        row["worker_progress_memory"] = meta.get("worker_progress_memory", False)
        row["num_subagents"] = meta.get("num_subagents", len(metrics["per_agent"]))
        row["min_subagents"] = meta.get("min_subagents", 1) if row["team_mode"] == "orchestrated" else 0
        row["max_subagents"] = meta.get("max_subagents", 8) if row["team_mode"] == "orchestrated" else 0
        row["supervision_mode"] = meta.get("supervision_mode", "per_tick") if row["team_mode"] == "orchestrated" else "none"
        row["check_in_interval"] = meta.get("check_in_interval", 0)
        row["orchestrator_kind"] = meta.get("orchestrator_kind", "none")
        row["orchestrator_calls"] = metrics.get("orchestrator_calls", 0)
        row["worker_calls"] = metrics.get("worker_calls", metrics["api_calls"])
        row["num_rooms"] = meta.get("num_rooms", 1)
        row["torch_affordance"] = meta.get("torch_affordance", "legacy")
        row["turn_budget"] = meta.get("turn_budget", meta.get("recommended_turn_budget", 0))
        row["max_steps"] = meta.get("max_steps", row["turn_budget"])
        row["action_interface"] = meta.get("action_interface", "text")
        row["observation_version"] = meta.get("observation_version", "unspecified")
        row["communication"] = meta.get("communication", "enabled")
        row["worker_broadcast"] = meta.get("worker_broadcast", True)
        row["communication_instruction"] = meta.get("communication_instruction", "original")
        row["framing"] = meta.get("framing", "unspecified")
        row["total_payoff"] = sum(s.get("payoff", 0) for s in metrics["per_agent"].values())
        row["elapsed_seconds"] = metrics.get("elapsed_seconds")
        for metric in ("input_tokens", "output_tokens", "model_latency_seconds"):
            row[metric] = metrics.get(metric, 0)
        roles = controller_roles(meta, metrics["per_agent"])
        llm = [s for a, s in metrics["per_agent"].items() if roles[a] == "LLM"]
        row["llm_agents"] = len(llm)
        row["llm_escaped_agents"] = sum(s["escaped"] for s in llm)
        row["llm_escape_actions_sum"] = sum(s["actions"] for s in llm if s["escaped"])
        row["llm_observed_escape_steps"] = sum(s.get("escape_step") is not None for s in llm)
        row["llm_escape_steps_sum"] = sum(s["escape_step"] for s in llm if s.get("escape_step") is not None)
        row["reciprocal_use"] = len(llm) >= 2 and all(s.get("used_shared_resources", 0) > 0 for s in llm)
        for metric in ("actions", "invalid_actions", "wrong_attempts", "conflicts", "messages_sent", "reports_sent", "notes_written",
                       "notes_read", "items_transferred", "sharing_opportunities", "useful_shares", "used_shared_resources"):
            row[f"llm_{metric}"] = sum(s.get(metric, 0) for s in llm)
        row["llm_unsuccessful_actions"] = row["llm_invalid_actions"] + row["llm_wrong_attempts"]
        row["llm_eligible_shares"] = sum((s.get("sharing_rate") or 0) * s.get("sharing_opportunities", 0) for s in llm)
        row["llm_mean_payoff"] = mean(s["payoff"] for s in llm) if llm else None
        for agent, stats in metrics["per_agent"].items():
            for key, value in stats.items():
                row[f"{agent}_{key}"] = value
        rows.append(row)
        grouped[tuple(row[key] for key in SUMMARY_DIMENSIONS)].append(row)
    summary = []
    for key, group in sorted(grouped.items()):
        complete = [r for r in group if r["reason"] in {"success", "turn_budget", "race_winner"}]
        clusters = defaultdict(list)
        for row in complete:
            clusters[row["room_id"]].append(float(row["joint_escape"]))
        values = [mean(v) for v in clusters.values()]
        interval = None
        if len(values) >= 2:
            rng = random.Random(0)
            samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(1000))
            interval = [samples[24], samples[974]]
        summary.append(dict(
            zip(SUMMARY_DIMENSIONS, key),
            **aggregate_rows(group), rooms=len(clusters),
            completed_joint_escape_rate=mean(values) if values else None,
            room_bootstrap_95_interval=interval,
        ))
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[tuple(row[key] for key in CONDITION_DIMENSIONS)].append(row)
    condition_summary = [dict(zip(CONDITION_DIMENSIONS, key),
                              **aggregate_rows(group)) for key, group in sorted(by_condition.items())]
    fields = sorted({k for row in rows for k in row})
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "condition_summary.json").write_text(json.dumps(condition_summary, indent=2) + "\n")
    for filename, data in (("summary.csv", summary), ("condition_summary.csv", condition_summary)):
        columns = sorted({k for row in data for k in row})
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, columns)
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, dict) else v for k, v in row.items()} for row in data)
    (output / "summary.md").write_text(describe_report(records, rows, summary, condition_summary))
    if any(r["metadata"].get("experiment") == "phase2_collaboration" for r in records):
        from motiverooms.collaboration_report import write_collaboration_report
        write_collaboration_report(records, rows, summary, output)
    return summary
