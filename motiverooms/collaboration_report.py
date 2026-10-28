"""Plain-language reports and checked communication pairs for phase 2."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from motiverooms.experiment import digest, read_events

FINISHED = {"success", "turn_budget", "race_winner"}


def pair_identity(row):
    return tuple(row[k] for k in (
        "room_id", "condition", "evaluated_seat", "repeat", "mode", "observation_version", "communication_instruction", "action_interface"))


def signature(record):
    """Match the actual logged room, engine, source and model settings."""
    start = read_events(record["path"])[0]
    config = {k: v for k, v in start["config"].items() if k != "communication"}
    meta = start["metadata"]
    policy_fields = () if meta["condition"] == "reference" else ("policies", "script_policy_versions")
    return digest(dict(room=start["room_sha256"], config=config,
                       source=start["code"]["source_sha256"],
                       settings={k: meta.get(k) for k in (
                           "model", "temperature", "max_output_tokens", "framing") + policy_fields},
                       communication_instruction=meta.get("communication_instruction", "original"),
                       protocol={k: v for k, v in meta["protocol"].items() if k != "communication"}))


def write_collaboration_report(records, rows, summary, output):
    output = Path(output)
    paired = defaultdict(dict)
    for record, row in zip(records, rows):
        group = paired[pair_identity(row)]
        if row["communication"] in group:
            raise ValueError("Duplicate communication condition in a matched pair")
        group[row["communication"]] = (record, row)
    pairs = []
    for key, group in paired.items():
        item = dict(room_id=key[0], repeat=key[3], communication_instruction=key[6], status="missing_condition")
        if set(group) == {"enabled", "disabled"}:
            on_record, on = group["enabled"]
            off_record, off = group["disabled"]
            if signature(on_record) != signature(off_record):
                item["status"] = "settings_mismatch"
            elif on["reason"] not in FINISHED or off["reason"] not in FINISHED:
                item["status"] = "interrupted"
            else:
                item.update(status="complete", profile=on["profile"],
                            steps_saved_with_communication=off["joint_steps"] - on["joint_steps"],
                            actions_saved_with_communication=off["total_actions"] - on["total_actions"],
                            team_payoff_change=on["total_payoff"] - off["total_payoff"],
                            both_escape_on=on["joint_escape"], both_escape_off=off["joint_escape"],
                            on_log=on["log"], off_log=off["log"])
        pairs.append(item)
    complete = [p for p in pairs if p["status"] == "complete"]
    plan_path = output / "plan.json"
    planned = json.loads(plan_path.read_text()) if plan_path.exists() else []
    planned_pairs = len({(p["room_id"], p["condition"], p["seat"], p["repeat"], p["mode"], p["observation_version"],
                          p.get("communication_instruction", "original"), p.get("action_interface", "text"))
                         for p in planned}) if planned else len(pairs)
    comparison = dict(planned_pairs=planned_pairs, recorded_pairs=len(pairs),
                      completed_matched_pairs=len(complete), pairs=pairs)
    (output / "communication_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    offline = all(r["metadata"]["condition"] == "reference" for r in records)
    lines = ["# Phase 2: can two AI agents help each other?", "",
             "Instruction condition: **" + ", ".join(sorted({r["communication_instruction"] for r in rows})) + "**. "
             "The explained condition describes delivery, turn costs and the possible benefit of exchanging information. "
             "It also states that communication is optional and may not help every room.", "",
             ("**Offline systems check.** These results use programmed agents, not AI models. "
              "Communication on follows the known joint solution; communication off follows each agent’s solo route. "
              "This checks the environment, logging and comparison only."
              if offline else
              "Both characters are controlled by separate calls to the same AI model. Each sees its own room and results, "
              "plus any messages it receives. Neither sees the other’s private clues or current choice. "
              "The model is being evaluated; no training or weight updates occur."), "",
             "We compare the same rooms with communication on and off. With communication on, agents can send messages, "
             "use shared notes and transfer items. With it off, all four actions—say, write_note, read_notes and give—are blocked. "
             "Rejected attempts still cost a turn. Room layouts, scoring and model settings stay fixed.", "",
             "Balanced rooms let both agents benefit from help. Unequal rooms make help more valuable to one agent. "
             "No-benefit rooms check behavior when sharing provides no shortcut.", "",
             "AI actions are moves per AI agent, including mistakes and messages. Episode length counts steps until the whole "
             "attempt ends; agents can finish at different times. Both active agents choose once per step.", "",
             "| Room type | Interface | Communication | Finished / recorded | Both escaped | Avg. actions per agent | Avg. episode length | Length range |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    number = lambda v: f"{v:.2f}" if v is not None else "—"
    profiles = {"balanced": "Both benefit", "asymmetric": "Unequal benefit", "zero": "No benefit"}
    for s in summary:
        selected = [r for r in rows if all(r[k] == s[k] for k in (
            "family", "profile", "split", "condition", "mode", "observation_version", "communication", "communication_instruction", "action_interface", "num_rooms", "team_mode", "num_subagents", "orchestrator_kind"))]
        finished = [r for r in selected if r["reason"] in FINISHED]
        actions = mean(r["total_actions"] / 2 for r in finished) if finished else None
        length_range = f"{s['min_completed_joint_steps']}–{s['max_completed_joint_steps']}" if finished else "—"
        lines.append(f"| {profiles.get(s['profile'], s['profile'])} | {s['action_interface']} | {'On' if s['communication'] == 'enabled' else 'Off'} | "
                     f"{s['completed']}/{s['episodes']} | {s['successful_episodes']}/{s['completed']} | "
                     f"{number(actions)} | {number(s['mean_completed_joint_steps'])} | {length_range} |")
    lines += ["", "Finished attempts include escapes and failures that ran out of turns. API failures and other interruptions "
              "are excluded from those averages and remain visible in the recorded counts.", "",
              "## Did they exchange useful information?", "",
              "Sending a message alone does not establish cooperation. We count relevant code/item shares and whether the "
              "recipient later used that resource. Shared-note actions are counted separately; their contents are not included "
              "in these message-based sharing counts.", "",
              "| Room type | Interface | Communication | Messages | Note writes | Relevant shares | Received resources used | Attempts where both used help |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        if offline:
            selected_metrics = [r["metrics"] for r in records if r["metadata"]["profile"] == s["profile"]
                                and r["metadata"]["communication"] == s["communication"]]
            stats = [a for m in selected_metrics for a in m["per_agent"].values()]
            messages = sum(a["messages_sent"] for a in stats)
            notes = sum(a["notes_written"] for a in stats)
            shares = sum(a["useful_shares"] for a in stats)
            used = sum(a["used_shared_resources"] for a in stats)
            reciprocal = sum(all(a["used_shared_resources"] > 0 for a in m["per_agent"].values()) for m in selected_metrics)
        else:
            messages, notes, shares, used, reciprocal = (s[k] for k in (
                "llm_messages_sent", "llm_notes_written", "llm_useful_shares", "llm_used_shared_resources", "reciprocal_use_episodes"))
        lines.append(f"| {profiles.get(s['profile'], s['profile'])} | {s['action_interface']} | {'On' if s['communication'] == 'enabled' else 'Off'} | "
                     f"{messages} | {notes} | {shares} | {used} | {reciprocal} |")
    lines += ["", f"**Checked communication pairs: {len(complete)}/{planned_pairs} planned.** "
              "Each pair uses the same room, trial index, engine/source version and model settings. "
              "Interrupted, missing or mismatched pairs do not contribute to the paired differences below. "
              "Trial indices are fresh samples, not reproducible API random seeds.", ""]
    for profile in profiles:
        selected = [p for p in complete if p["profile"] == profile]
        if selected:
            saved = mean(p["steps_saved_with_communication"] for p in selected)
            lines.append(f"- {profiles[profile]}: communication saved **{saved:.2f} episode steps** on average across {len(selected)} pairs. "
                         "A negative value means it took longer.")
    lines += ["", ("These programmed controls show a possible benefit from cooperation. They do not estimate how AI agents behave."
                   if offline else
                   "These are small development tests. Useful sharing and resource use describe observed behavior; "
                   "the matched comparison estimates the effect of allowing communication as a whole, not the effect of an individual message."), "",
              f"Provider requests: {sum(r['api_calls'] for r in rows)}. Recorded tokens: {sum(r['total_tokens'] for r in rows):,}. "
              f"Requests with unknown usage: {sum(r['unknown_usage_calls'] for r in rows)}. "
              "Dollar cost is available in the data files only when token prices and usage were supplied.", "",
              "Per-agent actions, escape timing, mistakes, sharing and usage are saved in `episodes.csv`. "
              "Averages, medians, variation and ranges are in `summary.csv`/`.json`. "
              "Pair checks and differences are in `communication_comparison.json`. Exact observations and model replies are in the JSONL logs."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
