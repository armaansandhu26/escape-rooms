#!/usr/bin/env python3
"""Build the API prototype report directly from the completed historical logs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median, stdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motiverooms.experiment import compute_metrics, read_events

COHORTS = [
    ("Calibration A", "batch_508f226664134038a83b8ac748794454"),
    ("Calibration B", "batch_3a7aaf3c687a4266a7bed1fc4db3765b"),
    ("Shared room partners", "batch_9366695fc7a24103986a068bef092376"),
    ("Private code handoffs", "batch_1f065035d2cd4f0eb9f69aac4b3e3bf6"),
    ("Incentive comparison", "batch_c67fd1a4575b4d61a2113183dadd2526"),
]


def distribution(values, prefix):
    return {
        f"{prefix}_mean": mean(values),
        f"{prefix}_sd": stdev(values) if len(values) > 1 else None,
        f"{prefix}_median": median(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def collect():
    rows = []
    for cohort, batch in COHORTS:
        groups = defaultdict(list)
        paths = sorted((ROOT / "runs" / batch).glob("*.jsonl"))
        if not paths:
            raise ValueError(f"Missing source logs: {batch}")
        for path in paths:
            events = read_events(path)
            start = next(e for e in events if e["type"] == "start")
            if not any(e["type"] == "end" for e in events):
                raise ValueError(f"Incomplete source episode: {path}")
            meta, metrics = start["metadata"], compute_metrics(events)
            if metrics["termination_reason"] not in {"success", "turn_budget", "race_winner"}:
                raise ValueError(f"Review interrupted source episode: {path}")
            seat = "A" if meta["condition"] == "solo" else meta["evaluated_seat"]
            stats = metrics["per_agent"][seat]
            groups[(start["config"]["observation_version"], meta["condition"], meta["mode"])].append(
                (meta, metrics, stats)
            )
        for (version, condition, mode), records in groups.items():
            n = len(records)
            actions = [s["actions"] for _, _, s in records]
            steps = [m["joint_steps"] for _, m, _ in records]
            invalid = sum(s["invalid_actions"] for _, _, s in records)
            wrong = sum(s["wrong_attempts"] for _, _, s in records)
            successful_steps = [m["joint_steps"] for _, m, _ in records if m["joint_escape"]]
            rows.append(dict(
                cohort=cohort, batch_id=batch, room_family=records[0][0]["family"],
                observation_version=version, condition=condition, reward_mode=mode,
                episodes=n, escapes=len(successful_steps), escape_rate=len(successful_steps) / n,
                distinct_rooms=len({meta["source_room_id"] for meta, _, _ in records}),
                **distribution(actions, "llm_actions"), **distribution(steps, "episode_length"),
                mean_successful_episode_length=mean(successful_steps) if successful_steps else None,
                llm_invalid_actions=invalid, llm_wrong_attempts=wrong,
                llm_unsuccessful_actions=invalid + wrong, llm_total_actions=sum(actions),
                llm_unsuccessful_rate=(invalid + wrong) / sum(actions),
                llm_messages=sum(s["messages_sent"] for _, _, s in records),
                llm_notes=sum(s["notes_written"] for _, _, s in records),
                llm_received_resources_used=sum(s["used_shared_resources"] for _, _, s in records),
                api_calls=sum(m["api_calls"] for _, m, _ in records),
                total_tokens=sum(m["total_tokens"] for _, m, _ in records),
            ))
    return rows


def build_report(output):
    rows = collect()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    find = lambda cohort, version: next(r for r in rows if r["cohort"] == cohort and r["observation_version"] == version)
    a1, a2 = find("Calibration A", "v1"), find("Calibration A", "v2")
    b2, b3 = find("Calibration B", "v2"), find("Calibration B", "v3")
    current = [r for r in rows if r["observation_version"] == "v3"]
    refinements = [a1, a2, b2, b3]
    for name, data in (("statistics", current), ("refinement_statistics", refinements)):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        (output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    lines = [
        "# Improving the escape room environment", "",
        "MotiveRooms progress report | 19 October 2026", "",
        "We made the escape rooms easier for the AI to understand and measured how well it worked alone and with a partner. "
        "The model itself stayed the same throughout these tests.", "",
        "## 1 What we improved", "",
        "The early tests showed that the AI often misunderstood basic actions. It tried to pick up clue cards, "
        "unlock doors that were already open, or type the number in a key’s name instead of using the key.", "",
        "We made these changes:", "",
        "- **Made objects easier to identify.** The room now clearly labels clues, items, boxes and doors.",
        "- **Explained how each lock works.** Code locks need a number; item locks need a key or another object. Numbers in object names are not automatically lock codes.",
        "- **Made progress easier to follow.** Open locks are clearly marked, carried items have exact names, and the AI can see its recent actions and their results.",
        "- **Kept the comparison fair.** We reused the same rooms, model and turn limits. Mistakes still cost a turn; we did not automatically correct the AI’s choices.",
        "- **Improved the test partners.** Preprogrammed partners can change tasks when an item is unavailable instead of blindly continuing a broken plan.",
        "- **Recorded what happened.** We save every action and result, including who did the work, how long it took and whether information was shared.", "",
        "### Improvement in completing the room alone", "",
        "| Setup | Successful escapes | Success rate |",
        "|---|---:|---:|",
        f"| Original room information | {a1['escapes']}/{a1['episodes']} | {a1['escape_rate']:.0%} |",
        f"| Clearer objects and progress information | {a2['escapes']}/{a2['episodes']} | {a2['escape_rate']:.0%} |",
        f"| Clearer information plus lock instructions | {b3['escapes']}/{b3['episodes']} | {b3['escape_rate']:.0%} |", "",
        "Each row covers five attempts on the same five room layouts. The middle setup was tested twice and achieved "
        "three escapes both times. These were small checks across two rounds of testing, so the results are encouraging rather than a guarantee of success on every room.", "",
        "## 2 Results using the current setup (v3)", "",
        "We wanted to find out whether the AI could use help from a teammate, and whether that help reduced the work needed to escape. "
        "These checks help us understand the rooms and the AI’s starting behavior before we train agents to work together.", "",
        "**The AI is the language model we are testing. The teammate is a separate character controlled by rules we wrote.** "
        "For example, we can program the teammate to share a useful code. This gives us a predictable way to test how the AI responds to help. "
        "We have not yet tested two AI models learning to cooperate.", "",
        "We compared three setups:", "",
        "- **AI alone:** the AI solves the room without a teammate.",
        "- **AI + helpful teammate:** the programmed teammate shares useful information and helps solve tasks.",
        "- **AI + non-sharing teammate:** the programmed teammate works on its own tasks without offering information or items. "
        "Its actions can still help when both characters use the same room.", "",
        "All **35 attempts** shown below ended in a successful escape. To see how much work the AI actually did, we also measured its actions and the length of each attempt.", "",
        "Two measures help explain the results:", "",
        "- **AI actions:** how many moves the AI made, including mistakes, waiting and messages.",
        "- **Episode length:** how many steps the whole attempt lasted. Each active agent can make one move per step. "
        "The AI may escape before its teammate, so it can make fewer moves than the episode lasts.", "",
        "### Sharing one room", "",
        "The AI and its teammate can use the same objects and leave through the same exit. "
        "This checks whether having a teammate helps finish the room faster, even without exchanging information.", "",
        "| Setup | Attempts | Avg. AI actions | Avg. episode length | Episode length range |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {"solo": "AI alone", "honest": "AI + helpful teammate", "silent": "AI + non-sharing teammate"}

    def result_line(row, label):
        return (f"| {label} | {row['episodes']} | {row['llm_actions_mean']:.2f} | "
                f"{row['episode_length_mean']:.2f} | {row['episode_length_min']}–{row['episode_length_max']} |")

    lines.append(result_line(b3, "AI alone"))
    for row in current:
        if row["cohort"] == "Shared room partners":
            lines.append(result_line(row, labels[row["condition"]]))
    lines += [
        "", "The AI finished in fewer steps with a helpful teammate than with a non-sharing teammate. "
        "However, the teammate sometimes did most of the work, so escaping together does not by itself show that the AI cooperated. "
        "The AI-alone row covers five room layouts; each teammate row covers three layouts, tested with the AI in both character positions.", "",
        "### Solving separate rooms", "",
        "The AI and its teammate each have their own exit. The teammate can share a code that gives the AI a shortcut, "
        "but cannot finish the AI’s room for it. This checks whether the AI can use information from someone else.", "",
        "| Setup | Attempts | Avg. AI actions | Avg. episode length | Episode length range |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in current:
        if row["cohort"] == "Private code handoffs":
            lines.append(result_line(row, labels[row["condition"]]))
    lines += [
        "", "**Help let the AI leave sooner, but did not make the whole attempt finish sooner.** "
        "With a helpful teammate, the AI needed 2.83 moves on average instead of 4.67. The teammate then continued solving its own room, "
        "which is why the full episode averaged 6.00 steps.", "",
        "The AI used the codes it received but sent no messages back in these tests.", "",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return output / "report.md", rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/api_prototype_phase")
    args = parser.parse_args()
    path, rows = build_report(args.output)
    assert sum(r["episodes"] for r in rows) == 74
    assert (a := next(r for r in rows if r["cohort"] == "Calibration B" and r["observation_version"] == "v3"))["escapes"] == 5
    assert a["llm_total_actions"] == 53
    print(f"Report: {path}\nGroups: {len(rows)}; episodes: {sum(r['episodes'] for r in rows)}")
