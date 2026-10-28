"""Read-only projection of saved JSONL episodes for the replay workspace.

The browser receives recorded state and communication, never provider inputs,
reasoning items, credentials, or evaluator reference plans. Viewing cannot run
the environment or invoke a model.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from motiverooms.schema import Room


RUNS_DIRECTORY = Path(__file__).resolve().parents[1] / "runs"
BUNDLED_RECORDING = Path(__file__).resolve().parent / "data" / "successful_v5.json"
BUNDLED_RECORDING_ID = "bundled/successful_v5.json"


def _summary(start: dict, end: dict, metrics: dict, id: str) -> dict:
    room, meta = start["room"], start.get("metadata", {})
    agents = list(start["initial_state"]["agents"])
    return dict(id=id, room_id=room["id"], rooms=len(room.get("locations", {})) or 1,
                workers=len(agents), steps=end["state"]["env_step"], outcome=end["reason"],
                recorded_at=start.get("timestamp"), model=meta.get("model"),
                protocol=meta.get("orchestration_protocol", "fixed"),
                manager_calls=metrics.get("orchestrator_calls", 0),
                api_calls=metrics.get("api_calls", 0), total_tokens=metrics.get("total_tokens", 0))


def recording_catalog(directory: Path = RUNS_DIRECTORY) -> dict:
    """List recent completed model teams, reading headers and tails only."""
    paths = list(directory.glob("batch_*/*.jsonl")) + list(directory.glob("episode_*.jsonl"))
    recordings = []
    for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:250]:
        try:
            if not path.resolve().is_relative_to(directory.resolve()):
                continue
            with path.open("rb") as stream:
                start = json.loads(stream.readline())
                meta = start.get("metadata", {})
                if meta.get("team_mode") != "orchestrated" or meta.get("condition") == "reference":
                    continue
                stream.seek(max(0, path.stat().st_size - 256_000))
                tail = stream.read().splitlines()
            final = {}
            for line in tail:
                try:
                    event = json.loads(line)
                    if event.get("type") in {"end", "metrics"}:
                        final[event["type"]] = event
                except (ValueError, UnicodeDecodeError):
                    continue  # The first tail line may start midway through JSON.
            if "end" not in final:
                continue
            recordings.append(_summary(start, final["end"], final.get("metrics", {}),
                                       path.relative_to(directory).as_posix()))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if len(recordings) >= 30:
            break
    if not recordings:
        bundled = json.loads(BUNDLED_RECORDING.read_text(encoding="utf-8"))
        recordings.append(bundled["summary"])
    preferred = next((r["id"] for r in recordings if r["outcome"] == "success" and r["protocol"] == "v5"), None)
    return dict(recordings=recordings, default_id=preferred or (recordings[0]["id"] if recordings else None))


def load_recording(id: str, directory: Path = RUNS_DIRECTORY) -> dict:
    if id == BUNDLED_RECORDING_ID:
        return json.loads(BUNDLED_RECORDING.read_text(encoding="utf-8"))
    path = (directory / id).resolve()
    if (not path.is_relative_to(directory.resolve()) or path.suffix != ".jsonl" or not path.is_file()):
        raise ValueError("Recording not found")
    with path.open() as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    return project_recording(events, id=id)


def project_recording(events: list[dict], *, id: str = "imported") -> dict:
    """Normalize an episode, keeping explicit before/after-tick communication."""
    if not isinstance(events, list) or not events or any(not isinstance(e, dict) for e in events):
        raise ValueError("Expected a JSONL episode containing event objects")
    start = events[0]
    if start.get("type") != "start":
        raise ValueError("The recording must begin with a start event")
    end = next((e for e in reversed(events) if e.get("type") == "end"), None)
    if end is None:
        raise ValueError("This recording has not finished yet; load a completed episode")
    try:
        room = Room.from_dict(start["room"])
        initial = deepcopy(start["initial_state"])
        agents = list(initial["agents"])
        metrics = next((e for e in events if e.get("type") == "metrics"), {})
        summary = _summary(start, end, metrics, id)
        steps = [e for e in events if e.get("type") == "joint_step"]
        if [e["step"] for e in steps] != list(range(1, len(steps) + 1)):
            raise ValueError("Recording steps must be consecutive")
        frames = [dict(step=0, state=initial, actions=[], manager=False)]
        messages = []
        tasks = dict(start.get("metadata", {}).get("team_plan", {}).get("assignments", {}))
        controls: dict[int, list[dict]] = {}
        for event in events:
            if event.get("type") == "orchestrator_decision":
                controls.setdefault(event["step"], []).append(event)
        pending_reports: list[dict] = []
        last_briefings = {}

        def add(step, kind, sender, recipient, text="", **extra):
            row = dict(id=f"message-{len(messages)}", step=step, kind=kind, sender=sender,
                       recipient=recipient, text=text, **extra)
            messages.append(row)
            return row

        def controller_messages(step):
            for event in controls.get(step, []):
                result, view = event.get("result") or {}, event.get("input") or {}
                rationale = ""
                for item in event.get("output") or []:
                    if item.get("type") == "function_call":
                        try:
                            rationale = json.loads(item.get("arguments", "{}" )).get("rationale", "")
                        except (ValueError, TypeError):
                            pass
                if event.get("phase") == "spawn":
                    assignments = result.get("assignments", tasks)
                    add(step, "spawn", "Manager", "Team", rationale,
                        assignments=assignments, workers=result.get("agents", agents))
                    continue
                received = view.get("new_reports", [])
                for report in received:
                    for row in pending_reports:
                        if (row.get("received_step") is None and row["sender"] == report.get("agent_id")
                                and row["text"] == report.get("message")):
                            row["received_step"] = step
                            break
                updates = result.get("updated_assignments", {})
                tasks.update(updates)
                add(step, "check_in", "Manager", "Team", rationale, assignments=updates,
                    received=len(received), trigger=view.get("check_in", {}).get("reasons", ["scheduled"]))
                for briefing in result.get("exit_briefings", []):
                    key = briefing["node_id"]
                    facts = (briefing.get("room_id"), briefing.get("code"), briefing.get("status"))
                    if last_briefings.get(key) != facts:
                        add(step, "exit", "Manager", "All workers", briefing=briefing)
                        last_briefings[key] = facts
                if result.get("memory_update_error"):
                    add(step, "warning", "Manager", "Task board", result["memory_update_error"])
                if event.get("error"):
                    add(step, "warning", "Manager", "Team", event.get("error_detail") or event["error"])

        controller_messages(0)
        frames[0]["assignments"] = dict(tasks)
        for event in steps:
            step = event["step"]
            controller_messages(step)
            state, before = deepcopy(event["after"]), event["before"]
            actions = []
            for agent, outcome in event["outcomes"].items():
                action = outcome["action"]
                actions.append(dict(agent=agent, raw=event["actions"][agent], verb=action["verb"], args=action["args"],
                                    valid=outcome["valid"], result=outcome["result"],
                                    from_room=before["agents"][agent].get("location"),
                                    to_room=state["agents"][agent].get("location")))
            for report in state.get("reports", [])[len(before.get("reports", [])):]:
                row = add(step, "report", report["sender"], "Manager", report["text"], received_step=None)
                pending_reports.append(row)
            for chat in state.get("messages", [])[len(before.get("messages", [])):]:
                add(step, "chat", chat["sender"], "Team", chat["text"])
            frames.append(dict(step=step, state=state, actions=actions,
                               assignments=dict(tasks), manager=bool(controls.get(step))))
        # Retain a controller failure just before an uncommitted tick, without
        # inventing an environment transition or showing it in an earlier frame.
        if controls.get(len(steps) + 1):
            controller_messages(len(steps) + 1)
        nodes = []
        for node in room.nodes:
            data = node.to_dict()
            data.pop("text", None)
            data.pop("code", None)
            if data.get("requires"):
                data["requires"] = dict(code=bool(node.requires.code), items=list(node.requires.items))
            nodes.append(data)
        locations = room.locations or {wing: name for wing, name in room.wings.items()}
        return dict(summary=summary, agents=agents, turn_budget=start["config"]["turn_budget"],
                    locations=locations, nodes=nodes, passages=[p.to_dict() for p in room.passages],
                    items={key: item.name for key, item in room.items.items()},
                    frames=frames, messages=messages, outcome=end["reason"],
                    elapsed_seconds=end.get("elapsed_seconds"),
                    error_type=end.get("error_type"))
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("The recording is missing required room, state or action data") from error
