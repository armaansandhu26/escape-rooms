"""Stateless HTTP API used by Vercel and other serverless hosts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from motiverooms.catalog import ROOMS, catalog_payload, preset_by_key
from motiverooms.session import AppSession
from motiverooms.session_token import decode_session, encode_session


def default_session() -> AppSession:
    return AppSession(room_entry=ROOMS[0], turn_budget=12)


def _with_session(session: AppSession, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload["session"] = encode_session(session)
    return payload


def handle_get(path: str) -> dict[str, Any]:
    if path == "/api/catalog":
        return catalog_payload()
    if path == "/api/state":
        session = default_session()
        return _with_session(session, session.state_payload())
    raise ValueError(f"Unknown path: {path}")


def handle_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    token = body.get("session")
    session = decode_session(token) if token else default_session()

    if path == "/api/configure":
        preset_key = body.get("preset")
        if preset_key:
            preset = preset_by_key(str(preset_key))
            session.configure(
                room_key=preset.room_key,
                mode=preset.mode,
                partner_a=preset.partner_a,
                partner_b=preset.partner_b,
                turn_budget=body.get("turn_budget"),
            )
        else:
            session.configure(
                room_key=str(body.get("room_key", session.room_entry.key)),
                mode=str(body.get("mode", session.mode)),
                partner_a=str(body.get("partner_a", "human")),
                partner_b=str(body.get("partner_b", "human")),
                turn_budget=body.get("turn_budget"),
            )
        return _with_session(session, {"state": session.state_payload()})

    if path == "/api/reset":
        session.reset_episode()
        return _with_session(session, {"state": session.state_payload()})

    if path == "/api/peek-auto-step":
        return _with_session(session, {"turn": session.peek_scripted_turn()})

    if path == "/api/auto-step":
        outcomes = session.submit_auto()
        env_step = session.game.env_step
        step_records = [
            asdict(record)
            for record in session.episode_log
            if record.env_step == env_step
        ]
        last_turn = step_records[-1] if step_records else None
        return _with_session(
            session,
            {
                "env_step": env_step,
                "step": {"env_step": env_step, "agents": step_records},
                "outcomes": {
                    agent: {"valid": o.valid, "result": o.result}
                    for agent, o in outcomes.items()
                },
                "outcome": (
                    {"valid": last_turn["valid"], "result": last_turn["result"]}
                    if last_turn
                    else None
                ),
                "turn": last_turn,
                "state": session.state_payload(),
            },
        )

    if path == "/api/action":
        action = body.get("action")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Action must be a non-empty string")
        outcome = session.submit_human(action)
        return _with_session(
            session,
            {
                "outcome": {"valid": outcome.valid, "result": outcome.result},
                "state": session.state_payload(),
            },
        )

    if path == "/api/run-episode":
        steps = session.run_scripted_episode(max_steps=int(body.get("max_steps", 100)))
        return _with_session(
            session,
            {"steps_run": steps, "state": session.state_payload()},
        )

    raise ValueError(f"Unknown path: {path}")
