"""Local browser UI and JSON API for playing MotiveRooms."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Timer
from typing import Any
from urllib.parse import urlparse
import webbrowser

from motiverooms.catalog import ROOT, ROOMS, catalog_payload, preset_by_key
from motiverooms.session import AppSession

DEFAULT_ROOM = ROOT / "office_storage.yaml"
DEFAULT_SOLO_ROOM = ROOT / "office_solo.yaml"
STATIC_DIRECTORY = Path(__file__).with_name("static")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def make_handler(session: AppSession) -> type[BaseHTTPRequestHandler]:
    class GameRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/catalog":
                self._send_json(catalog_payload())
                return
            if path == "/api/state":
                self._send_json(session.state_payload())
                return
            if path in STATIC_FILES:
                filename, content_type = STATIC_FILES[path]
                self._send_file(STATIC_DIRECTORY / filename, content_type)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._read_json()
                if path == "/api/action":
                    action = body.get("action")
                    if not isinstance(action, str) or not action.strip():
                        raise ValueError("Action must be a non-empty string")
                    outcome = session.submit_human(action)
                    self._send_json(
                        {
                            "outcome": {
                                "valid": outcome.valid,
                                "result": outcome.result,
                            },
                            "state": session.state_payload(),
                        }
                    )
                    return
                if path == "/api/peek-auto-step":
                    self._send_json({"turn": session.peek_scripted_turn()})
                    return
                if path == "/api/auto-step":
                    outcome = session.submit_auto()
                    last_turn = (
                        asdict(session.episode_log[-1])
                        if session.episode_log
                        else None
                    )
                    self._send_json(
                        {
                            "outcome": {
                                "valid": outcome.valid,
                                "result": outcome.result,
                            },
                            "turn": last_turn,
                            "state": session.state_payload(),
                        }
                    )
                    return
                if path == "/api/run-episode":
                    steps = session.run_scripted_episode(
                        max_steps=int(body.get("max_steps", 100))
                    )
                    self._send_json(
                        {
                            "steps_run": steps,
                            "state": session.state_payload(),
                        }
                    )
                    return
                if path == "/api/reset":
                    session.reset_episode()
                    self._send_json({"state": session.state_payload()})
                    return
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
                    self._send_json({"state": session.state_payload()})
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except (json.JSONDecodeError, ValueError) as error:
                self._send_json(
                    {"error": str(error)},
                    status=HTTPStatus.BAD_REQUEST,
                )

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 10_000:
                raise ValueError("Request body is too large")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("Request body must be a JSON object")
            return data

        def _send_json(
            self,
            payload: dict[str, Any],
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def _send_file(self, path: Path, content_type: str) -> None:
            try:
                content = path.read_bytes()
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    return GameRequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play MotiveRooms in a local browser."
    )
    parser.add_argument("room", nargs="?", type=Path, default=DEFAULT_ROOM)
    parser.add_argument(
        "--solo",
        action="store_true",
        help="Load the single-agent office example (same as passing that room path).",
    )
    parser.add_argument("--turn-budget", type=int, default=12)
    default_port = int(os.environ.get("PORT", "8765"))
    default_host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.solo:
        room_entry = next(room for room in ROOMS if room.key == "office_solo")
    else:
        room_entry = next(
            (room for room in ROOMS if room.path.resolve() == args.room.resolve()),
            ROOMS[0],
        )
    session = AppSession(room_entry=room_entry, turn_budget=args.turn_budget)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    url = f"http://{args.host}:{args.port}"
    print(f"Escape Rooms UI running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        Timer(0.25, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
