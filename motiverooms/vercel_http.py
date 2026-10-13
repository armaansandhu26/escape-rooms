"""Shared Vercel Python handler factory."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Callable

from motiverooms.http_api import handle_get, handle_post


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length > 100_000:
        raise ValueError("Request body is too large")
    raw = handler.rfile.read(length) if length else b"{}"
    data = json.loads(raw or b"{}")
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


def _send_json(handler: BaseHTTPRequestHandler, payload: dict, *, status: int = 200) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(encoded)


def make_get_handler(api_path: str) -> type[BaseHTTPRequestHandler]:
    class GetHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                _send_json(self, handle_get(api_path))
            except ValueError as error:
                _send_json(self, {"error": str(error)}, status=400)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return GetHandler


def make_post_handler(api_path: str) -> type[BaseHTTPRequestHandler]:
    class PostHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                body = _read_json(self)
                _send_json(self, handle_post(api_path, body))
            except ValueError as error:
                _send_json(self, {"error": str(error)}, status=400)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return PostHandler
