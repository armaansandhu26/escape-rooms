"""Vercel ASGI entry — static UI plus /api/* routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from motiverooms.http_api import handle_get, handle_post
from motiverooms.recordings import load_recording

PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"

app = FastAPI()


@app.get("/")
def replay_workspace() -> FileResponse:
    return FileResponse(PUBLIC_DIR / "replay.html")


@app.get("/demo")
def scripted_demo() -> FileResponse:
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/api/recordings")
def get_recordings() -> dict:
    return handle_get("/api/recordings")


@app.get("/api/recording")
def get_recording(id: str) -> dict:
    return load_recording(id)


@app.post("/api/recordings/preview")
async def preview_recording(request: Request) -> dict:
    raw = await request.body()
    if len(raw) > 20_000_000:
        raise ValueError("Recording is too large")
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("Expected a recording object")
    return handle_post("/api/recordings/preview", body)


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/api/catalog")
def get_catalog() -> dict:
    return handle_get("/api/catalog")


@app.get("/api/state")
def get_state() -> dict:
    return handle_get("/api/state")


@app.post("/api/configure")
async def post_configure(request: Request) -> dict:
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    return handle_post("/api/configure", body)


@app.post("/api/reset")
async def post_reset(request: Request) -> dict:
    body = await request.json()
    if not isinstance(body, dict):
        body = {}
    return handle_post("/api/reset", body)


@app.post("/api/peek-auto-step")
async def post_peek(request: Request) -> dict:
    body = await request.json()
    if not isinstance(body, dict):
        body = {}
    return handle_post("/api/peek-auto-step", body)


@app.post("/api/auto-step")
async def post_auto_step(request: Request) -> dict:
    body = await request.json()
    if not isinstance(body, dict):
        body = {}
    return handle_post("/api/auto-step", body)


@app.post("/api/action")
async def post_action(request: Request) -> dict:
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    return handle_post("/api/action", body)


@app.post("/api/run-episode")
async def post_run_episode(request: Request) -> dict:
    body = await request.json()
    if not isinstance(body, dict):
        body = {}
    return handle_post("/api/run-episode", body)


if PUBLIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="site")
