"""Serialize browser sessions for stateless hosts (e.g. Vercel)."""

from __future__ import annotations

import base64
import pickle

from motiverooms.session import AppSession

_MAX_TOKEN_BYTES = 512_000


def encode_session(session: AppSession) -> str:
    raw = pickle.dumps(session, protocol=pickle.HIGHEST_PROTOCOL)
    if len(raw) > _MAX_TOKEN_BYTES:
        raise ValueError("Session state is too large for serverless transport")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_session(token: str) -> AppSession:
    if not token or not isinstance(token, str):
        raise ValueError("Missing session token")
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        session = pickle.loads(raw)
    except (ValueError, pickle.UnpicklingError) as error:
        raise ValueError("Invalid session token") from error
    if not isinstance(session, AppSession):
        raise ValueError("Invalid session payload")
    return session
