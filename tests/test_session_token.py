from motiverooms.catalog import ROOMS
from motiverooms.http_api import handle_post
from motiverooms.session import AppSession
from motiverooms.session_token import decode_session, encode_session


def test_session_token_roundtrip() -> None:
    session = AppSession(room_entry=ROOMS[0], turn_budget=12)
    session.configure(
        room_key="office_storage",
        mode="collaborative",
        partner_a="honest",
        partner_b="honest",
    )
    token = encode_session(session)
    restored = decode_session(token)
    assert restored.room_entry.key == "office_storage"
    assert restored.controllers["A"] == "honest"


def test_http_api_carries_session_on_configure() -> None:
    payload = handle_post("/api/configure", {"preset": "ideal_cooperation"})
    assert "session" in payload
    assert payload["state"]["scripted_turn"] is True
    again = handle_post(
        "/api/configure",
        {"session": payload["session"], "preset": "ideal_cooperation"},
    )
    assert "session" in again
