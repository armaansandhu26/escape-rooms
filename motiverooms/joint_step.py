"""Apply order and conflict rules for simultaneous environment steps."""

from __future__ import annotations

VERB_PHASES: tuple[tuple[str, ...], ...] = (
    ("inspect", "say", "write_note", "read_notes", "wait"),
    ("take",),
    ("enter", "use"),
    ("give",),
)

TAKE_CONFLICT_MESSAGE = "Someone else took it this step."
