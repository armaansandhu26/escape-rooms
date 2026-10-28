"""Apply order and conflict rules for simultaneous environment steps."""

from __future__ import annotations

VERB_PHASES: tuple[tuple[str, ...], ...] = (
    ("inspect", "say", "report", "write_note", "read_notes", "wait"),
    ("take",),
    ("enter", "use"),
    ("give", "give_to"),
    ("move",),
)

TAKE_CONFLICT_MESSAGE = "Someone else took it this step."
