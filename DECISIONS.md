# Implementation decisions

This file records choices where the design specification leaves room for more
than one reasonable implementation.

## Engine and action validity

- `RoomEngine` applies actions but does not choose turn order. The episode
  environment added in M3 will own A/B alternation and skipping escaped agents.
- A syntactically valid attempt with the wrong code or wrong item has
  `valid: true` and result `wrong`. `valid: false` is reserved for malformed or
  disallowed actions. Both outcomes consume one turn.
- Keys and other requirement items are reusable after `use`. The current room
  format gives each item a unique location, so this does not create duplicate
  handoffs.

## Solver

- Cooperative path length gives the agent every requirement sourced from the
  partner's wing at turn zero, at no action cost. Communication and handoff
  costs belong to the helping agent and are intentionally excluded.
- The solver returns both the path length and an executable shortest path. This
  makes generator failures and future calibration easier to inspect.
