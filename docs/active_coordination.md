# Active coordination protocol v4

V4 retains per-tick coordination as a historical comparison. New defaults use
[v5 manager supervision](manager_supervision.md), which preserves worker
objectives between event-driven check-ins.

V4 addresses the observed one-worker scout plan, repeated searches for absent
objects, missed open passages, confused item ownership and attempts to conserve
turns by waiting. It is a separate experimental protocol, not evidence of a
performance improvement until evaluated with the live model.

## Team selection

New CLI defaults and v4 configurations require **at least two and at most four
workers**. `spawn_subagents.assignments` has matching `minItems` and `maxItems`,
and local validation rejects counts outside the range. No heuristic repairs an
invalid model choice. The controller chooses the actual count within the range.

The roster is fixed after spawning. The prompt and initial public summary both
state that it cannot create a scout now and add workers later. Split-wing rooms
retain their physical seat count and reject incompatible bounds. The single-agent
torch diagnostic is a fixed solo condition and does not spawn a controller team.

Use `--min-subagents` and `--max-subagents` in either runner or set both YAML
fields. Historical configs explicitly retain their original minimum of one for
reproduction. Their v1–v3 prompts and logs remain available.

## Helping through assignments

The controller's new instructions call for distinct initial objectives, exploration
of reported open destinations, concrete item dependency assignments and exact
relaying of reported code fragments. Movement takes a destination `room_id`,
not a passage ID. Mentioning a clue in one room does not locate that clue there.
Assignments must account for reported locations and item carriers. An already-held
item calls for use or delivery by its holder, not another worker taking it again.

Workers receive corresponding v4 guidance: check assignments against current
observations, report missing targets and adjacent destinations, inspect fixtures
and clues, and take only objects listed as portable items. They can complete
multi-action objectives between reports instead of requesting permission each tick.
The engine still executes only the actions independently selected by workers.

Both parties are told explicitly that **wait and report each consume a turn**.
There is no passive arrival of new clues. Waiting can coordinate a same-tick
prerequisite, but it cannot conserve turns. Reports should convey meaningful new
discoveries or blockers rather than repeat unchanged status.

These are model instructions, not a forced room-by-room solver. The model can
still choose poorly; logs retain its assignments, actions and outcomes.

## Report-backed exploration memory

V4 extends the v3 task board with:

| Section | Fields | Meaning |
|---|---|---|
| `rooms` | `room_id`, `status`, `evidence_ids` | `known`, `visited`, or `surveyed` |
| `passages` | `passage_id`, `from_room`, `to_room`, `status`, `evidence_ids` | `open`, `locked`, or `unknown` |

Known destinations may be unvisited. A surveyed room may still have unresolved
mechanisms. These model-written records persist alongside items, dependencies,
searches and worker locations when recent conversation history is trimmed.
Every entry requires a received report/chat evidence ID and remains
`verified: false`. Validation checks structure and provenance, not truth or
semantic entailment. Empty updates preserve prior entries. Invalid deltas reject
atomically and return feedback while valid assignments still apply.

Reports remain private to the orchestrator. V4 does not grant access to worker
observations, inventories, positions, tool results, hidden room contents, codes,
seeds or evaluator plans. Room-map entries are never populated from engine state.
The controller can relay information in assignments. Worker broadcast remains
disabled in the supplied experiments; existing shared notes/transfers retain
their normal rules.

## Run the controlled retry

```bash
# Preview only: original three-room seed-42 puzzle, explicit station, 60 turns.
.venv/bin/python scripts/evaluate.py --config config/eval_exploration_coordinated_v4.yaml

# Offline reference and replay; no model calls.
.venv/bin/python scripts/evaluate.py --config config/eval_exploration_coordinated_v4.yaml --offline

# Live model run. Calls Azure.
.venv/bin/python -u scripts/evaluate.py --config config/eval_exploration_coordinated_v4.yaml --execute
```

This changes controller/worker guidance, navigation memory and the allowed team
range together. Compare it as a new protocol, not an isolated prompt ablation.
The puzzle, model, station wording and 60-turn limit match the preceding station
retry. The shared caps are 320 requests and three million tokens. Four workers
acting for all 60 ticks require at most 301 calls including controller requests
and the initial spawn. Early errors or exhausted token reservations can stop it
before that cap. Cost remains unknown until deployment prices are supplied.

The console prints the team rationale, initial tasks and changed assignments
alongside worker actions. Reports retain v4, the actual roster and min/max bounds
as separate grouping dimensions; full controller inputs, outputs, board deltas
and rationale remain in JSONL.

## Then test varied puzzles

`config/eval_exploration_varied_v4.yaml` selects the v4 varied experiment. It uses
three varied seeds and scaled per-room turn budgets with the same 2–4 worker
bounds and privacy rules. Use `--limit-episodes 1` for a first live check.
The single-rollout CLI defaults to v5 with two to four workers, one recent
controller exchange, three worker exchanges and private worker progress memory.
Explicit older context versions retain the original single-run memory behavior.

```bash
.venv/bin/python scripts/evaluate.py --config config/eval_exploration_varied_v4.yaml --offline
.venv/bin/python -u scripts/evaluate.py --config config/eval_exploration_varied_v4.yaml --limit-episodes 1 --execute
.venv/bin/python scripts/inspect_progress.py runs/batch_REPLACE_WITH_PRINTED_ID
```
