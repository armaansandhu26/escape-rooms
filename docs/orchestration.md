# Orchestrated escape-room teams

New research defaults use [v5 manager supervision](manager_supervision.md), with
two to four workers, persistent objectives and report/silence-triggered check-ins.
V4 active coordination remains a per-tick comparison. The sections below
also document earlier v1–v3 experiments; their explicit configs preserve the
original team bounds and protocol for reproduction.

New research rollouts start with one control-only orchestrator. It receives a
public environment-size summary and chooses an initial team by calling
`spawn_subagents(assignments, rationale)`. The number of assignments determines
how many independent worker policies the runner actually creates. The default
cap is eight workers; the model selects the count within that cap. There is no
hardcoded room-to-worker formula in the model-backed path.

The orchestrator remains active. Before every joint tick, it can call
`coordinate(assignments, rationale)` to update tasks for existing workers.
Workers then choose independently from the same pre-step state. After all choices
finish, the environment commits one joint tick and returns private tool results.
The initial roster is fixed for the episode; supervision does not spawn further
workers or remove them mid-run.

## Running

```bash
# Inspect size, limits and protocol without calling a model.
.venv/bin/python scripts/rollout_llm.py --num-rooms 5 --seed 42 --max-subagents 6 --dry-run

# Execute: the orchestrator chooses the team, then supervises it. Calls Azure.
.venv/bin/python scripts/rollout_llm.py --num-rooms 5 --seed 42 --max-subagents 6 --max-api-calls 500 --max-tokens 2000000

# Preview a batch; every executed episode gets a fresh orchestrator.
.venv/bin/python scripts/evaluate.py --config config/eval_connected_rooms.yaml

# Exercise spawning, supervision, engine transitions and replay without model calls.
.venv/bin/python scripts/evaluate.py --config config/eval_connected_rooms.yaml --num-rooms 5 --offline

# Reproduce a historical fixed two-seat protocol.
.venv/bin/python scripts/rollout_llm.py --team-mode fixed --room rooms/examples/shared_mat.yaml --dry-run
```

A dry run cannot predict the model's team choice and does not fabricate one.
`--orchestrator-model` selects a separate controller deployment; otherwise it uses
the configured default deployment. `--orchestrator-max-output-tokens` defaults to
1024. Workers use the existing model/framing settings; `--model-a`, `--model-b`
and seat swapping apply to the first two worker policies, with the default model
used for any additional workers.

The connected-room batch config uses:

```yaml
team_mode: orchestrated
max_subagents: 8
orchestrator_max_output_tokens: 1024
# orchestrator_model: optional-deployment-name
```

Connected evaluations default to orchestrated teams. Existing frozen-suite
configs retain fixed teams for reproducibility; `team_mode: orchestrated` or the
CLI override selects orchestration for self-play. Split-wing scenarios still
require their original physical seats, so their team count is constrained to
those seats. Shared maps support arbitrary positive team sizes within the cap.
Assisted choices remain available with `--team-mode fixed`.
The evaluation CLI defaults to `config/eval_exploration_varied_manager_v5.yaml`; select
`config/eval_connected_rooms.yaml` for locked chains or an older config explicitly
when reproducing a historical batch.

## Information and communication

The orchestrator's initial input contains only the room count, allowed team-size
bounds, turn budget, reward mode and shared-exit rule. It receives no codes,
object placements, seed, reference plan or hidden map contents.

During supervision, its inputs contain worker IDs, remaining turns, escape status,
current assignments and **new explicit reports**. It cannot automatically read
worker observations, tool results, inventories, locations or reasoning histories.
Each worker has its own private client/conversation. It receives its current
assignment in its system instructions and can verify guidance against its own
observation.

Workers can communicate in three ways:

- `report(message)` sends a private report to the orchestrator, capped at 40 words,
  and costs one worker turn. A report becomes available to the controller before
  the next tick. It does not automatically appear in another worker's observation.
- `say(message)` broadcasts to teammates under the existing turn/message limits.
- Teams larger than two use `give_to(item_id, agent_id)` for unambiguous item
  transfers. Both workers must be in the same room. Two-worker teams retain
  `give(item_id)`; a single supervised worker can still report to the orchestrator.

The orchestrator can relay reported discoveries through updated assignments.
Assignments cost provider calls/tokens but no physical worker turns; the
orchestrator has no inventory, position, room actions or escape payoff.
Orchestration requires communication enabled. Communication-off comparisons must
use `team_mode: fixed`; a controller must not silently create a communication
channel in those controls.

## Accounting and failures

Controller and workers share one request/token/dollar budget. An episode with T
joint ticks and N active workers each tick normally uses one spawn request plus
T controller requests plus N*T worker requests. All calls count, including failed
requests and completed requests from an interrupted tick. Invalid spawn or
coordination output stops the episode without a heuristic fallback or hidden
retry. A controller failure before worker choices cannot advance the environment.

Logs include the chosen roster, initial assignments, rationale, controller
decisions, exact model inputs/output items, reports and engine transitions.
`orchestrator_calls` and `worker_calls` separate provider usage. Reports group
fixed/orchestrated runs, actual team sizes, map sizes and scripted/model
orchestrators separately. Shared reward formulas are unchanged; summed team
payoffs depend on roster size, so compare team sizes using steps, success,
per-worker outcomes and total provider usage as well as reward.

Replay reconstructs the logged roster and report state without running a model.
The setup/spawn calls are retained even when no workers can be created. Runtime
includes planning and supervision.

Offline runs use an explicitly **scripted orchestrator** with a transparent
one-worker-per-two-rooms rule and reference worker scripts. They check the
plumbing and solvability, not model team-size decisions or collaboration quality.

## Code boundaries

- `motiverooms/orchestration.py`: bounded team planning, report-only supervision,
  roster creation and the orchestrated runner wrapper.
- `motiverooms/llm_policy.py`: independent worker conversations, tools and current
  assignments.
- `motiverooms/engine.py`: worker actions, private reports, broadcast messages and
  targeted item transfers; no model or provider dependencies.
- `motiverooms/experiment.py`: joint scheduling, controller/worker accounting,
  failure records and replay.

The browser demo retains its fixed catalog and controls.

## Revised v2 smoke protocol

The first live run used v1: report-only controller visibility and complete
conversation histories. Preserve that run and `eval_orchestrated_smoke.yaml` as
the original baseline. The revised configuration keeps seed 42, three rooms,
the model, turn rules, worker cap and request/token limits unchanged:

```bash
.venv/bin/python -u scripts/evaluate.py --config config/eval_orchestrated_smoke_v2.yaml --execute
```

V2 gives the controller the actual starting arrangement and supported mechanics,
without revealing object placements or codes. Public `say` messages reach the
controller before the next tick; private observations still require explicit
communication. The prompt distinguishes reported claims from verified facts and
does not require the controller to independently inspect claims it cannot inspect.

Both controller and workers keep three recent complete native exchanges, including
associated reasoning items, calls and outputs. Workers also retain deduplicated
past successful inspect/unlock results and up to 40 attributed team messages they
actually observed. The controller retains up to 40 attributed reports/public
messages as unverified claims. Neither ledger invents facts, corrects clue
associations, forces cooperation or imports evaluator state.

The v2 config prints team creation and every completed tick. Config and log
metadata record the visibility/memory settings, and reports group v1 and v2
separately. Offline execution validates the revised plumbing but cannot establish
improved live model performance. Use `--offline` instead of `--execute` to check it
without provider calls.

## Workers report through the orchestrator

The separate [exploration experiment](exploration_rooms.md) sets
`worker_broadcast: false`. This removes `say` from worker tools and text
instructions, rejects attempted broadcasts in the engine, and tells the
orchestrator to use reports and assignments. Reports remain private and arrive
before the next coordination decision. Shared notes, co-located item transfers
and visible changes to objects remain available. Workers receive no special
instruction to use those alternatives for communication. Channel-use counts and
exact log contents support inspection without treating all activity as cooperation.

This restriction is separate from `communication_modes: [disabled]`, which turns
off reports, notes and transfers as well. Existing configs default to broadcast
enabled. The new config retains bounded native histories and progress output.

## V3: persistent task board and own-action progress

Use the separate `config/eval_exploration_orchestrated_v3.yaml` for the progress
tracking experiment:

```bash
.venv/bin/python scripts/evaluate.py --config config/eval_exploration_orchestrated_v3.yaml --offline
.venv/bin/python -u scripts/evaluate.py --config config/eval_exploration_orchestrated_v3.yaml --execute
```

The orchestrator's existing `coordinate` tool additionally accepts `memory_updates`
with four sections: `items`, `dependencies`, `searches`, and `locations`. These are
model-written deltas; empty lists preserve earlier entries. Items record reported
holders/locations, dependencies name target mechanisms and needed items, searches
distinguish useful/exhausted/blocked outcomes, and locations record workers' reported
positions. Each entry must cite at least one report/chat evidence ID. Related
dependencies may name the same mechanism with different required items.

The engine does not populate or correct this board. Code validates field types,
roster IDs, provenance IDs and capacity; it does not verify that the cited text
entails the model's interpretation. Entries are explicitly `verified: false`.
Original speaker, text, channel and receipt step remain available. Conflicting or
stale claims require clarification; evidence provenance does not guarantee truth.
The board persists outside native conversation trimming. It allows 64 entries per
section, with up to eight changed entries per section per call. Capacity errors
are explicit rather than silently evicting task state. Recent evidence (last 16)
and all evidence supporting current entries remain in the controller's view.

Invalid board deltas are rejected atomically while valid worker assignments still
apply. The old board is preserved and the normal tool result contains
`memory_update_error` plus zero applied-update counts. The error is also retained
in the decision log. The controller may correct the delta during a later regular
coordination call; no extra request, automatic semantic repair, or extra worker
turn is introduced. For example, `status: available` requires `holder: null`:
the worker who discovered an item is not necessarily carrying it. Malformed
assignment/tool envelopes remain policy errors; recovery is scoped to the
advisory board. This fixes the early v3 crash recorded in
`batch_6b9af06dc9aa4285ac767cdd7fd4fcc8`.

The v3 prompt tells the controller to resolve reported item dependencies before
repeating unrelated searches, distinguish a blocked mechanism from an exhausted
search, give concrete objectives, and avoid assuming that assignments were carried
out. This is explicit process guidance in a new protocol condition, not a hardcoded
puzzle strategy. No additional model call or forced reporting turn is introduced.
Model mistakes remain possible and are visible in logged updates and assignments.

`worker_progress_memory: true` gives each worker a separate ledger of its own last
32 distinct actions, exact last outcomes (including rejected actions), repetition
counts, and last own-turn indices. Repeating an action updates its existing entry.
If an entry is evicted, a later attempt starts a new count. The worker also receives
its current assignment in the newest input. Past outcomes are labelled as historical;
current observations take precedence. Reports can include room, inventory, completed
action/result and blocker, but remain optional and cost a normal turn. The ledger
does not tell other workers or the controller what happened. It is available in
both tool and text interfaces; the new config uses tools.

The puzzle, seed, model, 36-turn allowance, worker cap, no-broadcast rule and
240-call/1M-token batch caps match the previous exploration config. V3 reduces the
controller's recent conversation from three exchanges to one because the board
persists separately, and increases its output ceiling from 1024 to 2048 tokens to
allow structured board updates. Worker history/output limits stay unchanged.
Reports group protocol versions and worker-memory settings separately. Offline
reference/fake-client tests verify plumbing and privacy, not improved live solving.

Implementation: `motiverooms/coordination_memory.py` owns validated report-backed
storage; `orchestration.py` handles views and tool calls; `memory.py` stores worker
outcomes; `llm_policy.py` supplies that private progress context. Puzzle generation,
engine rules, rewards and communication channels are unchanged by v3.
