# Generated connected episodes

Select `--room-source connected` for this locked-chain generator. New rollouts
default to the [exploration generator](exploration_rooms.md) instead. This source
creates a fresh seeded episode with three shared physical rooms by default.
An orchestrator chooses the worker team and supervises its
progress. Workers start in the first room and move independently. See
[orchestration](orchestration.md) for team-size decisions and reporting.
The room count is configurable; it is not limited to two or three.

```bash
# Preview a reproducible map; no model calls.
.venv/bin/python scripts/rollout_llm.py --room-source connected --num-rooms 3 --seed 42 --dry-run

# Run two connected rooms with a fresh seed; calls Azure.
.venv/bin/python scripts/rollout_llm.py --room-source connected --num-rooms 2

# Reproduce a five-room map; calls Azure.
.venv/bin/python scripts/rollout_llm.py --room-source connected --num-rooms 5 --seed 42

# Preserve a fixed, existing scenario.
.venv/bin/python scripts/rollout_llm.py --room rooms/examples/shared_mat.yaml --dry-run

# Preview a batch with multiple map seeds.
.venv/bin/python scripts/evaluate.py --config config/eval_connected_rooms.yaml

# Check complete trajectories, logs and replay without model calls.
.venv/bin/python scripts/evaluate.py --config config/eval_connected_rooms.yaml --num-rooms 5 --offline
```

## What changes between seeds

The generator varies the physical rooms holding code cards and locked containers,
their furniture placement descriptions, four-digit lock codes, and whether each
connecting passage or final exit requires a code or a key. Node display order is
also shuffled. Randomness comes from a local `random.Random(seed)` instance; the
global random generator is never used.

The current generated topology is a bidirectional chain:

```text
room_1 <--> room_2 <--> ... <--> room_N --> final exit
```

Unlocking passage i gives access to room i+1. Its resources are placed only in
rooms reachable before that lock, so the episode cannot require an item behind
its own lock. Cards and containers can be in different rooms, requiring agents
to exchange information or revisit earlier rooms. Each stage has a visible code
card, a locked box and a key or code card inside that box.

The generator produces an executable reference route. It proves that every room
and the final exit can be reached, without running an exponential optimal-path
search for large maps. This route and all evaluator metadata stay outside agent
observations. The route is not claimed to be optimal or to measure cooperation.

## Movement and interaction

- `enter(code, passage_id)` or `use(item_id, passage_id)` unlocks a passage.
- `move(room_id)` moves one agent through an adjacent open passage and costs one
  turn. Unlocking does not move the agent. Open passages work in both directions.
- Agents see and manipulate objects only in their current room. Observations show
  adjacent passages and destination IDs, inventory, and the codes that agent has
  already read. Reading clues remains private.
- Chat and shared notes retain their existing delivery rules across rooms. Giving
  an item requires the two agents to be in the same room.
- Both agents still choose before the joint tick executes. Movement resolves
  after lock actions, so one agent can open a passage while another crosses it
  during that tick. Both spend a turn.

Only opening the final exit triggers escape. Existing shared-exit rules remain:
`together` releases both agents, regardless of their current positions; `race`
releases the first opener and ends the round. Connecting passages never trigger
either rule. Communication-off still blocks chat, notes and item transfers.

## Batch configuration

`config/eval_connected_rooms.yaml` provides the new batch setup:

```yaml
room_source: connected
num_rooms: 3
episodes: 3
seed: 0
split: all
conditions: [self_play]
both_seats: false
repeats: 1
action_interface: tools
team_mode: orchestrated
max_subagents: 8
```

This selects seeds 0, 1 and 2. Alternatively, specify `seeds: [42, 57, 91]`.
Different seeds generate different maps. Matching incentive/communication
conditions and repeat trials reuse each seeded map so comparisons hold mechanics
fixed. Changing `num_rooms` changes every generated map's size. Existing diagnostic
configs continue loading their frozen suites. Connected maps currently support
self-play and offline reference controls; the older handoff-specific scripted
partners are not used for these maps.

Turn budgets default to the reference route length plus headroom, and the default
joint-step cutoff follows that budget. Explicit `--turn-budget`, `--max-steps` or
evaluation config limits still apply. Larger maps also require larger API-call,
token and output budgets; those caps remain explicit.

## Code and compatibility

- `motiverooms/connected_generator.py`: validated size configuration and constructive
  seeded generation, separate from the frozen diagnostic generator.
- `motiverooms/schema.py`: optional physical locations, agent starts and passages.
  The existing `Room` object represents the complete episode map; its `locations`
  are the individual physical rooms. Nodes belong to one location.
- `motiverooms/engine.py`: per-agent position, local interaction and movement.
- `motiverooms/solver.py`: position-aware shortest paths for small maps.
- The native tool interface exposes `move` only for maps with navigation.

Old YAML files need no new fields and retain their serialized representation.
Old replay snapshots remain compatible. New logs store the entire generated map,
seed, generator version and positions; replay does not regenerate randomness or
call a model. Reports keep different map sizes separate. Existing handoff metrics
do not identify every useful share in these new shared maps; interpret them only
when the scenario supplies handoff annotations.

The browser demo continues using its existing catalog. The generated setup is
available through the research rollout and evaluation runners.
