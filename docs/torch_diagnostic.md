# Torch diagnostic and controlled exploration retry

The subsequent [v4 active coordination retry](active_coordination.md) addresses
the observed exploration/assignment deadlock with two to four workers. The
commands below retain the earlier v3 conditions for comparison.

The torch is a fixed assembly station. Bring batteries to it, use the batteries
on the station, and take the working UV torch it reveals. The working torch is
portable. The station keeps the existing `unpowered_torch` identifier, so tool
schemas, engine mechanics and saved action formats remain compatible.

New varied episodes explicitly describe this distinction. Original generator
version 1 retains its original wording by default. The clarified three-room
comparison changes only the drawer/station descriptions and protocol identity;
object locations, lock requirements, codes and constructive actions stay fixed.

## 1. Check the isolated dependency

Run commands from the repository root using the existing `.venv` and Azure `.env`
configuration. A command without `--execute` or `--offline` only previews the plan.

```bash
# Preview ten single-worker, single-room trials. No API calls.
.venv/bin/python scripts/evaluate.py --config config/eval_torch_dependency.yaml

# Exercise all ten scripted reference solutions and replay. No API calls.
.venv/bin/python scripts/evaluate.py --config config/eval_torch_dependency.yaml --offline

# One live model trial first. Calls Azure; at most 12 worker requests.
.venv/bin/python -u scripts/evaluate.py --config config/eval_torch_dependency.yaml --limit-episodes 1 --execute

# If the first trial works, evaluate the full ten-seed diagnostic. Calls Azure.
.venv/bin/python -u scripts/evaluate.py --config config/eval_torch_dependency.yaml --execute
```

This diagnostic has visible batteries, a station, painting and exit. There is no
controller, navigation, hidden battery search or partner. Seeds vary the code,
battery surface and object order. The shortest dependency chain is six actions:

```text
take(batteries)
use(batteries, unpowered_torch)
take(uv_torch)
use(uv_torch, painting)
inspect(painting_inscription)
enter(<discovered code>, final_exit)
```

The reference and answer stay outside policy inputs. Workers see normal object
descriptions and normal tool results. Each trial allows 12 turns; the full batch
allows 150 requests and 500,000 tokens in total. A scripted or mocked success
checks implementation, not model capability. Failure here calls for inspecting
the first blocked dependency before increasing the full environment budget.

## 2. Retry the original three-room puzzle with more time

```bash
# Preview: same version-1 seed-42 puzzle, original descriptions, 60 turns.
.venv/bin/python scripts/evaluate.py --config config/eval_exploration_retry_60.yaml

# Execute one live episode. Calls Azure.
.venv/bin/python -u scripts/evaluate.py --config config/eval_exploration_retry_60.yaml --execute
```

This retains the earlier v3 memory/controller settings, model, three-worker cap,
no-broadcast condition and original puzzle. Per-agent turns and the joint-step
cutoff both increase from 36 to 60. The shared batch caps increase to 300 requests
and two million tokens. A three-worker team running all 60 ticks requires at most
241 requests: one spawn plus 60 controller and 180 worker calls. Roster selection
and model sampling may still vary, so one retry cannot isolate a causal effect.

The runner now honors `turn_budget` in YAML. `--turn-budget` and `--max-steps` can
override it for future experiments. API budgets remain independent of turn limits.

## 3. Compare clarified descriptions if needed

```bash
# Same puzzle, model, protocol and 60-turn limits; explicit station wording.
.venv/bin/python -u scripts/evaluate.py --config config/eval_exploration_retry_60_station.yaml --execute
```

Treat this as a separate condition. Its room ID ends in `_station`, and logs record
`torch_affordance: explicit_station`. Original wording is recorded as `legacy`
in report grouping. Machine-readable summaries separate affordances, turn budgets
and step cutoffs. Earlier logs and original configs are preserved.

If the micro-task succeeds but full-room runs stall, inspect navigation,
resource ownership, repeated work and assignment/report overhead. If the original
60-turn retry succeeds, evaluate more seeds before claiming reliable solving.

## Read the results

Every execution prints a new run directory containing `summary.md`,
`episodes.csv`, structured summaries and replayable episode logs. Inspect torch
milestones directly from the printed directory:

```bash
.venv/bin/python scripts/inspect_progress.py runs/batch_REPLACE_WITH_PRINTED_ID
```

This prints when batteries were collected, the torch was powered and taken,
the painting was illuminated/read, and escape occurred. It also counts rejected
actions, wrong attempts and repeated inspections. Repeated inspections can be
justified after state changes; the count is not automatically a waste measure.
The command is read-only and makes no model requests.
