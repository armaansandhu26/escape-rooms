# Manager supervision, protocol v5

Earlier protocols invoked the controller before every tick. V5 makes it an
overseeing manager: it chooses the full two-to-four-worker team upfront, gives
durable objectives and lets workers choose and execute their own actions.
Assignments persist between check-ins. The manager helps with blockers, connects
discoveries and relays information when workers report back.

## When the manager is invoked

- New explicit reports trigger one manager call before the next worker tick.
  Reports arriving in the same joint tick are batched into that call.
- If enabled, newly observed public team chat also triggers a call. The supplied
  no-broadcast experiments do not enable this channel.
- Each active worker has a silence timer. After five completed ticks without
  a report or silence check for that worker, the manager is invoked with that
  worker listed as overdue. A report from another worker does not reset it.
- A silence check resets the checked worker's timer so an unanswered check does
  not invoke the manager again every tick. New reports can still trigger it.

Initial assignments come from spawning, so no redundant controller call occurs
before tick 1. With no reports and a five-tick interval, check-ins occur before
ticks 6, 11, 16, and so on. In a 20-tick reference run, this yields four manager
calls including spawn, compared with 21 controller calls in v4. These are call
counts for that particular no-report trajectory, not a predicted live cost saving.

The scheduler uses report arrivals, elapsed ticks, active worker IDs and public
chat only when enabled. It never reads private worker observations or action
results to detect progress. Reports remain private and still consume one worker
turn. Workers are instructed to report meaningful discoveries, objective
completion or genuine blockers rather than unchanged status. Nothing forces a
report every tick or prevents ordinary autonomous actions between check-ins.

The manager can keep all current objectives by returning empty assignment updates.
Its new prompt asks it to check on overdue workers without unnecessarily changing
other useful objectives. Worker prompts explicitly say they do not need approval
for individual actions, and should verify stale guidance against current state.
This is model guidance, not a guarantee of successful reasoning.

## Shared exit briefing

When a worker reports an exit, the manager records its exact node ID and room ID
in the new `exits` board section, with `code: null` until known. Entries require
received evidence IDs. When reports support the answer, the manager can assemble
and store the full code, preserving leading zeros. The engine does not supply or
validate the truth of that interpretation.

After a manager check-in, these model-written exit records are relayed to every
worker as a **persistent manager exit briefing**, separate from each worker's
objective. It remains visible on future calls even when assignments change or
conversation history is trimmed. This is an intentional manager-to-team channel;
raw worker reports and private observations are still not automatically shared.

Workers can navigate to the reported exit room and enter the code themselves, or
report their findings for the manager to relay and assign a finisher. The prompt
asks the manager to choose someone already near the exit when recent reports
support that choice. A remote-target rejection should lead to correcting the
route/location, not inventing another exit in the current room. Model errors
remain possible; briefings retain evidence and `verified: false` labels.

The console prints changed exit briefings. Structured controller results record
the delivered information. No exit location or answer is filled in from hidden
map state; if workers have not reported it and the manager has not recorded it,
the briefing remains empty. V1–v4 tools and memory schemas remain unchanged.

## Run the same original puzzle

```bash
# Preview only; no provider requests.
.venv/bin/python scripts/evaluate.py --config config/eval_exploration_manager_v5.yaml

# Scripted reference and replay; no provider requests.
.venv/bin/python scripts/evaluate.py --config config/eval_exploration_manager_v5.yaml --offline

# Live model run. Calls Azure.
.venv/bin/python -u scripts/evaluate.py --config config/eval_exploration_manager_v5.yaml --execute
```

The puzzle, seed 42, explicit station, two-to-four-worker range, 60-turn limit,
model and batch caps match the v4 controlled retry. V5 changes scheduling and
the corresponding manager/worker autonomy instructions. Prior v1–v4 configs
retain their original per-tick behavior.

Change the silence interval with YAML `orchestrator_check_in_interval: 5` or
`--orchestrator-check-in-interval 5` in either runner. This option applies to v5;
older protocols still run per tick. An interval of one permits per-tick silence
checks. A frequently reporting team can also cause frequent manager calls.

`config/eval_exploration_varied_manager_v5.yaml` is the evaluation default for
varied maps; the single-rollout CLI also defaults to v5. The console identifies
actual manager check-ins and their triggers, then prints changed objectives.
JSONL records every real controller invocation and its trigger, without inventing
calls on skipped ticks. Metadata and summaries retain `supervision_mode: manager`
and the silence interval separately from older runs. Replay needs no model.
