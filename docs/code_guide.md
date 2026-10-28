# Code guide

Start with `scripts/rollout_llm.py` for a single research run or
`scripts/evaluate.py` for a batch. The browser demo uses a separate fixed catalog.

## Follow one episode

1. `load_episode()` loads a specified YAML room or generates an exploration map.
   `exploration_generator.py` selects the generator version. `varied_exploration.py`
   owns the searchable fixture catalog, placements, clue-to-lock wiring and a
   constructive solution. `schema.py` validates the resulting map.
2. `orchestration.py` asks the controller to choose a bounded worker roster and
   initial assignments. Each worker gets a separate `LLMAgentPolicy` and private
   conversation. New research defaults use two to four workers and v5 manager
   supervision: workers keep their objectives between report/silence check-ins.
   Earlier protocols retain per-tick coordination for comparison.
   `coordination_memory.py` validates persistent report-backed task/map updates.
3. `llm_policy.py` supplies observations and tool definitions to the provider.
   `azure_openai.py` handles transport and request accounting.
4. `tool_calls.py` validates named calls and converts arguments to engine actions.
   Discovery and validation share the same capability rules. Schemas contain no
   hidden room facts. The engine checks visibility and physical feasibility.
5. `experiment.py` collects choices from the same pre-step state, commits one
   joint step, delivers private results and logs the transition. `engine.py`
   applies mechanics; `rewards.py` computes rewards. Replay needs no provider.

## Configure rooms and randomness

```bash
# Five exploration rooms, reproducible placements and clue assignments; no API calls.
.venv/bin/python scripts/rollout_llm.py --num-rooms 5 --seed 42 --dry-run

# Your own YAML scenario; no API calls.
.venv/bin/python scripts/rollout_llm.py --room rooms/examples/shared_mat.yaml --dry-run
```

Omit `--seed` to generate a fresh recorded seed. `--room` selects the complete
scenario; `--num-rooms` controls the number of physical rooms in generated maps.
Exploration maps have open ring passages (one- and two-room maps use a chain).
Search fixtures to reveal hidden items or clues, then take items or inspect clues.
Fixtures stay in their rooms; keys, batteries, the torch and tokens are portable.
`--room-source connected` retains the separate locked-chain generator.

## Tool protocol

The interaction is MCP-style: discover named tools with JSON schemas, submit a
structured call, validate it, execute it and return a result tied to its call ID.
The current transport is Azure Responses native function calling. This project
does not implement an MCP server or MCP wire protocol.

`--action-interface tools` and `--team-mode orchestrated` are the single-run
defaults. Exactly one worker action is accepted per tick; invalid calls consume
a turn. Controller calls consume the shared API budget only when invoked.
Worker reports trigger a next-tick check-in without exposing private histories;
otherwise v5 checks on a worker after five silent ticks.

See [tool contracts](tool_interface.md), [room generation](exploration_rooms.md),
and [orchestration](orchestration.md) for details. Run the offline suite with:

```bash
.venv/bin/python -m pytest -q
```
