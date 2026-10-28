# Escape Rooms

Multi-agent escape rooms with autonomous LLM workers and a supervising orchestrator. Workers explore, discover clues, use items and report progress; the orchestrator coordinates from those reports. Current work runs inference-time episodes with Azure OpenAI and saves replayable JSONL logs.

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/armaansandhu26/escape-rooms.git
cd escape-rooms
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,azure]'
```

For live model runs, create a `.env` in the project root:

```dotenv
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/v1/
AZURE_OPENAI_MODEL=your-deployment-name
```

The runners load `.env` automatically; existing environment variables take precedence. Keep credentials out of Git. Preview, offline validation and replay require no API key.

## Run

```bash
# Preview the default three-room episode; no model calls.
python scripts/rollout_llm.py --dry-run

# Run one live episode; writes a recording under runs/.
python scripts/rollout_llm.py --num-rooms 3 --seed 42

# Preview the evaluation batch; add --execute for live Azure calls.
python scripts/evaluate.py

# Validate scripted reference solutions and deterministic replay offline.
python scripts/evaluate.py --offline

# Open the replay UI (use --port 8766 if needed).
python scripts/play_web.py
```

Open [localhost:8765](http://127.0.0.1:8765/). The replay loads a completed recording automatically, preferring a successful v5 run; **Load JSONL** imports another completed episode. Play/pause, restart, step and seek keep rooms, worker-colored item highlights, manager messages and the trajectory synchronized. The expandable **How this puzzle unlocks → Easy interpretation** guide sits below the rooms, turns completed steps green and reveals clue values as playback advances. The guide currently supports the original exploration puzzle. Replay makes no model calls. The older scripted demo is at [/demo](http://127.0.0.1:8765/demo).

## Orchestrator

The default **v5 manager** chooses the full worker team upfront (2–4 by default) and assigns objectives. Workers act independently between check-ins. New reports trigger a manager call; five silent ticks per worker also trigger a check-in. Assignments persist, and reported exit information can be relayed to everyone.

The orchestrator receives the public setup and explicit worker reports, plus its report-backed memory. It does not receive hidden placements, private worker observations/tool results or the reference solution. It reasons about reported evidence using general strategy guidance; its conclusions can be wrong. Set team bounds with `--min-subagents` / `--max-subagents`.

See [manager supervision](docs/manager_supervision.md), [orchestration details](docs/orchestration.md) and [implementation](motiverooms/orchestration.py).

## MCP-style tools

Workers use native function calls with strict JSON schemas: `inspect`, `take`, `use`, `enter`, `move`, `report` and communication/item-transfer tools where enabled. Each active worker submits one action per tick; choices use the same pre-step state, then the engine applies the joint step and returns private tool results. Invalid actions consume turns.

This is an MCP-style structured tool interface implemented through Azure Responses function calling; **no MCP server is required**. Contracts live in [tool_calls.py](motiverooms/tool_calls.py), worker prompts and memory in [llm_policy.py](motiverooms/llm_policy.py), and scheduling/logging in [experiment.py](motiverooms/experiment.py). See the [tool interface guide](docs/tool_interface.md).

## Rooms and randomness

The default **version 2 exploration generator** varies object placements, item hiding places, clue-to-lock associations, symbol values and final code order. The dependency structure is authored; seeds vary the puzzle within it. Workers start in the Office, which contains the final exit. Extra rooms add hidden symbol fragments.

```bash
# Fix map size and seed for reproducible environment generation.
python scripts/rollout_llm.py --num-rooms 5 --seed 42 --dry-run

# Load a specified YAML scenario instead of generating one.
python scripts/rollout_llm.py --room rooms/examples/office_storage.yaml --dry-run
```

Omitting `--seed` selects a fresh seed and records it. A fixed seed reproduces the environment, not necessarily the model's decisions. Use `--exploration-version 1` for the original pilot. See [generation rules](docs/exploration_rooms.md), [varied generator](motiverooms/varied_exploration.py), [original generator](motiverooms/exploration_generator.py) and the [default evaluation config](config/eval_exploration_varied_manager_v5.yaml).

## Key files and checks

- [Code guide](docs/code_guide.md): module responsibilities and execution flow.
- [Room engine](motiverooms/engine.py): state transitions and action legality.
- [Recording projection](motiverooms/recordings.py): saved episodes for the UI.
- Replay UI: [HTML](motiverooms/static/replay.html), [JavaScript](motiverooms/static/replay.js), [CSS](motiverooms/static/replay.css).
- CLI entry points: [single rollout](scripts/rollout_llm.py), [evaluation](scripts/evaluate.py), [web server](scripts/play_web.py).

Run `python -m pytest -q` for the test suite. Keep the core environment independent of provider/GPU dependencies, use explicit seeded RNGs, and leave third-party sources unchanged; see [design decisions](DECISIONS.md).
