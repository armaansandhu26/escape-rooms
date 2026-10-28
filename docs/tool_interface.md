# Native room tools

New LLM rollouts default to `action_interface: tools`. The Azure Responses client
sends function definitions with strict JSON schemas, `tool_choice="required"`
and `parallel_tool_calls=False`. A model chooses a tool; the runner validates its
arguments, submits one action to the room engine and returns the outcome as a
native `function_call_output` tied to the original `call_id`. No MCP server is
needed for this interface.

This follows the [documented function-calling loop](https://developers.openai.com/api/docs/guides/function-calling).
The implemented provider is Azure OpenAI Responses; other providers would need
an adapter for their request, output and continuation formats.

## Tool contracts

| Tool | Required arguments | Effect |
|---|---|---|
| `inspect` | `node_id: string` | Inspect a visible node; read a clue |
| `take` | `item_id: string` | Pick up a visible item |
| `enter` | `code: string`, `node_id: string` | Try a code on a lock |
| `use` | `item_id: string`, `node_id: string` | Use an item from the agent's bag |
| `give` | `item_id: string` | Transfer an item to the partner |
| `say` | `message: string` | Send a message |
| `write_note` | `text: string` | Append a shared note |
| `read_notes` | none | Read shared notes |
| `wait` | none | Spend a turn |
| `move` | `room_id: string` | Move through an open passage (connected maps only) |
| `report` | `message: string` | Report to an orchestrator (supervised workers only) |
| `give_to` | `item_id: string`, `agent_id: string` | Transfer an item in teams larger than two |

Identifiers must exactly match observation IDs. Codes are strings, preserving
leading zeros. Schemas reject extra fields; local validation also rejects missing
fields, incorrect types, malformed JSON and duplicate argument keys. Message
whitespace is normalized before conversion to the engine's existing action format;
the engine applies its 40-word limit. Tool schemas are static: they expose no
node lists, private codes, feasibility hints or oracle information.

Communication-off and solo policies do not receive `give`, `say`, `write_note`
or `read_notes`. The validation layer rejects those calls if a model returns one
anyway; the engine remains the authority for room visibility and action legality.
Connected maps additionally expose `move`; legacy maps retain their original
tool set. See [connected episodes](connected_rooms.md) for map generation and
movement rules.
New research runs also use [an orchestrator](orchestration.md) to choose the
worker roster and update assignments throughout the episode.

## Execution and memory

Both agents choose from the same pre-step state. The runner waits for both choices,
applies the joint step once, then returns each agent only its own outcome. The
result includes the canonical action, validity and engine result. Public messages
and shared changes reach the other agent through the existing observation rules.
Each call consumes one turn, including a rejected action or a wrong code.

Native tool mode retains a private conversation per policy: observations, native
model output (including reasoning items needed for continuation) and tool results.
The next user message supplies the next observation. Do not reuse a policy across
episodes: construct a fresh policy/client for each seat, as the evaluation runner
does. Standalone callers using `choose_action()` must call
`policy.record_outcome(outcome)` after committing the joint step and before the
next choice. `run_episode()` handles this automatically.

In strict simultaneous mode, zero calls, multiple calls, invalid arguments and
incomplete responses become one invalid engine action. Text resembling an action
is never executed as a fallback. Multiple calls are all rejected, consume one
turn in total and receive failure outputs. Provider errors stop the episode
without inventing a transition. Assisted mode remains an explicit separate
condition; it returns rejection feedback before requesting another choice.

## Code boundaries

- `motiverooms/tool_calls.py`: tool contracts, schema generation, native call
  representation and conversion to `Action`. No provider SDK or room-state access.
- `motiverooms/azure_openai.py`: Responses transport, native output serialization,
  request accounting and logging. Text and tool modes share one request path.
- `motiverooms/llm_policy.py`: prompts, action-interface selection and private
  conversation lifecycle. No engine mutation.
- `motiverooms/experiment.py`: simultaneous scheduling, engine execution,
  outcome delivery and replayable logs.

The room engine, reward rules and scripted policies continue using their existing
action interface. Tool mode converts validated arguments at that boundary.

## Running and comparing

```bash
# Preview the new protocol without a provider request.
.venv/bin/python scripts/rollout_llm.py --dry-run

# Run native tools; this command calls Azure.
.venv/bin/python scripts/rollout_llm.py --action-interface tools

# Reproduce the older text-action interface; this also calls Azure.
.venv/bin/python scripts/rollout_llm.py --action-interface text

# Preview a native-tool evaluation.
.venv/bin/python scripts/evaluate.py --config config/eval_collaboration_explained.yaml --action-interface tools

# Validate both interfaces locally with mocked provider responses.
.venv/bin/python -m pytest -q
```

Evaluation YAML accepts `action_interface: tools` or `action_interface: text`;
omitting it selects tools. The CLI option overrides YAML. Offline controls remain
scripted and make no model calls.

Logs record the interface per policy, tool definitions, actual provider inputs,
native output items, canonical actions, tool results and usage. Budget reservations
include schemas and conversation history. Replay verifies engine transitions and
metrics from canonical actions without calling a model. Reports group tool and
text runs separately; historical records without interface metadata are treated
as text mode.

Tools change both action encoding and memory compared with the previous runner.
Recalibrate and repeat matched conditions under the same interface. This change
alone does not measure the isolated effect of JSON formatting or memory.
