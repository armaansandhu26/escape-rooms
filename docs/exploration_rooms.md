# Open-room exploration episodes

## Varied environment (generator version 2)

The unpowered torch is now explicitly described as a fixed assembly station,
which reveals a takeable working torch after battery use. This is a wording
clarification using the existing mechanics. See the [torch diagnostic and
controlled retry](torch_diagnostic.md) for isolated tests and a separate
budget-only retry of the original puzzle.

New single rollouts and the default evaluation use version 2. The physical map
has open passages; puzzle objects are distributed across **all** configured rooms.
All workers start in the Office, which retains the final exit.

- Searchable fixture templates include a bookshelf, calendar, doormat, tool
  cabinet, sofa and noticeboard. Larger maps add more searchable surfaces.
- The desk key and brass token are hidden in randomly selected fixtures. A
  shipping receipt and annotated calendar page are hidden in other fixtures.
- The receipt and calendar page supply distinct four-digit codes. Which one
  opens the supply locker and which opens the document safe varies by seed.
  Each clue explicitly names its target; agents do not need to guess associations.
- The locker holds batteries. The desk key opens a drawer containing an
  unpowered UV torch. Use batteries there and take the working torch, then use
  it on the painting. A brass token opens the display case.
- The painting, display case and safe reveal symbol fragments. Their symbol
  assignments, fragment values and the plaque's required assembly order vary.
- Each room beyond three contains an additional fragment hidden in a searchable
  fixture. Thus four rooms require four fragments and an eight-digit exit code.

Fixtures are searched with `inspect`; portable items are acquired with `take`.
Receipts, calendar pages and inscriptions are readable clue nodes, rather than
portable inventory items. Reading clues is private; opened fixtures are shared
state. Existing structured tools and orchestration apply without engine changes.

```bash
# Reproduce a four-room exploration environment without calling a model.
.venv/bin/python scripts/rollout_llm.py --num-rooms 4 --seed 42 --dry-run

# Validate three seeds with scripted orchestration and deterministic replay.
.venv/bin/python scripts/evaluate.py --num-rooms 4 --offline

# Explicit config; add --execute to call Azure.
.venv/bin/python scripts/evaluate.py --config config/eval_exploration_varied.yaml
```

The generator uses a private seeded RNG and records `generator_version: 2`, the
seed, full map and executable reference in evaluator metadata. `profile: varied`
keeps summaries separate from the original puzzle's `shared` profile. Hidden
item sources, clue associations and reference actions are never inserted into
worker or controller prompts. Placements are balanced across available rooms;
there are no within-room coordinates. The dependency skeleton remains authored,
not an arbitrary random puzzle graph. Its reference demonstrates solvability,
not optimality or a requirement for cooperation.

## Original pilot (generator version 1)

The remainder describes the original puzzle. Both original evaluation configs
explicitly set `exploration_version: 1`. To reproduce it in a single run, use
`--exploration-version 1`; from Python, pass `version=1` to
`generate_exploration_episode`. Generator version 2 is the Python API default.

The `exploration` source is a separate experiment from the original locked-chain
`connected` source. It keeps the structured tools and supervising orchestrator,
but uses open passages, concealed objects and distributed exit-code fragments.

## Run

```bash
# Preview only; no model calls.
.venv/bin/python scripts/evaluate.py --config config/eval_exploration_orchestrated.yaml

# Execute the constructive reference and verify replay; no model calls.
.venv/bin/python scripts/evaluate.py --config config/eval_exploration_orchestrated.yaml --offline

# One live episode, including model-selected team size and ongoing supervision.
.venv/bin/python -u scripts/evaluate.py --config config/eval_exploration_orchestrated.yaml --execute
```

The supplied config uses three rooms, seed 42, up to three workers and private
reports to the orchestrator. Edit `seeds` or `repeats` for additional trials. Use
`--num-rooms N` to vary size. The explicit API budget is shared by the whole batch.

For the revised progress tracking condition, use
`config/eval_exploration_orchestrated_v3.yaml` instead. It adds a report-backed
controller task board and private own-action memory while preserving this puzzle
and its batch caps. See [orchestration v3](orchestration.md#v3-persistent-task-board-and-own-action-progress)
for the exact differences and interpretation limits.

## Puzzle

All workers start together in the Office, which contains the final exit. The
three-room map forms an open bidirectional triangle. For larger maps the passages
form an open ring. One- and two-room maps also work; objects share rooms as needed.

- Inspect the doormat to uncover a desk key. Take it and unlock the drawer.
- Search the tool cabinet for batteries. Bring them to the unpowered torch
  revealed inside the drawer, use the batteries there, then take the working torch.
- Use the torch on the painting to reveal a hidden inscription. Inspect that
  inscription to read a two-digit fragment marked with a moon.
- Search between the sofa cushions for a brass token. Use it on the display case
  and inspect the inscription inside to read a sun fragment.
- Inspect the shipping receipt to read a star fragment.
- Inspect the wall plaque to learn the fragment order. Concatenate the two-digit
  fragments in that order, preserving leading zeros, and enter the resulting
  six-digit code at the Office exit.

Seeds change Storage/Lounge placement, fragment values, their assembly order and
object display order. Each room also contains a small amount of ordinary scenery
that can be inspected. Extra rooms beyond three add one symbol fragment each and
extend the exit code by two digits. The exit remains in the Office.

Searching uses `inspect`, not a new free-text search command. Concealed items
cannot be taken before their parent is searched or opened. Unlocking/searching
changes shared state; reading a revealed inscription remains private to the
reader. There are no spatial coordinates within a room. Movement, searches,
reading, item use, transfers and communication each cost one turn, including
rejected attempts. Together-escape still releases everyone when one worker opens
the final exit.

The generator supplies a constructive reference route and a turn budget equal to
its length plus headroom. For the default seed the route takes 20 joint ticks and
the allowance is 36. This is a feasibility witness, not an optimality claim or a
cooperation requirement. The existing numeric-code BFS oracle does not interpret
text clues; use the constructive reference for these episodes.

## Communication experiment

`worker_broadcast: false` removes `say` from workers' tool schemas and text action
instructions. The engine also rejects attempted broadcasts and charges the usual
turn, so the restriction cannot be bypassed with a text action. Communication is
otherwise enabled:

- `report(message)` sends a private message to the orchestrator, available before
  its next coordination decision. Other workers cannot read the report.
- The orchestrator can relay information through individual task assignments.
- Shared notes retain their existing semantics: `write_note` appends to a global
  note store and `read_notes` explicitly reads it. Writing alone does not broadcast
  the text into workers' observations.
- Item transfers require co-location. Workers can observe shared object changes,
  but teammate action arguments remain private.

Workers receive ordinary tool descriptions, with no special instruction to use
notes or transfers as a substitute messaging strategy. The orchestrator cannot
read shared notes or inspect room state; it depends on reports. This allows us to
observe alternative coordination without requiring or rewarding it explicitly.
Shared notes are an existing explicit channel, so note use alone is not evidence
that agents invented a new communication protocol.

The report includes counts of broadcasts, private reports, note writes, note
reads and item transfers. The JSONL log retains exact actions, messages, shared
state transitions and controller assignments. Counts describe usage, not intent
or causal cooperation. Report grouping distinguishes broadcast-enabled runs from
broadcast-disabled runs and keeps this puzzle family separate from locked chains.

## Implementation boundaries

`exploration_generator.py` owns the seeded puzzle and evaluator reference. Generic
schema additions are searchable nodes, inspectable scenery, text clues and numeric
codes of variable length. The engine implements those mechanics without knowing
the puzzle's symbols or assembled answer. Agent observations and tool schemas do
not receive evaluator routes or unrevealed clue text. Full maps and references are
stored in evaluator logs for inspection and deterministic replay.

The browser demo's catalog remains separate; this experiment runs through
`scripts/evaluate.py`. Existing configs retain their original puzzle source and
broadcast behavior.
