# Next run: observation calibration

## Current follow-up: v2 versus v3

The first paired calibration is complete in
`runs/batch_508f226664134038a83b8ac748794454`. V1 escaped on 0/5 rooms with
43 rejected commands and 4 wrong attempts in 70 actions. V2 escaped on 3/5,
with 3 rejected commands and 10 wrong attempts in 59 actions. Thus v2's combined
unsuccessful-action rate was 13/59 (22.0%), despite its low invalid-action rate.

The two v2 failures included interpreting the digits in `exit_key_2334` and
`right_token_7677` as codes, rather than using/obtaining the physical items.
V3 adds explicit CODE LOCK/ITEM LOCK labels and verb templates, plus a rule
that identifier suffixes are not evidence of codes. It keeps the v2 renderer,
system prompt, room files, model settings, rewards, turn budgets and engine
semantics otherwise unchanged. Wrong attempts still return `valid: true` with
result `wrong`; they now appear prominently alongside invalid actions in reports.

The next configuration is `config/eval_lock_calibration.yaml`: the same five
development seeds, solo, one trial each of v2 and v3, ten episodes total.
Preview it before adding `--execute` for paid requests:

```bash
.venv/bin/python scripts/evaluate.py --config config/eval_lock_calibration.yaml
.venv/bin/python scripts/evaluate.py --config config/eval_lock_calibration.yaml --execute
```

Check escape count, invalid actions, wrong attempts, their combined rate, and
repeated failures. Keep the practical 4/5-escape and no-three-turn-failure-loop
targets; use **combined unsuccessful actions at most 5%** as the mechanics screen,
rather than judging invalid commands alone. These are screening heuristics for
this small development sample, not established statistical thresholds. If v3
still fails, inspect its wrong attempts before expanding the social experiments.

The original v1/v2 plan below is retained for reproducibility.

## Evidence from the first pilot

Source: `runs/batch_ddf84d76c77c468f9b0413817533ba25`.
All four episodes used `parallel_branches`, development seed 0, mixed rewards,
private partner actions, together-escape, a 14-turn budget, and gpt-5.4-mini.

The LLM made 29 invalid actions in 50 submitted actions:

| Observed failure | Count | What the observation was missing or making hard to track |
|---|---:|---|
| Trying to take a clue card | 11 | Object types were implicit; the model treated a node as an inventory item |
| Trying to unlock an already-open node | 15 | Open status was present, but the summary did not pair the previous action with the last result or highlight completed unlocking |
| Losing a simultaneous item conflict | 3 | Shared item competition; this needs partner adaptation later and cannot be diagnosed by solo trials |

The solo agent repeated the same unsuccessful unlock for nine consecutive turns.
Successful team escape was also insufficient as a competence measure: in one
helpful-partner episode the script opened all five locks/exits and the LLM opened
none. These observations motivate a solo mechanics check before a larger social
comparison. They do not establish that unclear rendering is the sole cause.

## Change for the next run

The new `v2` observation contains the same underlying visible state with:

- Explicit node types, including `type=clue` and the distinction between inspecting
  a clue and taking an inventory item.
- Exact item IDs in the bag and a separate list of currently takeable items with
  their visible sources. Locked-container contents remain hidden.
- Explicit completed-lock text: an open node cannot be unlocked again. Its
  available contents remain independently takeable/inspectable.
- The agent's last three own actions and its last action beside the existing last
  result. This uses only actions the agent already submitted.

The old `v1` renderer is retained for matched comparisons and historical replay.
No solver path, secret code, partner private action, valid-action mask, automatic
correction or retry is supplied. Engine transitions and invalid-action turn costs
are unchanged. The policy system prompt is identical across v1 and v2.

## Paired calibration design

`config/eval_calibration.yaml` selects five development room seeds (0–4), one
solo trial per format per room: **10 episodes**, interleaving v1/v2 within each
room. It keeps the deployment, temperature (0.2), output cap (300), reward mode,
room files and per-room turn budgets fixed. At most 140 policy requests are
needed if all ten episodes exhaust their 14-turn budgets. The session guards
allow 200 requests and 500,000 accounted/reserved tokens; conservative token
reservations can still stop the batch early. Prices remain unset until supplied.

Preview:

```bash
.venv/bin/python scripts/evaluate.py --config config/eval_calibration.yaml
```

Execute when ready (paid Azure requests):

```bash
.venv/bin/python scripts/evaluate.py --config config/eval_calibration.yaml --execute
```

The report separates observation versions and includes escape counts, invalid
actions/action totals, the longest repeated failing action, and the number of
completed matched v1/v2 pairs. Inspect error/truncation counts first; an incomplete
pair should not be treated as a clean format comparison.

## How the result changes the following run

Use these as practical screening criteria, not statistical claims:

1. v2 escapes on at least four of five rooms.
2. At most 5% of v2 LLM actions are invalid.
3. No v2 episode repeats the same failing action for three or more consecutive
   turns. Check these per episode as well as in aggregate.

If v2 clears these checks and improves on v1, keep it for the next partner pilot:
helpful and solo-route scripts across several rooms, both LLM seats. Before
interpreting that shared-room comparison, address the scripts' limited recovery
when the LLM takes an item they planned to use. Follow with private-handoff rooms
to measure information sharing instead of inferring cooperation from team escape.

If v2 still fails, inspect the saved observations/responses at the first failure
and the first repeated failure. Fix a specific remaining representation or policy
issue and repeat this calibration. Do not simultaneously increase turn budgets,
enable repair, change rewards, replace the model, and claim an observation effect.
If both formats already perform well on the new rooms, repeat trials before
attributing the first pilot's result to rendering. Five rooms with one trial each
are only an initial diagnostic sample.
