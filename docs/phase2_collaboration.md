# Phase 2: collaboration between two AI agents

**Status: original Azure pilot complete; communication-explained follow-up ready.**

The original pilot (`runs/batch_a92d10cf0689493f8fe408e8723d4d6f`) completed all
12 attempts and six matched pairs. Both agents escaped every time, with no
messages or shared notes. They did not acquire partner-needed codes. Communication
on/off produced the same actions and episode lengths. These observations do not
establish whether the agents understood the benefit of communicating.

## Follow-up: explain communication clearly

`config/eval_collaboration_explained.yaml` keeps the original rooms, model,
sampling, team scoring, turn costs and 12-attempt matrix. It adds an explicitly
labelled `communication_instruction: explained` condition. The original config
and logs retain the original instructions.

The added instructions explain that some clues help the other agent, exchanging
information can sometimes shorten a route, messages appear on the next step,
and sending costs a turn. They explain free reading of received messages,
shared-note costs, the 40-word limit, and the inability to act after escaping.
Communication remains optional, with no guaranteed benefit in every room. The
off condition gets an explicit explanation that channels are unavailable.

This adds information about the potential strategy benefit as well as tool
mechanics. Any change in behavior would show sensitivity to that combined
explanation, not isolate which sentence caused it or demonstrate learned
cooperation. Configs, episode metadata, pair checks and summary data retain the
instruction condition separately. The original condition remains the default.

Preview, then rerun the follow-up:

```bash
.venv/bin/python scripts/evaluate.py --config config/eval_collaboration_explained.yaml
.venv/bin/python scripts/evaluate.py --config config/eval_collaboration_explained.yaml --execute
```

Use `--limit-episodes 2 --execute` instead for a single-pair smoke test.

The first phase checked the room design and one AI working with programmed
teammates. This phase asks whether two AI agents exchange useful information,
and whether allowing communication improves their performance.

Both characters use the same Azure deployment, with separate clients and separate
observations. They choose at the same time from the state before the step.
Neither receives the other's private clues, reasoning or current choice. Each
request contains that character's current observation; there is no shared model
conversation or persistent reasoning history. Own recent actions/results and
received messages remain part of the observation.

## First experiment

`config/eval_collaboration_pilot.yaml` defines **12 attempts / six matched pairs**:

- Three existing development rooms: both agents benefit from help, one benefits
  more, and neither benefits.
- Communication on and off, with two trials of each room. The second trial
  reverses the order of the communication conditions.
- The same room, v3 information, model, sampling settings, turn costs and team
  scoring in both conditions. No strategy coaching, action repair or RL updates.
- Both seats are AI-controlled. We do not duplicate an attempt by relabelling
  which identical AI is the evaluated agent.

With communication **on**, the existing chat, shared-note and item-transfer
actions are available. With communication **off**, the engine rejects `say`,
`write_note`, `read_notes` and `give`; rejected attempts still use a turn. Past
partner actions stay private in both conditions. The prompts explain which
actions are available. Room descriptions remain fixed, including references to
optional help; this preserves the same puzzle across conditions. Partner escape
status remains visible, so this removes explicit communication channels rather
than every possible signal between characters.

All rooms retain a solo route. Cooperation can help, but is never required for
escape. This first pilot tests access to all explicit channels together; it does
not isolate chat from shared notes.

## Run commands

Preview the paid pilot without making requests:

```bash
.venv/bin/python scripts/evaluate.py --config config/eval_collaboration_pilot.yaml
```

Check programmed controls without Azure:

```bash
.venv/bin/python scripts/evaluate.py --config config/eval_collaboration_pilot.yaml --offline
```

Run a two-attempt Azure smoke test (one communication-on/off pair):

```bash
.venv/bin/python scripts/evaluate.py --config config/eval_collaboration_pilot.yaml --limit-episodes 2 --execute
```

After reviewing that pair, run the full pilot as a fresh batch:

```bash
.venv/bin/python scripts/evaluate.py --config config/eval_collaboration_pilot.yaml --execute
```

The session cap is 250 provider calls and 600,000 tokens. The selected rooms have
at most 224 active-agent turns across the full matrix. Dollar cost stays unknown
until actual deployment prices are supplied in the budget configuration.

Offline controls follow the known joint solution with communication on and each
character's solo route with it off. They check that help can save steps and that
the pipeline records it correctly; they are not AI collaboration results.

## What to inspect

The generated `summary.md` explains the experiment in plain language. It keeps
the three room types and communication conditions separate and reports:

- Whether both agents escaped, plus errors and interrupted attempts.
- Average actions per agent, average episode length and length range.
- Messages, relevant code/item shares, later use by the recipient, and attempts
  where both agents used help.
- Paired changes in episode length, plus provider requests and recorded tokens.

The CSV/JSON files retain medians, variation, payoffs and separate A/B statistics,
including escape timing, invalid actions, wrong attempts, sharing and usage.
Shared-note writes are counted, but their contents are not attributed by the
existing chat/item sharing metrics. Inspect note trajectories if agents use them;
zero chat shares alone would not establish zero cooperation.

`communication_comparison.json` checks pairs against the actual logged room hash,
engine configuration, source hash, controller mapping and model settings.
Missing, interrupted and mismatched pairs are excluded from paired differences.
`plan.json` preserves the intended batch so early stops remain visible.
Sharing rates without acquired opportunities remain undefined.

Review the smoke test for privacy, valid commands and complete logs first. Then
review the pilot by room type: do agents acquire and send useful codes, use what
they receive, and reduce total work or time? Retain negative results. Do not
change prompts just to obtain sharing. Expand to more development rooms and
repeats only after reviewing this pilot; keep held-out rooms reserved. These
tests establish a fixed-model baseline before later MARL training.

## Local verification

All 131 tests passed, including 14 phase-2 checks. The 12-attempt offline batch
completed with six verified pairs and zero provider calls:
`runs/batch_4c18520472094664ae9704a2c6dc26df`.
Reference cooperation saved three episode steps in the balanced room, five in
the unequal room and zero in the no-benefit room.
