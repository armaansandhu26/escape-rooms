# MARL env — feedback follow-ups

Consensus: **(b) simultaneous actions per environment step** (not alternating agent turns). Defer async comms, interrupts, action chunking, MAPF/collisions.

## In progress — env layout

- [x] **Shared room layout** (`layout: shared`, one `space`, nodes `wing: shared`) — see `rooms/examples/shared_mat.yaml`; split wings kept for `office_storage`.
- [ ] More puzzles / handoffs in shared space as needed.

## P0 — Core semantics

- [x] Define **environment step**: one tick, all active agents submit one action; apply together.
- [x] Implement `RoomEngine.step_joint({agent: action, ...})` (`WebGame.submit_joint`, session `submit_auto`).
- [x] Turn budget: each acting agent spends 1 per joint step (README updated).
- [x] README: simultaneous (b) vs legacy alternating `step`.

## P0 — Conflict resolution

- [x] `take(same_item)` → one winner, tie-break **A before B** (`motiverooms/joint_step.py`).
- [x] Verb phases: inspect/say/notes/wait → take → enter/use → give.
- [x] Tests: `tests/test_joint_step.py` (say+enter same tick, take conflict).

## P1 — Observations & logging

- [ ] JSONL file writer (dev): one line per joint step (schema: `episode_steps` in API).
- [x] API **`episode_steps`**: grouped `{ env_step, agents: { A: …, B: … } }` in `state_payload`.
- [ ] Observations: partner **past actions** (last step or last k) in `render_observation`.

## P1 — Demo / viz

- [x] Trajectory: **Step N:** with A + B under one number.
- [x] Caption / copy: simultaneous steps; peek shows “Step N — both agents act together”.
- [x] Dual avatar highlight during joint steps (shared + split).

## P2 — Metrics (after joint step lands)

- [ ] Reconcile `help_value` / solver / `turn_budget: auto` with joint-step definition (wall-clock speedup story).

## Out of scope (v1)

- Async message buffers, mid-turn interrupts.
- Action chunking (n-step futures) unless miscoordination is a real problem in practice.
- Same-location physics / MAPF; multiple agents “near” same object is fine.
