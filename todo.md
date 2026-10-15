# MARL env — feedback follow-ups

Consensus: **(b) simultaneous actions per environment step** (not alternating agent turns). Defer async comms, interrupts, action chunking, MAPF/collisions.

## In progress — env layout

- [x] **Shared room layout** (`layout: shared`, one `space`, nodes `wing: shared`) — see `rooms/examples/shared_mat.yaml`; split wings kept for `office_storage`.
- [ ] More puzzles / handoffs in shared space as needed.

## P0 — Core semantics ✅

- [x] Environment step = `step_joint`; conflict rules; README; demo wired.

## P1 — Observations & logging (demo)

- [ ] JSONL file writer (dev): one line per joint step (mirror `episode_steps` schema).
- [x] API **`episode_steps`** in `state_payload`.
- [x] Observations: partner **past actions** (last k joint steps) in `render_observation`.

## P1 — Demo / viz ✅

- [x] Step-grouped trajectory, simultaneous copy, dual highlight, parallel animation.

## P2 — Metrics

- [ ] Reconcile `help_value` / solver / `turn_budget: auto` with joint-step wall-clock.

---

## Next: LLM agents + RL pipeline (replace hardcoded scripts)

**Goal:** Prototype **collaborative two-agent rollouts** with **Azure-hosted small models** (budget ~$200–300 for env prototyping), then plug the same env into **AdAlignLLM** for real RL—not a permanent dependency on demo `JOINT_SCRIPTS` / `PartnerBrain`.

### Phase A — Env API for rollouts (no GPU, no trainer yet)

- [ ] **`MotiveRoomsEnv`** (thin wrapper): `reset()` → obs per agent; `step_joint(actions)` → obs, rewards, done, `info` with `episode_steps`-shaped log. Pure Python; no HTTP.
- [ ] **Joint-step contract** documented for callers: must pass one action per `active_agent`; document verb phases + tie-break (already in engine).
- [ ] **Observations for MARL:** extend `render_observation` with `PARTNER LAST STEP` (and optional last-k); include recent `say` lines; keep context ≤ ~4k tokens (README target).
- [ ] **Rewards + mode hooks (M3 slice):** `collaborative` / `mixed` / `competitive` payoff + reward transform on round end (even if round=1 for prototyping).
- [ ] **JSONL rollout logger:** append `{room_id, seed, mode, episode_steps[], metrics}` per episode for debugging and cheap analysis.

### Phase B — Azure API prototyping (replace scripted `honest` brain)

- [ ] **Secrets & config:** `.env` / env vars for Azure OpenAI (endpoint, deployment name, api version)—**never commit keys**; example `config/azure.example.env`.
- [ ] **`AzureChatPolicy`:** one completion per agent per joint step; system = framing + action grammar; user = observation; parse **last line** via existing `parse_action`.
- [ ] **Collaborative rollout driver:** each tick: build obs for A and B → **parallel** API calls → `step_joint({A, B})` → log tokens/cost estimate per step.
- [ ] **Model defaults:** small instruct models only (e.g. 4B–8B class deployments on Azure); temperature + max tokens capped for cost.
- [ ] **Cost guardrails:** session/episode token budget; dry-run mode (scripted fallback); simple `$` estimator in logs to stay under team cap.
- [ ] **CLI:** e.g. `scripts/rollout_llm.py --room shared_mat --rounds 1 --partner-a azure --partner-b azure` (and `--partner-b honest` for ablations).

### Phase C — Collaborative training setup (AdAlignLLM / RL)

- [ ] **`adapters/adalign/`** (per README M7): `Simulation` subclass wrapping `MotiveRoomsEnv` with **joint step** (not alternating `step` unless legacy split-wing).
- [ ] **`Agent` wrapper:** render MotiveRooms obs; parse action line; optional Azure or vLLM backend behind same interface.
- [ ] **`install_adapter.sh`** symlink into `third_party/AdAlignLLM/mllm/markov_games/motiverooms/`.
- [ ] **Hydra configs:** collaborative self-play 2-agent shared room; scripted partner baseline; single-agent curriculum entry.
- [ ] **Rollout semantics for RL:** clarify sample efficiency—**2 LLM calls per env step** (true MARL) vs **1 call + scripted partner** (cheaper eval); document when to use each.
- [ ] **Statistics hooks:** route escape, sharing, truthfulness metrics (README table) into AdAlign `statistics_runner` once rewards exist.

### Phase D — De-hardcode demo (optional bridge)

- [ ] Web UI: preset `llm_cooperation` calling rollout API or websocket (later); keep `JOINT_SCRIPTS` as **replay / golden trace** for CI only.
- [ ] Tests: mock LLM returns fixed action lines; one optional **smoke test** gated by `AZURE_OPENAI_*` env.

### Dependencies / blockers

- [ ] AdAlignLLM cloned + `run.py` smoke on GPU box (M0)—parallel track to Azure CPU/API prototyping.
- [ ] Human call: collaborative-only first vs all three modes in Phase 1 (README open question).
- [ ] Confirm Azure deployment names + which “small” models are approved for prototyping.

## Out of scope (v1)

- Async message buffers, mid-turn interrupts, action chunking, MAPF/collisions.
- Full room generator (M4) before first LLM rollouts complete on fixed `shared_mat`.
