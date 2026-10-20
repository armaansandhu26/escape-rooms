# Implementation decisions

This file records choices where the design specification leaves room for more
than one reasonable implementation.

## Engine and action validity

- `RoomEngine.step_joint` applies simultaneous submissions in verb phases with
  A-first conflict resolution. Sequential `step` remains available for legacy
  callers. Research rollouts collect independent decisions before stepping.
- A syntactically valid attempt with the wrong code or wrong item has
  `valid: true` and result `wrong`. `valid: false` is reserved for malformed or
  disallowed actions. Both outcomes consume one turn.
- Keys and other requirement items are reusable after `use`. The current room
  format gives each item a unique location, so this does not create duplicate
  handoffs.

## Solver

- Cooperative path length gives the agent every requirement sourced from the
  partner's wing at turn zero, at no action cost. Communication and handoff
  costs belong to the helping agent and are intentionally excluded.
- The solver returns both the path length and an executable shortest path. This
  makes generator failures and future calibration easier to inspect.

## Research protocol (2026-10-19)

- Prompt framing, numerical reward mode, and shared-exit mechanics are separate
  controls. `MotiveRoomsEnv.framing` is retained as compatibility metadata and
  never changes transitions or rewards. The runner passes framing to policies.
- Terminal payoffs follow the README: `10 + turns_left` on escape, otherwise zero.
  Rewards use the collaborative sum, own payoff, or competitive difference. No
  shaping, multi-round reveal, or trainer transition interface is added here.
- Strict rollouts default to simultaneous decisions, private partner actions, no
  strategy coaching, and no parse/engine repair. Invalid model actions cost turns.
  Existing parser normalization (spaces/hyphens in identifiers) remains shared
  by every condition. Assisted rollouts are an explicit separate baseline.
- Public action history exposes code arguments. Private action history hides all
  partner action entries, while chat, notes, visible state changes and partner
  status retain their existing mechanics. Communication-channel restrictions are
  separate future work.
- Seat counterbalancing swaps stable policy identities and their model/framing
  between fixed engine seats; it does not alter physical room roles or tie-breaks.
- Together-escape with equal payoffs yields zero competitive reward. Split-wing
  rooms are the default recommendation for incentive comparisons. Escape rules
  must be held fixed within a comparison.

## Diagnostic suite and records (2026-10-19)

- Three diagnostic families cover parallel branches, private code handoffs and
  costly item help. Split-room profiles include asymmetric dependence and zero
  useful help. Ten seeds per family per split produce 60 frozen YAMLs. Held-out
  nonzero tasks add dependency depth and locked helper branches. This is a small
  diagnostic generator, not the full arbitrary-graph generator from M4.
- Every room is schema-checked and its solo and joint reference plans execute in
  the real engine. Shared-room solo solving now uses all shared nodes. Free-help
  path lengths remain optimistic own-action bounds, not joint-step optima.
- JSONL is an event stream: start, decisions, provider calls, joint transitions,
  end, metrics. It is flushed as events arrive and opened exclusively to avoid
  overwriting evidence. Logs embed the room, source hashes and experiment settings;
  replay does not require another API call or the original room file.
- Provider calls record usage and latency, with SDK retries disabled. Empty final
  text is a model-output failure that consumes a strict turn. Transport failures
  stop the episode separately. No credentials or exception messages enter logs.
- Session budgets reserve estimates before concurrent calls and charge unknown
  requests conservatively. Prices must be supplied; absent prices yield null
  costs. Token/dollar reservations are estimates rather than billing guarantees.
- Scripted partners know dependency plans. Their inspect-before-code sequence is
  an instrumentation baseline. Code deception is meaningful in code-handoff
  rooms, not item-only tasks. Solo trials project the reachable wing and retain
  its solo route; physical seat is retained in metadata.
- The batch evaluator previews by default. Offline execution validates reference
  trajectories; explicit --execute is required for Azure evaluations. Summaries
  separate incomplete/error runs from completed-task denominators and bootstrap
  room clusters instead of treating every seat/trial as an independent room.

## Observation calibration

- The first real pilot had 11 clue-take errors, 15 repeated/unnecessary unlocks
  of open nodes, and 3 item conflicts in 50 LLM actions. This motivates an
  observations-only calibration; it does not establish rendering as the sole cause.
- Research environments default to v2 typed observations. v1 remains available,
  and replay treats historical logs without a version field as v1. Engine state,
  incentives, turn costs and the policy system prompt stay fixed across formats.
- A paired solo configuration evaluates v1/v2 on the same five development seeds
  with the pilot deployment and sampling settings. Reports group by format to
  avoid pooling the treatment and control. No paid calibration run was launched
  as part of preparing these changes.

## Lock-method clarification

- The real v1/v2 calibration improved escape from 0/5 to 3/5, but v2 still had
  ten valid-but-wrong attempts in addition to three rejected commands. ID suffixes
  were used as codes at item locks in the failed runs.
- v3 adds lock-type/verb templates and the distinction between identifiers and
  revealed code values, without changing the policy system prompt, graph or engine.
  Historical v1/v2 rendering is preserved. New research environments default to v3.
- Reports show invalid, wrong and combined unsuccessful actions, with arbitrary
  pairwise observation-version coverage. The next config compares v2/v3 on the
  same five development rooms; no paid calls are launched during preparation.

## Report duration and sharing statistics

- Aggregate reports retain mean, median, sample standard deviation and min/max
  joint steps for finished tasks, plus successful-only means. Turn-budget failures
  remain in finished-task means; errors and truncations are separate counts.
- Individual escape steps and own action counts remain distinct from round end.
  This is material in the private-handoff pilot: a model can escape after three
  actions while the helping script finishes at step eight.
- Condition/profile CSV and JSON files include role-specific action quality,
  payoff, sharing, latency and provider usage. Usage totals include observed work
  from interrupted episodes. Missing prices/usage and zero sharing denominators
  remain null rather than being silently converted to zero.

## Closing the initial API prototype phase

- The final comparison interleaves mixed/collaborative reward transforms and
  matching prompt framing on fixed private-handoff tasks, helpful scripts, both
  seats and two trials (24 episodes / 12 pairs). These are inference-time
  treatments, not RL-trained policy differences. Reports show mode explicitly.
- The run completed with all escapes and no malformed/wrong actions. Neither
  treatment sent codes or shared notes. Reference mutual-help trajectories still
  achieve higher team payoffs on nonzero-help tasks, so the objective remains
  capable of rewarding cooperation despite the fixed model's null result.
- Freeze development baselines and move to trainer adapter/rollout validation.
  Do not keep changing prompts merely to obtain a positive sharing result.
  Actual RL, held-out model evaluation and any required multi-round features
  remain work for the training phase, not completed claims of this prototype.
