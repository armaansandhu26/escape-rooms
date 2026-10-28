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

## Phase 2: two AI agents and explicit communication controls

- At the user's request, add a fixed-model collaboration pilot before trainer
  integration. Preserve the completed API-prototype baseline and report.
- Reuse strict simultaneous self-play, with one independent Azure client per
  character and the existing private v3 observations. No coaching, parse repair,
  shared reasoning history, RL training or new held-out evaluations.
- Communication defaults to enabled for compatibility. Disabled communication
  rejects chat, note writes/reads and item transfers in both engine step APIs,
  charging the normal turn cost. Public partner action history is forbidden in
  that condition. Visible room effects and partner escape status remain visible.
- Evaluate three development private-handoff rooms, including a no-benefit
  control, with two trials and alternating on/off order: 12 episodes / six pairs.
  Identical self-play seats are not redundantly evaluated twice.
- Summaries group by communication setting and preserve both agents' statistics.
  Pair checks compare actual logged room/configuration/source/model settings and
  exclude interrupted or mismatched comparisons. Offline on/off controls use
  known joint/solo solutions respectively and are explicitly labelled as scripts.
- Existing sharing attribution covers chat and item transfers. Note writes are
  counted separately; note-based resource use is not inferred as chat sharing.
  Channel-specific experiments and note attribution can follow if observed use
  makes them necessary. Paid execution remains an explicit `--execute` action.

## Communication-explained follow-up

- The original two-AI pilot completed 12 escapes and six matched pairs without
  acquiring partner-needed codes, sending messages or writing notes. This does
  not establish whether agents understood the communication affordance.
- At the user's request, add an opt-in `communication_instruction: explained`
  condition. Describe message recipient/timing/cost, optional information exchange,
  possible shortcuts, note costs and no actions after escape. Do not prescribe
  exact moves, reveal codes, force messages, or promise benefits in zero-help rooms.
- Keep original instructions as the default and preserve the original pilot
  config/logs. Use a separate follow-up config with the same 12-attempt matrix,
  model and rooms. Metadata, aggregation and pair checks distinguish instruction
  conditions. The added explanation combines mechanics and potential strategy
  benefit; it is an instruction treatment, not RL learning or proof of which
  wording caused any change.

## Native tool-call interface

- Make native function calls the default for new API rollouts and evaluations.
  Retain `action_interface: text` for the older last-line parser and stateless
  prompts; never silently fall back between protocols.
- Keep room tool contracts/argument validation separate from provider transport,
  policy conversation state and joint-step execution. Use strict static schemas
  without room-specific enums or oracle facts. Leave engine legality, rewards,
  phase order and turn costs unchanged.
- Preserve one call per active seat per tick. Collect choices before execution;
  deliver each seat only its own result after the tick. Invalid/missing/multiple
  calls spend one turn without executing a subset or retrying in strict mode.
- Retain native conversation items (including reasoning continuation items) in
  private per-episode policy history. Budget and log schemas/history, not just the
  latest observation. This changes memory as well as encoding relative to text
  mode, so treat it as a new experimental condition and recalibrate.
- Record interface/schema/call/result metadata; group reports and matched pairs
  by interface. Historical metadata without this field means text mode. Replay
  continues validating canonical engine actions without provider requests.
- Verify with mocked provider calls and the installed SDK's output models; live
  Azure deployment compatibility remains a separate execution check.

## Configurable connected-room episodes

- At the user's request, use shared physical rooms that both agents can explore,
  with independent positions. The existing `Room` remains the episode map; add
  optional locations/start positions and bidirectional passages. Preserve the
  old serialized form and replay snapshots when navigation is absent.
- Default research rollouts to a fresh seeded three-room map. Support any positive
  configured count and explicit seeds; retain `--room` for fixed scenarios. Keep
  frozen diagnostic generation and existing evaluation configs unchanged.
- Start with a chain topology so every room lies on the route to the final exit.
  Randomize actual card/container locations among already reachable rooms,
  furnishing descriptions, unique lock codes, code-versus-key locks and node
  order. Construct a feasible reference instead of requiring a large BFS.
- `move(room_id)` spends one turn and resolves after unlocks. All object actions
  are local to the current room; item transfers require co-location. Chat/notes
  retain their existing cross-room delivery rules. Connecting passages do not
  trigger escape; the existing final-exit together/race rules remain explicit.
- Keep seeds and reference metadata evaluator-only, record positions for replay,
  scale default turn/step budgets with reference length, and group reports by map
  size. Existing handoff annotations do not claim to measure all useful sharing
  in newly generated shared maps. Keep the old browser catalog unchanged.

## Adaptive orchestrator and ongoing supervision

- At the user's request, start new research rollouts with a control-only LLM
  orchestrator. It selects a bounded initial roster from public environment size
  through a native spawn_subagents call. Create exactly the requested number of
  independent policies. Keep the roster fixed after creation; continue coordinating
  it before every joint tick, as clarified by the user.
- Default to a configurable cap of eight, not a room-count-to-worker formula.
  Shared rooms support arbitrary identifier-based rosters; split wings retain
  their physical seats. Keep explicit fixed-team mode and historical suite configs
  for reproducibility. Connected configs default to orchestration.
- Send only size/rule summaries to the initial planner. Supervision receives
  public worker activity/turn status and explicit report actions, never raw worker
  observations, inventories, locations, tool results or reasoning. Reports cost
  one physical turn. Controller task updates cost provider usage but no worker turn.
- Broadcast say across the roster, add targeted give_to transfers for larger
  teams, and preserve co-location requirements. A single worker can report to its
  orchestrator. Require communication enabled for orchestration so control messages
  cannot bypass communication-off experiments.
- Charge planning, supervision and worker calls to one budget. Record setup errors
  and partial completed requests without inventing environment steps; do not retry
  or fall back to a heuristic on invalid controller output. Keep control calls out
  of worker payoff/action accounting and include them in total usage/runtime.
- Record roster, decisions and reports for replay and group reports by team mode,
  actual worker count and controller kind. Offline runs use a labelled scripted
  controller and feasibility references, not model decisions or evidence of learned
  orchestration quality. Keep engine/provider concerns separate.

## V2 follow-up to the first live orchestration run

- Keep the environment/model/seed/turn rules and original run/config unchanged.
  Add a separately versioned smoke config for the approved setup improvements.
- Give the controller actual starting arrangements and supported action mechanics;
  include public team chat on the next tick. Keep private observations, inventory,
  location and reasoning hidden. Treat reports as attributed claims, not role
  instructions or automatically verified facts, and do not discard them simply
  because the controller cannot inspect the room.
- Retain three complete recent native exchanges. Preserve deduplicated own
  successful inspect/unlock results and attributed messages, without generated
  summaries or oracle access. Retain at most 40 claims/messages per ledger. Keep
  v1 full-history behavior selectable and group protocol versions separately.
- Add explicit team/step progress output to the revised config. Do not force
  reports, correct model code associations, provide a scripted solving strategy,
  raise spending caps, or automatically launch a paid rerun.

## 2026-10-22: open exploration puzzle and orchestrator messaging

- Implement the user-approved mat/key, drawer/torch, battery, painting, sofa/token,
  display-case and ordered-fragment concept as a separate `exploration` source.
  Preserve the locked-chain generator and its configs for comparisons.
- Reuse inspect for searches. Add generic searchable objects, inspectable scenery,
  text clues and variable-length numeric codes; keep puzzle interpretation out of
  engine handlers. Powering the torch is an item-use dependency, not a general
  crafting system. Text clues remain private until explicitly communicated.
- Keep passages open from reset. Three rooms form a triangle; larger maps use an
  open ring. Extra rooms add a fragment each, while one/two-room configurations
  co-locate the same core puzzle. Seed the Storage/Lounge placement, numbers,
  assembly order and display order. The exit stays in the starting Office.
- Set `worker_broadcast: false` only in the new config. Remove and reject say in
  both native/text interfaces. Retain private reports to the controller, shared
  notes with explicit reads, item transfers and observable shared state changes.
  Do not coach workers to invent an alternative messaging strategy.
- Add note-read and item-transfer counts and a communication-channel report table.
  Group broadcast conditions and puzzle families separately. Channel counts do
  not demonstrate intent or causal cooperation; retain exact contents for review.
- Keep a constructive reference for solvability and replay. The existing numeric
  BFS oracle explicitly rejects text-clue puzzles rather than claiming a bound.
  Give the new puzzle reference-length turn allowance plus headroom for exploration
  and reporting (36 ticks at default seed); use an explicit 240-call/1M-token batch
  cap. This is a separate condition, not a matched change to the old smoke run.
- Validate offline and with fake native clients; leave live provider execution to
  the user-provided command.

## 2026-10-22: persistent progress tracking after the exploration failure

- Add a separate v3 condition in response to the failed 36-tick run, where workers
  reported a battery holder and a blocked torch but repeatedly searched scenery.
  Keep the puzzle and the previous config available unchanged for comparison.
- Have the controller maintain typed item, dependency, search and location entries
  through deltas in its existing coordinate call. Require citations to received
  reports/chat; retain original evidence and label interpretations unverified.
  Validate provenance/shape atomically without inspecting the true room state.
  Do not pretend that a valid citation proves the interpretation is correct.
- Add generic process guidance to distinguish blocked mechanisms from exhausted
  searches, connect item claims to dependencies, and assign concrete objectives.
  Do not hardcode the battery/torch solution, force reports, auto-repair task
  choices, or expose worker action results privately to the controller.
- Add optional worker-private action memory: exact own outcomes including failures,
  attempt counts over the retained window and the latest assignment. Cap it at 32
  distinct actions. Reports stay optional and cost a turn; notes/transfers remain
  available without special encouragement.
- Preserve seed 42, puzzle mechanics, model, turn allowance, team cap, broadcast
  restriction and session spending caps. Use one recent controller exchange plus
  the persistent board, and a 2048-token controller output ceiling for the larger
  structured response; retain worker history/output limits. Record and group the
  new condition explicitly. No paid rerun is launched automatically.

## 2026-10-22: isolate advisory memory errors from physical episode execution

- The first live v3 run stopped before tick 3 because the model described an item
  as available but named the reporting worker as its holder. Cross-field
  validation correctly rejected the board entry, but incorrectly aborted the
  entire episode despite valid worker assignments.
- Keep atomic validation. Reject a bad board delta, preserve existing memory,
  apply valid assignments, and return the validation message through the normal
  tool output and decision log. Do not silently rewrite claims, retry a provider
  call, or grant a free worker action. Invalid tool/assignment envelopes remain
  fatal policy errors. Clarify physical-carrier vs reporting-worker semantics in
  the schema and controller prompt.
- Save the actual failed tool payload as a regression fixture. Verify recovery,
  subsequent valid updates, full offline escape/replay, and exactly one regular
  coordination request per tick. Retain the failed live log as originally
  recorded; leave live reruns to the user.
