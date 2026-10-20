# v3 partner pilot

## Fixed setup

The solo lock-method calibration achieved 5/5 escapes with v3 and one rejected
action in 53 submitted actions. The next pilot keeps that observation format,
gpt-5.4-mini, temperature 0.2, 300 output tokens, mixed rewards, 14-turn budgets,
private partner actions and simultaneous independent decisions.

`config/eval_partner_pilot.yaml` selects development seeds 0, 1 and 3 from
`parallel_branches`. Helpful and silent scripted partners are each paired with
the LLM in seats A and B: **12 episodes**. At most 168 LLM calls are needed if
every episode uses all turns. Session caps are 200 calls and 500,000 accounted/
reserved tokens. Token prices are unset, so dollar cost is not inferred.

## What changed in the scripted partners

Script version 1 advanced through its sequence even when the LLM took a required
item or a prerequisite had not opened. Version 2 rechecks actions against the
current visible room state, its own bag and its own read codes:

- Completed inspections and unlocks are skipped.
- An unavailable item is not assumed to be held; blocked unlocks are deferred.
- Other feasible actions in the dependency plan can proceed, allowing a switch
  to another supply branch.
- Code actions require that this script acquired the code; a hidden helper clue
  is not bypassed.
- Helpful scripts may request use of a missing shared item once they run out of
  feasible work. Silent scripts do not proactively communicate and wait when
  blocked.

Reference scripts keep their original fixed trajectories. Reactive scripts know
dependency plans, but do not inspect the partner's private bag, private codes or
current action. Two independent agents can still choose the same item/lock in a
tick; these real conflicts continue to cost turns. The code does not repair or
retry the LLM's action.

Script policy versions are stored in the episode metadata and shown in reports.
The new pilot changes both the observation format and the partner implementation
relative to the initial four-episode pilot. Compare conditions within the new
pilot; do not attribute the entire before/after difference to either change alone.

## Validation and interpretation

Offline checks cover stolen-item recovery in both seats, hidden code acquisition,
and two-script rollouts. Honest/honest and silent/silent pairs each escape all
60 frozen diagnostic rooms. This checks script execution; it is not a model result.

The actual Azure run must be read per seat as well as per condition. Inspect:

1. Escape, joint steps and LLM invalid/wrong attempts.
2. Who opened the locks/exits and whether the script carried a struggling LLM.
3. Item conflicts and stalls when one agent owns an item another planned to use.
4. Helpful versus silent performance within the same room and LLM seat.

All rooms here have together-escape, so each agent benefits when either opens the
exit. This pilot measures physical coordination and interference; it does not
establish strategic sharing, exploitation or incentive transfer. A later
private-handoff pilot should add a solo competence check on those split rooms,
then measure truthful sharing and actual use of partner-provided information.

## Recorded result

The Azure pilot completed in `runs/batch_9366695fc7a24103986a068bef092376`:

| Condition | Escapes | Mean steps | LLM unsuccessful actions |
|---|---:|---:|---:|
| Helpful script | 6/6 | 7.83 | 13/47 |
| Silent script | 6/6 | 9.17 | 16/55 |

All twelve trajectories replayed successfully. The batch used 102 provider
requests and 92,217 recorded tokens, with no API/budget failures. Across the
LLM's rejected actions, there were 15 same-tick item conflicts, six attempts to
unlock an already-open node, four attempts to take an already-taken item, and
four attempts to use an item it did not carry. There were no wrong code/item
attempts. These are coordination/state-tracking errors, not a recurrence of the
numeric-ID-as-code failure.

In all three silent-script trials with the LLM in seat B, the script opened all
five locks/exits. That pattern is consistent with the fixed A-first conflict
rule and the script's reliable solo plan; it is not evidence that the LLM chose
to free-ride. Helpful partners were faster in this small matched sample, but
team escape by itself does not demonstrate reciprocal cooperation.

The next prepared configuration is `config/eval_private_handoff_pilot.yaml`:
18 episodes spanning balanced, asymmetric and zero-help private-information
rooms, with solo/honest/silent conditions and both original wing assignments.
Keep v3 and model settings fixed. The split layout removes the shared-item
competition in this pilot and provides explicit information-sharing opportunities.
This follow-up is prepared but was not executed as part of the twelve-episode run.

## Statistics retained for subsequent runs

Aggregated reports now save the average, median, sample standard deviation and
min/max steps across finished tasks, plus successful-only averages. LLM-specific
escape timing/actions, unsuccessful actions, payoff, sharing opportunities and
rates are separate from whole-round duration. This matters in split rooms where
an LLM may escape early while its partner continues. Condition-level CSV/JSON
pool profiles for overview tables; profile-level CSV/JSON keep asymmetric and
zero-help controls separate. Unknown costs and undefined sharing rates remain
null instead of becoming misleading zeroes.
