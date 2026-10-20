# Closing API incentive comparison

Run: `runs/batch_c67fd1a4575b4d61a2113183dadd2526`.
Config: `config/eval_incentive_comparison.yaml`.

All 24 episodes escaped, all trajectories replayed, and all 12 matched pairs
passed room/mechanics/partner consistency checks. The run has one source-code
hash across episodes. No API failures, budget stops, invalid actions or wrong
code/item attempts occurred. There were 74 provider requests and 58,399 recorded
tokens. Dollar cost remains unknown because token prices were not supplied.

| Incentives and matching framing | Episodes | Mean round steps | Mean LLM actions | LLM messages / notes | Mean total payoff |
|---|---:|---:|---:|---:|---:|
| Mixed | 12 | 6.00 | 2.75 | 0 / 0 | 29.92 |
| Collaborative | 12 | 6.00 | 3.42 | 0 / 0 | 29.25 |

Each row pools balanced, asymmetric and zero-help profiles. Both physical LLM
seats and two trial indices were run; mode order reverses on the second trial.
Actual reward transforms and matching prompts changed together. Model weights,
room files, observation version, policy settings and scripted partner stayed fixed.

## Behaviour

Both modes used received codes eight times and sent no messages or shared notes.
Mixed agents never acquired a partner-needed code. Collaborative agents acquired
one such code in one trial, but did not transmit it. Acquired-resource sharing
rates are therefore undefined for mixed (no opportunities), and 0/1 for
collaborative. Neither denominator supports a broad estimate of cooperation.

Collaborative agents sometimes did extra inspections or solo-route work before
using the script's code. This increased their own action count without speeding
up the partner. The scripted partner's remaining route dominated round duration,
which stayed unchanged. The small payoff difference is descriptive; this pilot
contains only one room per help profile and no held-out model evaluations.

## Objective sanity check

Offline execution of the existing mutual-help reference trajectories gives:

| Profile | Reference round steps | Reference total payoff |
|---|---:|---:|
| Balanced | 3 | 34 |
| Asymmetric | 3 | 38 |
| Zero-help | 2 | 28 |

Thus the room and scoring can reward reciprocal help: the reference beats the
observed team payoffs in the nonzero-help tasks, while the zero-help control has
no benefit. This does not imply that a particular RL algorithm will discover
that behaviour without appropriate exploration or a curriculum.

## Phase decision

Close the initial API-prototype phase and freeze these development baselines.
The environment executes deterministic tasks, observes private information,
scores incentive transforms, captures role-specific timing/sharing/usage, and
replays outcomes reliably. More prompting to force sharing would change the
question instead of resolving the observed null result.

The next phase is trainer integration and CPU/mock rollout validation, followed
by a GPU smoke test and actual RL updates when hardware is available. Keep held-
out model evaluation reserved, add multi-round memory/reveal only if needed for
the training question, and compare trained policies against the fixed-model
baselines. See `docs/api_phase_handoff.md` for the remaining scope.
