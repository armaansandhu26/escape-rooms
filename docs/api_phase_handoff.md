# API prototype phase: closing comparison and handoff

**Status: initial API prototype phase closed.** The 24-episode comparison in
`runs/batch_c67fd1a4575b4d61a2113183dadd2526` completed and all 12 matched pairs
passed replay and consistency checks. It showed no code sharing under either
mixed or collaborative incentives. The null result is retained as a baseline;
see `docs/incentive_comparison_results.md` for the metrics and phase decision.

## What this phase covers

The initial API prototype phase consists of room/mechanics calibration, clear
versioned observations, persistent/replayable measurements, scripted-partner
checks, solo/partner baselines, and the final inference-time incentive comparison.
The mixed/collaborative comparison plus a results review is the last planned
experiment in this phase. A failing check may justify a targeted repair; it does
not require adding an open-ended sequence of new experiments.

## Closing comparison

`config/eval_incentive_comparison.yaml` holds the three private-handoff room files,
v3 observations, gpt-5.4-mini, temperature 0.2, output cap 300, helpful reactive
script v2, mechanics, and turn costs fixed. Both LLM seats and two trial indices
are evaluated under mixed and collaborative rewards with matching prompt framing:
24 episodes, 12 matched pairs. The mode order reverses on the second trial.

The comparison changes numerical scoring and stated incentives together. It
measures inference-time behaviour of a fixed model; it is not evidence of an
effect caused by RL training. Trials are fresh API samples, not guaranteed
reproducible model RNG seeds. Partner scripts do not change between modes.

Read the per-profile and per-seat results before pooled averages. Compare:

- Code acquisition, actual sharing, and use of received codes.
- Whole-round steps versus the LLM's own actions and escape timing.
- Total payoff and per-agent payoff, not the arbitrary scale of rewards across
  different transforms.
- Balanced/asymmetric rooms against the zero-help control.
- Unsuccessful commands, API failures, truncations and completed matched-pair
  coverage. A sharing rate with no acquired opportunities is undefined.

## Phase-close review

Before freezing the prototype, confirm that logs replay, room hashes and engine
rules match across modes, rewards follow the documented transform, sharing
metrics agree with actual messages, and the report exposes failure denominators.
Keep both positive and null results. Do not tune away a lack of cooperation if
the model is executing the task correctly.

The frozen suite, v3 renderer, source hashes, configurations and baseline logs
then provide the reference point for the training phase. Preserve old logs and
version identifiers rather than rewriting their observations.

## What comes after this phase

The user subsequently chose a separate fixed-model collaboration phase before
training integration: two AI-controlled agents with communication on/off on
matched development rooms. This does not reopen the frozen prototype results.
See [phase 2](phase2_collaboration.md) for its implementation and pilot protocol.

The subsequent training stage begins with a CPU/mock rollout of the
trainer adapter and a small end-to-end smoke test when GPU access is available.
The current wrapper exposes joint outcomes, observations and terminal rewards;
the trainer-specific transition, termination/truncation and token/log-probability
contracts still need implementation and validation.

Later work includes actual RL updates, multi-round reveal/partner memory if
required by the research question, broader partner policies, arbitrary-graph
generation, and held-out evaluation. The present API results are development
pilots on a few fixed graph families, not held-out generalization or incentive-
transfer results. Completing this phase does not complete the MARL research
project or imply the model has learned a social strategy.
