# Private-handoff pilot results

Run: `runs/batch_1f065035d2cd4f0eb9f69aac4b3e3bf6`.
Config: `config/eval_private_handoff_pilot.yaml`.

All 18 episodes escaped and replayed correctly. There were no API/budget failures
and no invalid or wrong LLM actions. The run used 73 provider requests and 57,932
recorded tokens. Token prices were unset, so no dollar cost is inferred.

| Setup | Episodes | Average whole-round steps | Average LLM actions |
|---|---:|---:|---:|
| Solo | 6 | 4.67 | 4.67 |
| Helpful scripted partner | 6 | 6.00 | 2.83 |
| Silent scripted partner | 6 | 5.33 | 4.67 |

These means pool balanced, asymmetric and zero-help profiles (two seat assignments
per profile). Per-profile means, successful-only means, medians, sample standard
deviations, min/max, sharing, usage and timing are retained in the run's aggregate
CSV/JSON files. The spreads describe this small heterogeneous sample, not trained
policy uncertainty. All tasks succeeded, so finished-run and successful-only
duration means coincide here.

## What happened

- On the balanced room, solo/silent LLMs used six actions. With a helpful partner,
  the LLM used its received exit code and escaped in three actions. The helpful
  script then needed eight actions to finish its own exit.
- On the asymmetric room, a helpful script reduced the LLM's actions from eight
  to four in wing A, and from four to three in wing B. Whole-round times were six
  and ten steps respectively, depending on which partner remained inside.
- In the zero-help control, every condition took two LLM actions and two round
  steps. Adding a partner provided no shortcut, as intended.
- LLMs sent no messages in any episode and used received resources four times.
  They did not acquire the partner-needed codes. Acquired-resource sharing
  opportunities are therefore zero and sharing rates undefined, rather than 0%.

## Interpretation

The model handles this small set of split-room mechanics and uses useful partner
information. The helpful script improves the LLM's own escape time, but there is
no observed reciprocal sharing. The whole round can take longer because the
script pays the helping cost and then follows its slower independent route.
This explains why whole-round averages must not be used as a substitute for the
evaluated agent's own action/escape timing.

These are mixed-reward episodes: each agent is scored on its own payoff. The
observed behaviour is consistent with exploiting an available shortcut, but the
pilot does not identify the model's intent or demonstrate a learned strategy.
A useful next comparison would hold the room, v3 observations and scripted partner
fixed while changing actual reward mode and matching framing between mixed and
collaborative, to check whether team scoring changes code acquisition/sharing.
The present pilot contains one room per help profile and one trial per seat.
