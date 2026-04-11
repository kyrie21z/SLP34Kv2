# M08A-lite E1 Static KD Report

## Setup
- KD mode: confidence-tail static distillation (tail-only).
- lambda: 0.10
- tau: 2.0
- tail split: train_tail_split_v1_1.csv (120 / 27501).

## Key Results
- Baseline overall: 82.8733%
- E1 overall: 82.7716%
- Delta overall: -0.1017 pp
- hard delta: -0.0991 pp
- OOV delta: +0.1867 pp
- long_21+ delta: +0.1609 pp
- single delta: -0.2938 pp

## Error Ops
- replace: total 2337 -> 2337 (delta +0), mean/sample +0.000000
- insert: total 2275 -> 2253 (delta -22), mean/sample -0.003196
- delete: total 1861 -> 1857 (delta -4), mean/sample -0.000581

## Length Side Effects
- expansion ratio delta: -0.0872 pp
- trunc ratio delta: -0.0291 pp
