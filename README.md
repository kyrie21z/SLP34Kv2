# SLP34Kv2 Current Project README

This repository is currently being used as an active research workspace for `SLP34Kv2`, an ongoing Ship License Plate Recognition research project built on top of the SLP34K dataset and the AAAI-25 strong baseline.

Project name:

- current project name: `SLP34Kv2`
- dataset name: `SLP34K`
- current local repository directory: `SLP34K` (kept unchanged to avoid path-side effects)

The original public README from the AAAI-25 baseline release has been archived at [archive/README_AAAI25_original.md](/mnt/data/zyx/SLP34K/archive/README_AAAI25_original.md).

## Current Objective

The current mainline is not to rebuild the public baseline from scratch. The current goal is:

- keep the strong baseline as the reference line
- evaluate whether `+length-head` actually reduces length-related errors
- only enter Phase 2 (`EOS-aware decoding`) if Phase 1 evidence is strong enough

## Current Project Status

Completed:

- baseline training
- baseline `+length-head` training
- unified Phase 1 acceptance evaluation

Not yet started:

- EOS-aware decoding implementation
- EOS-aware decoding ablation

## Key Directories

- [ocr_training](/mnt/data/zyx/SLP34K/ocr_training): main OCR training and evaluation code
- [mae](/mnt/data/zyx/SLP34K/mae): MAE pretraining code
- [archive](/mnt/data/zyx/SLP34K/archive): archived historical documentation

## Important Entry Points

- training entry: [ocr_training/train.py](/mnt/data/zyx/SLP34K/ocr_training/train.py)
- standard evaluation entry: [ocr_training/test.py](/mnt/data/zyx/SLP34K/ocr_training/test.py)
- unified Phase 1 evaluation: [ocr_training/evaluation/evaluate_unified.py](/mnt/data/zyx/SLP34K/ocr_training/evaluation/evaluate_unified.py)
- Phase 1 report generation: [ocr_training/evaluation/generate_phase1_report.py](/mnt/data/zyx/SLP34K/ocr_training/evaluation/generate_phase1_report.py)

## Current Model Runs

Baseline:

- [2026-04-06_07-57-40_baseline](/mnt/data/zyx/SLP34K/ocr_training/outputs/new_oov/maevit_infonce_plm/2026-04-06_07-57-40_baseline)
- checkpoint: [last.ckpt](/mnt/data/zyx/SLP34K/ocr_training/outputs/new_oov/maevit_infonce_plm/2026-04-06_07-57-40_baseline/checkpoints/last.ckpt)

Baseline + length-head:

- [2026-04-05_22-13-13_baseline_length-head](/mnt/data/zyx/SLP34K/ocr_training/outputs/new_oov/maevit_infonce_plm/2026-04-05_22-13-13_baseline_length-head)
- checkpoint: [last.ckpt](/mnt/data/zyx/SLP34K/ocr_training/outputs/new_oov/maevit_infonce_plm/2026-04-05_22-13-13_baseline_length-head/checkpoints/last.ckpt)

## Phase 1 Acceptance Outputs

Unified evaluation outputs:

- baseline samples: [samples.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/phase1_baseline_eval/samples.csv)
- length-head samples: [samples.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/phase1_length_head_eval/samples.csv)

Acceptance report outputs:

- report: [phase1_report.md](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/phase1_acceptance/phase1_report.md)
- EOS analysis: [eos_analysis.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/phase1_acceptance/eos_analysis.csv)
- baseline per-sample export: [baseline_per_sample.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/phase1_acceptance/baseline_per_sample.csv)
- length-head per-sample export: [length_head_per_sample.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/phase1_acceptance/length_head_per_sample.csv)
- minimal EOS export: [eos_analysis_minimal.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/phase1_acceptance/eos_analysis_minimal.csv)

## Phase 1 Conclusion

Current evidence says:

- `length-head` is useful as a length prior
- `length-head` reduces length-related errors
- `length-head` does not improve overall OCR accuracy yet
- `EOS-aware decoding` is worth pursuing next

Important observations from the current Phase 1 report:

- baseline accuracy: `81.04%`
- `+length-head` accuracy: `80.90%`
- length error rate: `11.08% -> 10.59%`
- length-head exact length accuracy: `81.93%`
- length-head MAE: `0.770`
- long-sequence (`21+`) accuracy improves: `75.95% -> 77.07%`

The main priority for the next phase is:

- length-related errors
- `eos_early` / `eos_late`
- insert / delete
- long-sequence samples

## Environment Notes

The current server context used in this workspace:

- single GPU server
- GPU: `RTX PRO 6000 Blackwell 96GB`
- current runtime environment used for recent evaluation: `slpr_ocr`

## Training Notes

This repo now uses semantic output directory tags in Hydra. New runs will be written as:

- `ocr_training/outputs/<run_group>/<model>/<timestamp>_<run_tag>`

The relevant config is in [ocr_training/configs/main.yaml](/mnt/data/zyx/SLP34K/ocr_training/configs/main.yaml).

Example:

```bash
cd /mnt/data/zyx/SLP34K/ocr_training
python train.py model=maevit_infonce_plm dataset=SLP34K run_tag=baseline
python train.py model=maevit_infonce_plm dataset=SLP34K run_tag=baseline_length-head model.use_length_head=true
```

## Recommended Next Step

Do not add more training modules first.

The next recommended step is:

1. implement a minimal EOS-aware decoding prototype
2. evaluate it with the existing Phase 1 pipeline
3. compare it against the current baseline and `+length-head`

