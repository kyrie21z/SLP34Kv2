# SLP34Kv2

Ship License Plate Recognition research workspace based on SLP34K.

## Overview

This repository tracks our post-AAAI25 iterative research on SLP recognition, including baseline reproduction, length-prior validation, and decoder paradigm feasibility probes.

- Base dataset: `SLP34K`
- Original AAAI25 README: [`archive/README_AAAI25_original.md`](archive/README_AAAI25_original.md)
- Internal progress README: [`README.md`](README.md)

## Latest Status (2026-04-11)

### 1) Phase1 Reproduction (`baseline` vs `+length-head`)

- Baseline accuracy: `82.83%` (5702/6884)
- `+length-head` accuracy: `82.87%` (5705/6884)
- Length-error rate: `9.69% -> 9.75%`

Takeaway: length head is a usable signal, but gains are limited under current decoding.

### 2) M05: Replace-Confusion Probe

Takeaway: high-value error concentration is character-level replacement confusion (especially in `single / hard / OOV / long_21+`).

### 3) M06: Substitution-aware / Joint Training

Takeaway: no net benefit on unified full set; overall and key subsets regressed. Archived as non-mainline.

### 4) M07: Decoder Paradigm Reroute (CTC Probe)

Takeaway: stability can be partially improved (S4), but recognition remains unusable (`overall=0.0%`). M07 is closed.

## Key Artifacts

- Phase1 report: [`ocr_training/evaluation/results/phase1_acceptance_repro_v3/phase1_report.md`](ocr_training/evaluation/results/phase1_acceptance_repro_v3/phase1_report.md)
- M05 report: [`ocr_training/reports/M05/m05_replace_visual_confusion_probe.md`](ocr_training/reports/M05/m05_replace_visual_confusion_probe.md)
- M06 receipt: [`ocr_training/reports/M06/m06_stage_receipt_for_master.md`](ocr_training/reports/M06/m06_stage_receipt_for_master.md)
- M07 report: [`ocr_training/reports/M07/m07_decoder_paradigm_reroute.md`](ocr_training/reports/M07/m07_decoder_paradigm_reroute.md)
- M07 S4 report: [`ocr_training/reports/M07/m07_s4_ctc_stability_probe.md`](ocr_training/reports/M07/m07_s4_ctc_stability_probe.md)

## Code Entry Points

- Train: [`ocr_training/train.py`](ocr_training/train.py)
- Test: [`ocr_training/test.py`](ocr_training/test.py)
- Unified evaluation: [`ocr_training/evaluation/evaluate_unified.py`](ocr_training/evaluation/evaluate_unified.py)

## Next Direction

Current recommendation is to move from M07 closure to the next module (`M08`) rather than continue CTC-probe-line budget expansion.
