
# Failure Analysis

This document keeps only evidence-supported failure categories and avoids claiming new reruns.

## Evidence-supported historical records

| Failure | Category | Evidence status | What is known | What is not claimed |
| --- | --- | --- | --- | --- |
| Windows WinError 1455 during deep-learning work | Resource failure | HISTORICAL_PROJECT_RECORD | Historical local project records indicate a Windows paging/host-memory resource issue. | No new rerun, benchmark, or clinical conclusion is claimed. |
| Low-VRAM probe | Resource failure | HISTORICAL_PROJECT_RECORD | Historical local records indicate resource-aware nnU-Net experimentation/probing. | No new GPU measurement or training outcome is claimed in this release. |

## Failure categories

- Software bug: code logic, parsing, metric calculation, IO behavior, or test coverage failure.
- Environment failure: missing packages, incompatible Python/CUDA versions, broken command-line tools, or unset environment variables.
- Resource failure: host RAM, GPU VRAM, disk, paging, or runtime limits.
- Scientific/model failure: poor generalization, missed lesions, false positives, calibration errors, OOD behavior, or clinically meaningful metric degradation.

These categories matter because the response is different. A software bug needs code and tests. An environment failure needs reproducible setup documentation. A resource failure needs configuration and hardware constraints. A scientific/model failure needs controlled evaluation without leakage or test-set tuning.

## Public-release boundary

The public repository contains synthetic tests and engineering smoke checks only. Historical failures are documented as historical records, not as newly reproduced evidence.
