
# Internal Preflight Table

| File / module | Status | Keep / refactor / remove / new | Reason |
| --- | --- | --- | --- |
| `AGENTS.md` | Working policy file | KEEP | Public mission, evidence hierarchy, privacy, attribution, coding standards. |
| `README.md` | Useful but had portability and planned/completed boundary issues | REFACTOR | Remove local Python path, add PI-oriented structure, add demo and lesion status. |
| `pyproject.toml` | Working package metadata | KEEP | Python 3.10-compatible dependencies and test extras. |
| `src/pet_ai/data/manifest.py` | Working code | KEEP | Public-safe CSV validation. |
| `src/pet_ai/data/split_validation.py` | Working code | KEEP | Patient-level leakage validation. |
| `src/pet_ai/qc/geometry.py` | Working code | KEEP | NIfTI geometry comparison. |
| `src/pet_ai/qc/labels.py` | Working code | KEEP | Segmentation label QC. |
| `src/pet_ai/evaluation/segmentation.py` | Working code | KEEP | Voxel TP/FP/FN, Dice, FPV, FNV. |
| `src/pet_ai/evaluation/lesion_metrics.py` | Missing | NEW | Required lesion-level connected-component evaluation. |
| `src/pet_ai/reproducibility/*` | Working code | KEEP | SHA256 and run manifest helpers. |
| Existing scripts | Mostly working | KEEP/REFACTOR | Keep CLI wrappers; add demo; improve safety scanner and CI. |
| `docs/FINAL_VERIFICATION_REPORT.md` | Old report | REMOVE | Should not be copied as public release content. |
| `docs/PREFLIGHT_REPORT.md` | Old internal report | REMOVE | Superseded by external release report. |
| `docs/TECHNICAL_SUMMARY_CN.md` | Private learning-style content | REMOVE | Keep learning guide outside public repo. |
| `docs/README.md` | Duplicative | REMOVE | README and focused docs are clearer. |
| `docs/ARCHITECTURE.md` | Missing | NEW | Required public architecture explanation. |
| `docs/CODE_WALKTHROUGH.md` | Missing | NEW | Required public code walkthrough. |
| `.github/workflows/ci.yml` | Missing demo step | REFACTOR | CI must run ruff, pytest, safety scanner, and synthetic demo. |
| Historical sibling repository | Optional context | NOT_VERIFIED | No claims promoted without evidence. |
