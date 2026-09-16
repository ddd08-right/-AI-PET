# Repository mission

This repository demonstrates reproducible PET/CT AI engineering: data integrity, QC, model orchestration, debugging, evaluation and failure analysis.

# Evidence hierarchy

- COMPLETED_REAL_RUN
- ENGINEERING_SMOKE
- DEVELOPMENT_EXPOSED
- PLANNED
- NOT_VERIFIED

Never silently promote status.

# Patient safety

No patient images or PHI in Git.

# Attribution

AutoPET and nnU-Net architectures remain upstream work. This repository must not claim upstream model architectures, challenge code, or framework code as original work.

# Coding standards

Python 3.10-compatible where possible.

Use:
- type hints
- pathlib
- logging
- argparse
- dataclasses where useful
- small testable functions
- explicit exceptions
- deterministic seeds where relevant

Avoid:
- hard-coded absolute paths in source
- hidden state
- silent exception suppression
- broad except:
- magic numbers without explanation

# Research standards

- No patient-level leakage.
- No test-set tuning.
- No unsupported scientific conclusions.
- No hidden exclusion of failed cases.
