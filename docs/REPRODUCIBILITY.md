
# Reproducibility

This release records reproducibility at the engineering level. It does not include patient data, model weights, or completed clinical validation.

## Public mechanisms

- CSV manifest schema for public-safe metadata.
- Patient-level split validation to prevent leakage across train/validation/test.
- SHA256 hashing for manifests, configs, checkpoints when available, and other artifacts.
- JSON run manifests with command, exit status, Git commit when available, Python/platform details, seed, optional GPU name, optional checkpoint hash, and notes.
- Synthetic CI checks that require no GPU and no external data downloads.

## Evidence boundaries

Missing files produce `None` hashes rather than invented values. Unavailable Git or GPU information is recorded as unavailable rather than inferred.
