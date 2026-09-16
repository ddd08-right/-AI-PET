
# Data Provenance

The public release contains schemas and synthetic tests only. It does not include DICOM, patient NIfTI files, clinical spreadsheets, segmentation masks from patients, model checkpoints, or raw private logs.

A public-safe manifest should use de-identified `case_id` and `patient_key` values. It should not contain names, dates of birth, accession numbers, medical record numbers, hospital numbers, addresses, phone numbers, local absolute paths, or raw DICOM metadata dumps.

Development-exposed data must not be described as an independent test set. If evidence is unavailable, use `NOT_VERIFIED`.
