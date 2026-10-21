# Changelog

## 1.1.0 — prepared for major-revision resubmission (not yet released)

- Added original five-seed Kaggle notebook and exported executable code cells.
- Added 30-row aggregate results (two datasets × three models × five seeds).
- Added SHA-256 manifest and local ZIP integrity verification tool.
- Added five-seed experiment documentation and Zenodo archive guidance.
- Preserved original v1.0.0 experiment code and historical outputs.
- Full trained models, predictions and plots remain outside Git, intended
  for a new version of the existing Zenodo record.

# Changelog

## 1.0.0 - 2026-08-21

- Replaced exploratory and failed notebook branches with a clean seven-cell
  Kaggle workflow.
- Removed hard-coded recovery results.
- Added train-only feature selection, scaling, and balancing.
- Preserved naturally imbalanced held-out test partitions.
- Implemented the actual four-node causal propagation matrix and trainable
  nonnegative alpha parameter.
- Added matched Plain-MLP and Causal-MLP comparisons.
- Added XGBoost as the Bot-IoT reference model.
- Added balanced accuracy, ROC-AUC, average precision, FPR, confusion counts,
  repeated inference timing, learned alpha, and gradient sensitivity outputs.
- Corrected Bot-IoT provenance to identify the evaluated Parquet file as a
  third-party CICFlowMeter-derived representation.
- Added exact package versions, metadata, trained models, histories, and
  SHA-256 checksums.
- Updated the manuscript to use the validated results and limitations.