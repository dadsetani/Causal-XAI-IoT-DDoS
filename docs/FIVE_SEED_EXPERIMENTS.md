# Five-seed experiment reproduction (major revision)

## Files and source fidelity

`notebooks/five_seed_reproducibility.ipynb` is the provided Kaggle notebook.
`src/five_seed_kaggle.py` is a direct export of its code cells 1–7 (the final
notebook cell only splits the generated ZIP into 20-MiB chunks for download).
The original single-seed implementation remains at
`src/kaggle_validated_experiments.py`.

## Data sources (Kaggle)

1. `rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files`
2. `dhoogla/cicbotiot` — third-party CICFlowMeter-derived Parquet representation

Attach both datasets to a Kaggle notebook. The code uses `/kaggle/input/...`
paths and writes to `/kaggle/working/validated_outputs/`.

## Execution

1. Start a fresh Kaggle runtime with both inputs attached; enable package
   installation access if dependencies are missing.
2. Open the provided notebook, check the input paths, and run cells in order.
   Alternatively execute the exported script in a matching Kaggle environment.
3. Keep `MSSQL_NROWS = None` for a full run; any integer is only for debugging.
4. Confirm `TRAINING_SEEDS = (42, 43, 44, 45, 46)`.
5. Allow both benchmark cells and the synthetic experiment to finish.
6. Execute the aggregation and archive cells to generate
   `causal_ddos_validated_outputs.zip`.
7. Compare `validated_all_results.csv` to the published summary; results can
   differ slightly across hardware/software and nondeterministic operations.

**Design:** Fixed stratified train/test split with seed 42; five training
seeds. Feature selection, scaling and class balancing are performed on
training data, not the untouched held-out test partition. These are *not*
five independent train/test splits.

## Output archive

The separately deposited ZIP contains a top-level `validated_outputs/`
directory, including per-seed Keras models, XGBoost JSON models, test
prediction NPZs, CSV metrics/histories, precision-recall figures, environment
metadata, synthetic outputs, and `SHA256SUMS.txt`.

Once the ZIP is downloaded from the v1.1.0 Zenodo record, verify it locally:

```bash
python tools/verify_zenodo_outputs.py causal_ddos_validated_outputs.zip
```

The provided standalone checksum manifest has 121 entries. The supplied
aggregate CSV matches its recorded SHA-256 value. The remaining archived
files have **not** been independently checked here because the complete ZIP
was not provided.

## Scope and caveats

The causal graph is an authored mechanism-informed graph, not an empirically
identified real-network causal structure. Real benchmarks are binary
attack-vs-benign and do not independently validate low-rate attack detection.
The synthetic causal validation is a separate controlled experiment.

For the actual package versions used in the five-seed run, consult
`environment_versions.json` inside the Zenodo output archive. The root
`requirements.txt` records the prior v1.0.0 environment and is **not**
asserted to be an exact lockfile for the five-seed run.
