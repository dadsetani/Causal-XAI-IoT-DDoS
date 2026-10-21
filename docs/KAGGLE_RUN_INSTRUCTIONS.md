# Running the validated Kaggle cells

Use `kaggle_validated_experiments.py` instead of continuing the exploratory
cells in the older notebook. The script is divided into six blocks marked with
`# %%`; copy each block into one Kaggle code cell and run them in order.

## Kaggle inputs

Add these two Kaggle datasets to the notebook:

- `rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files`
- `dhoogla/cicbotiot`

The second input is a published, third-party CICFlowMeter-feature variant of
Bot-IoT. The validated cells do not claim that the paper authors re-extracted
the original PCAP files.

## Run procedure

1. Start a fresh Kaggle session and select a GPU accelerator.
2. Enable Internet for Cell 1 only if DoWhy or another dependency is missing.
3. Copy and run Cells 1--6 in their existing order.
4. For a quick pipeline check, temporarily set `MSSQL_NROWS = 500_000` in
   Cell 1. Do not use that run in the paper.
5. For the final run, restore `MSSQL_NROWS = None`, restart the session, and
   run all six cells again.
6. Do not execute the old recovery, synthetic-benign, GRU, or failed cells in
   the same session.
7. Download the complete `/kaggle/working/validated_outputs` directory.

## Files to return to this project

- `validated_synthetic_results.csv`
- `validated_synthetic_data.csv`
- `validated_all_results.csv`
- `CIC_DDoS2019_MSSQL_results.csv`
- `CIC_DDoS2019_MSSQL_metadata.json`
- `CIC_DDoS2019_MSSQL_causal_gradient_importance.csv`
- `CIC_Bot_IoT_results.csv`
- `CIC_Bot_IoT_metadata.json`
- `CIC_Bot_IoT_causal_gradient_importance.csv`
- all four MLP training-history CSV files

Only values produced by these validated files should replace the current
numbers in `main.tex`.

## Safeguards implemented in the cells

- train/test splitting occurs before fitted preprocessing;
- one proxy per semantic node is selected using training variance only;
- the scaler is fitted on training data only;
- under-sampling is restricted to the training partition;
- validation is stratified and shared by Plain-MLP and Causal-MLP;
- the natural test distribution is neither balanced nor used for fitting;
- the causal model uses the stated four-node adjacency matrix and a Keras
  `add_weight()` coefficient;
- accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR,
  confusion counts, timing mean/std, learned alpha, gradients, histories, and
  preprocessing metadata are exported.