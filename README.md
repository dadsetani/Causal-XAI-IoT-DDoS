# Causal-Aware Explainable DDoS Detection for Knowledge-Defined IoT Network Management

## Major-revision reproducibility update (proposed v1.1.0)

This repository preserves the original v1.0.0 experiment implementation in
`src/kaggle_validated_experiments.py` and adds the five-training-seed experiments
performed in Kaggle in response to peer-review feedback. The original v1.0.0
release remains available in GitHub and Zenodo.

**Status (21 October 2026):** The five-seed validated experimental outputs have been published in Zenodo as version v1.1.0. The complete experimental archive is provided in six sequential parts, accompanied by reconstruction instructions and an MD5 integrity check.

- **Zenodo v1.1.0 (validated outputs):** https://doi.org/10.5281/zenodo.23441576
- **Zenodo concept DOI (all versions):** https://doi.org/10.5281/zenodo.22280769
- **Original Zenodo release (v1.0.0):** https://doi.org/10.5281/zenodo.22280770

The GitHub repository contains the source code, experiment scripts, reproducibility documentation, and summarized results. The complete experimental artifacts are archived in Zenodo.

### Reproducibility resources

- Original archived release (v1.0.0): https://doi.org/10.5281/zenodo.22280770
- All-version (concept) DOI: https://doi.org/10.5281/zenodo.22280769
- Five-seed notebook: [`notebooks/five_seed_reproducibility.ipynb`](notebooks/five_seed_reproducibility.ipynb)
- Five-seed Kaggle script: [`src/five_seed_kaggle.py`](src/five_seed_kaggle.py)
- Five-seed protocol: [`docs/FIVE_SEED_EXPERIMENTS.md`](docs/FIVE_SEED_EXPERIMENTS.md)
- New aggregate results: [`results/five_seed_validated_all_results.csv`](results/five_seed_validated_all_results.csv)
- SHA-256 manifest for full output ZIP: [`results/five_seed_SHA256SUMS.txt`](results/five_seed_SHA256SUMS.txt)

The full `causal_ddos_validated_outputs.zip` archive (108,576,782 bytes,
containing `validated_outputs/` with trained models, predictions, figures,
per-seed results and metadata) is intentionally **not tracked by Git**.
The planned distribution location is a new **version of the existing Zenodo
record**, retaining the concept DOI above. The exact v1.1.0 version DOI and
archive download link will be added after Zenodo publication. This README
must be updated before final journal resubmission to avoid a pending link.

### Five-seed results (means across training seeds 42–46)

| Dataset | Model | Runs | Mean attack F1 | Mean false-positive rate |
|---|---|---:|---:|---:|
| CIC_DDoS2019_MSSQL | Plain-MLP | 5 | 0.999521 | 0.313059 |
| CIC_DDoS2019_MSSQL | Causal-MLP | 5 | 0.999521 | 0.313059 |
| CIC_DDoS2019_MSSQL | XGBoost | 5 | 0.995419 | 0.169946 |
| CIC_Bot_IoT | Plain-MLP | 5 | 0.967140 | 0.020528 |
| CIC_Bot_IoT | Causal-MLP | 5 | 0.967003 | 0.020476 |
| CIC_Bot_IoT | XGBoost | 5 | 0.982754 | 0.012619 |

These are arithmetic means calculated from the supplied
`validated_all_results.csv`; they have **not** been independently regenerated
by rerunning Kaggle. See the CSV for all 30 rows and other metrics.

### Important experimental boundaries

- Seeds 42–46 change model training; data partitioning uses a fixed seed 42.
- The two evaluated sources are CIC-DDoS2019 MSSQL and the third-party
  CICFlowMeter-derived `dhoogla/cicbotiot` Parquet representation.
- Neither real benchmark has a dedicated low-rate DDoS ground-truth label.
- The controlled synthetic causal experiment is separate from the real-data
  benchmarks; it does not establish real-world causal identification.
- Report attack-class F1 together with benign-class metrics, FPR and class
  counts because the test distributions can be strongly imbalanced.

## Original v1.0.0 documentation

The following section preserves the original README for historical context;
its single-seed result table refers to **v1.0.0**, not the new five-seed study.

---

# Causal-Aware Explainable DDoS Detection for Knowledge-Defined IoT Network Management

This repository contains the validated implementation and reproducibility
artifacts for a causal-aware flow-level DDoS detection framework. The method
maps traffic statistics to four semantic nodes---burst duration, TCP timeout,
queue degradation, and normal-traffic context---and applies a learnable causal
propagation layer before binary classification.

The Low-Rate DDoS mechanism motivates the authored causal graph. The real-data
benchmarks are broader flow-level attack datasets and do not contain a
dedicated Low-Rate DDoS ground-truth label.

## Validated results

All values below were regenerated in a clean Kaggle session on August 21,
2026. Feature selection, scaling, and class balancing were fitted on training
data only. Held-out tests retained their natural class distributions.

| Dataset | Model | Accuracy | Balanced accuracy | F1 | ROC-AUC | FPR | Mean latency (ms/flow) | Learned alpha |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CIC-DDoS2019 MSSQL | Plain MLP | 0.9990 | 0.8431 | 0.9995 | 0.9071 | 0.3131 | 0.002833 | -- |
| CIC-DDoS2019 MSSQL | Causal-MLP | 0.9990 | 0.8431 | 0.9995 | 0.9069 | 0.3131 | 0.002857 | 0.2323 |
| CIC-BoT-IoT | Plain MLP | 0.9391 | 0.9591 | 0.9685 | 0.9788 | 0.0208 | 0.002885 | -- |
| CIC-BoT-IoT | Causal-MLP | 0.9389 | 0.9592 | 0.9684 | 0.9787 | 0.0204 | 0.002913 | 0.0371 |
| CIC-BoT-IoT | XGBoost | 0.9631 | 0.9727 | 0.9812 | 0.9947 | 0.0177 | 0.000892 | -- |

The controlled synthetic experiment produced an ATE of `0.1409503`, a
random-common-cause refuted estimate of `0.1409490`, and a random-forest
attack-class F1-score of `0.9224490`.

Accuracy and attack-class F1 are prevalence-sensitive in these highly
attack-dominant datasets. Balanced accuracy, ROC-AUC, FPR, confusion counts,
and class distributions are provided in `results/` and should be reported
alongside accuracy.

## Repository structure

```text
.
├── src/
│   └── kaggle_validated_experiments.py
├── results/
│   ├── validated_all_results.csv
│   ├── validated_synthetic_results.csv
│   ├── *_metadata.json
│   ├── *_history.csv
│   ├── *.keras
│   └── SHA256SUMS.txt
├── manuscript/
│   ├── main.tex
│   └── references.bib
├── figures/
├── docs/
├── REPRODUCIBILITY.md
└── requirements.txt
```

## Datasets

Add these inputs to a Kaggle notebook:

1. `rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files`
2. `dhoogla/cicbotiot`

The second input is a third-party CICFlowMeter-derived representation of the
original Bot-IoT dataset. This repository does not claim that the authors
re-extracted the original PCAP files.

## Reproduction

1. Start a fresh Kaggle session and add both datasets above.
2. Copy the seven `# %%` blocks in
   `src/kaggle_validated_experiments.py` into separate notebook cells.
3. Run the cells from top to bottom with `MSSQL_NROWS = None`.
4. Use only `validated_synthetic_results.csv` and
   `validated_all_results.csv` for manuscript values.
5. Verify downloaded artifacts:

```bash
cd results
sha256sum -c SHA256SUMS.txt
```

See `REPRODUCIBILITY.md` and `docs/` for the complete protocol.

## Scientific scope and limitations

- The causal effect and refutation apply to the controlled synthetic SCM.
- The benchmark experiments assess within-dataset prediction and do not prove
  universal causal validity.
- Neither real-data benchmark supplies a dedicated Low-Rate DDoS label.
- Causal propagation does not consistently outperform the plain neural model.
- The MSSQL false-positive rate remains high because only 559 benign flows are
  present in its untouched test partition.
- Inference timings characterize the active Kaggle runtime, not a deployed SDN
  controller.

## Citation and archival release

The validated five-seed experimental outputs accompanying the manuscript's major revision are permanently archived in Zenodo:

**Version-specific DOI (v1.1.0):** https://doi.org/10.5281/zenodo.23441576

**Concept DOI (all versions):** https://doi.org/10.5281/zenodo.22280769

For reproducibility of the revised experiments, cite the version-specific DOI. For a reference that resolves to the latest available version, use the concept DOI.

## License

## License

The software source code in this repository is distributed under the MIT License. See the [LICENSE](LICENSE) file for the full license text.

Copyright (c) 2026 Bahareh Mohammadi.

The associated Zenodo record is published under Creative Commons Attribution 4.0 International (CC BY 4.0). Users should consult the licensing information associated with each resource and preserve the applicable copyright and attribution notices.
