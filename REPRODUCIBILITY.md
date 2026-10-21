# Reproducibility protocol

## Validated environment

| Package | Version |
|---|---:|
| Python runtime | Kaggle managed runtime |
| NumPy | 2.0.2 |
| pandas | 2.3.3 |
| TensorFlow | 2.20.0 |
| scikit-learn | 1.6.1 |
| XGBoost | 3.2.0 |
| DoWhy | 0.14 |
| PyArrow | 24.0.0 |

The random seed is 42 for NumPy, Python, TensorFlow, data splitting, training,
undersampling, XGBoost, and the synthetic random-common-cause refutation.

## Leakage-safe benchmark protocol

For each benchmark:

1. Normalize column names, convert the label to benign (`0`) or attack (`1`),
   and replace non-finite numerical values with zero.
2. Perform a stratified 80/20 train/test split before feature selection,
   scaling, or balancing.
3. Within each semantic group, select the highest-variance feature using the
   training partition only.
4. Fit `StandardScaler` on the selected training features only.
5. Randomly undersample only the training partition.
6. Reserve 10% of the balanced training data for validation.
7. Leave the test distribution untouched.

Selected semantic proxies:

| Node | MSSQL | CIC-BoT-IoT |
|---|---|---|
| Burst duration | Flow Duration | Flow Duration |
| TCP timeout | Fwd IAT Max | Bwd IAT Max |
| Queue degradation | Idle Max | Idle Max |
| Normal-traffic context | Flow IAT Max | Flow IAT Max |

## Models

The plain and causal MLPs use the same split, standardized inputs, dense
layers, and optimization settings:

- Dense(64, ReLU)
- Dropout(0.30)
- Dense(32, ReLU)
- Dense(1, sigmoid)
- Adam, learning rate 0.001
- Binary cross-entropy
- 20 epochs
- Batch size 64

The Causal-MLP inserts the residual propagation

```text
x_tilde = x + alpha * (x @ A_X)
```

before the shared MLP backbone. `alpha` is initialized at 0.1, constrained to
be nonnegative, and learned jointly with the classifier.

The Bot-IoT XGBoost reference uses 100 estimators, depth 6, learning rate 0.10,
and histogram-based tree construction.

## Timing protocol

- Warm-up prediction before timing
- Up to 100,000 held-out records
- Five repetitions
- Neural prediction batch size 1024
- Reported value: mean and standard deviation in milliseconds per flow

## Synthetic causal validation

The synthetic SCM contains `Burst_Duration`, `TCP_Timeout`, `Queue_Drop`,
`Normal_Traffic`, and `Attack_Label`. DoWhy identifies and estimates the unit
contrast for `Burst_Duration` from 0 to 1 with linear regression. A random
common cause is added in 100 refutation simulations.

## Integrity

Run the following command from `results/`:

```bash
sha256sum -c SHA256SUMS.txt
```

The manifest intentionally excludes itself.