"""Five-seed Kaggle experiment export from notebooks/five_seed_reproducibility.ipynb.

Run in a Kaggle environment with both datasets attached. The Python source is
exported verbatim from notebook code cells 1–7 (excluding optional ZIP splitter).
See docs/FIVE_SEED_EXPERIMENTS.md for the full reproduction protocol.
"""

# %% [Cell 1] Environment, dependencies, imports, and experiment constants
import importlib.util
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


PACKAGE_MAP = {
    "dowhy": "dowhy",
    "tensorflow": "tensorflow",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "pyarrow": "pyarrow",
}
missing_packages = [
    package
    for module, package in PACKAGE_MAP.items()
    if importlib.util.find_spec(module) is None
]
if missing_packages:
    print("Installing missing packages:", missing_packages)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing_packages]
    )

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import matplotlib.pyplot as plt
from dowhy import CausalModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
TRAINING_SEEDS = (42, 43, 44, 45, 46)
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TIMING_REPEATS = 5
TIMING_MAX_SAMPLES = 100_000

# Set to an integer for a quick debugging run; keep None for the final run.
MSSQL_NROWS = None

MSSQL_ROOT = Path(
    "/kaggle/input/datasets/rodrigorosasilva/"
    "cic-ddos2019-30gb-full-dataset-csv-files"
)
BOT_PATH = Path(
    "/kaggle/input/datasets/dhoogla/cicbotiot/CIC-BoT-IoT-V2.parquet"
)
OUTPUT_DIR = Path("/kaggle/working/validated_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NODE_ORDER = [
    "Burst_Duration",
    "TCP_Timeout",
    "Queue_Drop",
    "Normal_Traffic",
]

FEATURE_GROUPS = {
    "Burst_Duration": [
        "Flow Duration",
        "Fwd IAT Total",
        "Bwd IAT Total",
    ],
    "TCP_Timeout": [
        "Bwd IAT Max",
        "Fwd IAT Max",
    ],
    "Queue_Drop": [
        "Idle Max",
        "Idle Min",
        "Idle Mean",
        "Bwd IAT Mean",
    ],
    "Normal_Traffic": [
        "Flow IAT Max",
    ],
}
CANDIDATE_FEATURES = [
    feature
    for node in NODE_ORDER
    for feature in FEATURE_GROUPS[node]
]

# Row-vector convention: A[j, k] = 1 means node j -> node k.
# Burst -> TCP -> Queue and Normal-traffic context -> Queue.
A_X = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def set_global_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


set_global_seed()
print("TensorFlow:", tf.__version__)
print("Pandas:", pd.__version__)
print("Output directory:", OUTPUT_DIR)

ENVIRONMENT_VERSIONS = {
    package: importlib.metadata.version(package)
    for package in [
        "numpy",
        "pandas",
        "tensorflow",
        "scikit-learn",
        "xgboost",
        "dowhy",
        "pyarrow",
    ]
}
with open(OUTPUT_DIR / "environment_versions.json", "w", encoding="utf-8") as handle:
    json.dump(ENVIRONMENT_VERSIONS, handle, indent=2)
print("Environment versions:", ENVIRONMENT_VERSIONS)


# %% [Cell 2] Reusable preprocessing, models, evaluation, and artifact helpers
@tf.keras.utils.register_keras_serializable(package="CausalDDoS")
class CausalPropagation(tf.keras.layers.Layer):
    """Residual propagation x_tilde = x + alpha * (x @ A_X)."""

    def __init__(self, adjacency, **kwargs):
        super().__init__(**kwargs)
        self.adjacency_array = np.asarray(adjacency, dtype=np.float32)
        self.adjacency = tf.constant(self.adjacency_array, dtype=tf.float32)

    def build(self, input_shape):
        expected_width = self.adjacency_array.shape[0]
        if int(input_shape[-1]) != expected_width:
            raise ValueError(
                f"Expected {expected_width} semantic inputs, got {input_shape[-1]}"
            )
        self.alpha = self.add_weight(
            name="causal_alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.alpha * tf.linalg.matmul(inputs, self.adjacency)

    def get_config(self):
        config = super().get_config()
        config.update({"adjacency": self.adjacency_array.tolist()})
        return config


def normalize_label(series):
    """Return 0 for benign and 1 for attack for numeric or text labels."""
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(0)
        unique = sorted(pd.unique(numeric))
        if set(unique).issubset({0, 1}):
            return numeric.astype(np.int8)

    text = series.astype(str).str.strip().str.lower()
    benign_tokens = {"0", "benign", "normal", "normal traffic"}
    return (~text.isin(benign_tokens)).astype(np.int8)


def clean_feature_frame(frame):
    cleaned = frame.copy()
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cleaned.astype(np.float32)


def find_mssql_csv(root=MSSQL_ROOT):
    candidates = sorted(root.rglob("*.csv"))
    exact = [path for path in candidates if path.name.lower() == "mssql.csv"]
    fuzzy = [path for path in candidates if "mssql" in path.name.lower()]
    matches = exact or fuzzy
    if not matches:
        raise FileNotFoundError(f"No MSSQL CSV was found under {root}")
    return matches[0]


def load_mssql():
    path = find_mssql_csv()
    header = pd.read_csv(path, nrows=0)
    stripped_to_original = {str(column).strip(): column for column in header.columns}
    required = CANDIDATE_FEATURES + ["Label"]
    missing = [column for column in required if column not in stripped_to_original]
    if missing:
        raise KeyError(f"MSSQL is missing required columns: {missing}")

    usecols = [stripped_to_original[column] for column in required]
    frame = pd.read_csv(
        path,
        usecols=usecols,
        nrows=MSSQL_NROWS,
        low_memory=False,
    )
    frame.columns = frame.columns.str.strip()
    X = clean_feature_frame(frame[CANDIDATE_FEATURES])
    y = normalize_label(frame["Label"]).to_numpy()
    return X, y, str(path)


def load_bot_iot():
    if not BOT_PATH.exists():
        raise FileNotFoundError(
            f"{BOT_PATH} does not exist. Add dhoogla/cicbotiot to the notebook."
        )
    columns = CANDIDATE_FEATURES + ["Label"]
    frame = pd.read_parquet(BOT_PATH, columns=columns)
    frame.columns = frame.columns.str.strip()
    X = clean_feature_frame(frame[CANDIDATE_FEATURES])
    y = normalize_label(frame["Label"]).to_numpy()
    return X, y, str(BOT_PATH)


def select_semantic_proxies(X_train_raw):
    """Select one highest-variance training feature for each semantic node."""
    selected = {}
    training_variances = {}
    for node in NODE_ORDER:
        candidates = FEATURE_GROUPS[node]
        variances = X_train_raw[candidates].var(axis=0, ddof=0)
        selected[node] = str(variances.idxmax())
        training_variances[node] = {
            feature: float(variances.loc[feature]) for feature in candidates
        }
    return selected, training_variances


def undersample_training_only(X, y, seed=SEED):
    """Deterministically balance only the training partition."""
    rng = np.random.default_rng(seed)
    class_zero = np.flatnonzero(y == 0)
    class_one = np.flatnonzero(y == 1)
    n_per_class = min(len(class_zero), len(class_one))
    if n_per_class == 0:
        raise ValueError("Both benign and attack samples are required.")
    chosen = np.concatenate(
        [
            rng.choice(class_zero, n_per_class, replace=False),
            rng.choice(class_one, n_per_class, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return X[chosen], y[chosen]


def prepare_dataset(X_raw, y, dataset_name):
    """Split first; fit selector/scaler and balance on training data only."""
    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    selected, variances = select_semantic_proxies(X_train_raw)
    selected_columns = [selected[node] for node in NODE_ORDER]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw[selected_columns])
    X_test_scaled = scaler.transform(X_test_raw[selected_columns])
    X_train_scaled = X_train_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)

    X_train_balanced, y_train_balanced = undersample_training_only(
        X_train_scaled,
        np.asarray(y_train_raw, dtype=np.int8),
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_balanced,
        y_train_balanced,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=y_train_balanced,
    )

    metadata = {
        "dataset": dataset_name,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "validation_fraction_of_balanced_training": VALIDATION_SIZE,
        "raw_class_counts": {
            str(key): int(value)
            for key, value in zip(*np.unique(y, return_counts=True))
        },
        "held_out_test_class_counts": {
            str(key): int(value)
            for key, value in zip(*np.unique(y_test, return_counts=True))
        },
        "balanced_training_class_counts_before_validation_split": {
            str(key): int(value)
            for key, value in zip(
                *np.unique(y_train_balanced, return_counts=True)
            )
        },
        "selected_semantic_proxies": selected,
        "training_variances_by_group": variances,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "node_order": NODE_ORDER,
        "adjacency": A_X.tolist(),
    }

    print(f"\n{dataset_name}")
    print("Selected semantic proxies:", selected)
    print("Train / validation / untouched test:", end=" ")
    print(len(y_train), len(y_validation), len(y_test))
    print("Untouched-test class counts:", metadata["held_out_test_class_counts"])

    return (
        X_train,
        X_validation,
        X_test_scaled,
        y_train.astype(np.int8),
        y_validation.astype(np.int8),
        np.asarray(y_test, dtype=np.int8),
        metadata,
    )


def compile_binary_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_plain_mlp(input_dim=4):
    inputs = tf.keras.Input(shape=(input_dim,), name="semantic_proxies")
    hidden = tf.keras.layers.Dense(64, activation="relu", name="dense_64")(inputs)
    hidden = tf.keras.layers.Dropout(0.30, name="dropout_30")(hidden)
    hidden = tf.keras.layers.Dense(32, activation="relu", name="dense_32")(hidden)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="attack")(hidden)
    return compile_binary_model(tf.keras.Model(inputs, outputs, name="Plain_MLP"))


def build_causal_mlp(input_dim=4):
    inputs = tf.keras.Input(shape=(input_dim,), name="semantic_proxies")
    propagated = CausalPropagation(A_X, name="causal_propagation")(inputs)
    hidden = tf.keras.layers.Dense(64, activation="relu", name="dense_64")(
        propagated
    )
    hidden = tf.keras.layers.Dropout(0.30, name="dropout_30")(hidden)
    hidden = tf.keras.layers.Dense(32, activation="relu", name="dense_32")(hidden)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="attack")(hidden)
    return compile_binary_model(tf.keras.Model(inputs, outputs, name="Causal_MLP"))


def classification_metrics(y_true, probability):
    y_true = np.asarray(y_true).reshape(-1)
    probability = np.asarray(probability).reshape(-1)
    prediction = (probability >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    benign_precision = tn / (tn + fn) if (tn + fn) else 0.0
    benign_recall = tn / (tn + fp) if (tn + fp) else 0.0
    benign_f1 = (
        2 * benign_precision * benign_recall
        / (benign_precision + benign_recall)
        if (benign_precision + benign_recall)
        else 0.0
    )
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(y_true, probability),
        "benign_precision": benign_precision,
        "benign_recall": benign_recall,
        "benign_f1": benign_f1,
        "benign_average_precision": average_precision_score(
            1 - y_true, 1.0 - probability
        ),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_benign_precision_recall_curve(
    y_true, probabilities, dataset_name, training_seed
):
    """Plot benign as the positive class to expose minority-class behavior."""
    y_benign = 1 - np.asarray(y_true, dtype=np.int8).reshape(-1)
    baseline = float(np.mean(y_benign))
    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    for model_name, attack_probability in probabilities.items():
        benign_probability = 1.0 - np.asarray(attack_probability).reshape(-1)
        precision, recall, _ = precision_recall_curve(
            y_benign, benign_probability
        )
        ap = average_precision_score(y_benign, benign_probability)
        axis.step(
            recall,
            precision,
            where="post",
            label=f"{model_name} (AP={ap:.4f})",
        )
    axis.axhline(
        baseline,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Prevalence baseline ({baseline:.4f})",
    )
    axis.set(
        xlabel="Recall (benign class)",
        ylabel="Precision (benign class)",
        title=f"Benign-class precision--recall: {dataset_name}",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR
        / f"{dataset_name}_seed{training_seed}_benign_precision_recall.png",
        dpi=300,
    )
    plt.close(fig)


def save_mean_benign_precision_recall_curve(
    prediction_paths, dataset_name, model_keys
):
    """Plot mean and standard deviation across seeds on a fixed recall grid."""
    recall_grid = np.linspace(0.0, 1.0, 201)
    y_reference = None
    curves = {model: [] for model in model_keys}
    for path in prediction_paths:
        with np.load(path) as archive:
            y_benign = 1 - archive["y_true"].astype(np.int8)
            if y_reference is None:
                y_reference = y_benign.copy()
            elif not np.array_equal(y_reference, y_benign):
                raise ValueError(
                    "Held-out labels/order differ across training seeds."
                )
            for model in model_keys:
                score_key = f"{model}_attack_probability"
                if score_key not in archive:
                    continue
                benign_probability = 1.0 - archive[score_key]
                precision, recall, _ = precision_recall_curve(
                    y_benign, benign_probability
                )
                curves[model].append(
                    np.interp(recall_grid, recall[::-1], precision[::-1])
                )

    if y_reference is None:
        raise ValueError("No prediction archives were supplied.")

    baseline = float(np.mean(y_reference))
    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    for model, run_curves in curves.items():
        if not run_curves:
            continue
        values = np.asarray(run_curves)
        mean_precision = values.mean(axis=0)
        std_precision = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean_precision)
        axis.plot(recall_grid, mean_precision, label=f"{model} (n={len(values)})")
        axis.fill_between(
            recall_grid,
            np.clip(mean_precision - std_precision, 0, 1),
            np.clip(mean_precision + std_precision, 0, 1),
            alpha=0.18,
        )
    axis.axhline(
        baseline,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Prevalence baseline ({baseline:.4f})",
    )
    axis.set(
        xlabel="Recall (benign class)",
        ylabel="Precision (benign class)",
        title=f"Benign-class precision--recall: {dataset_name}",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{dataset_name}_mean_benign_precision_recall.png",
        dpi=300,
    )
    plt.close(fig)


def time_keras_model(model, X_test):
    timing_X = X_test[: min(len(X_test), TIMING_MAX_SAMPLES)]
    _ = model.predict(timing_X[: min(1024, len(timing_X))], verbose=0)
    samples = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        _ = model.predict(timing_X, batch_size=1024, verbose=0)
        elapsed = time.perf_counter() - start
        samples.append(1000.0 * elapsed / len(timing_X))
    return float(np.mean(samples)), float(np.std(samples)), len(timing_X)


def time_xgboost_model(model, X_test):
    timing_X = X_test[: min(len(X_test), TIMING_MAX_SAMPLES)]
    _ = model.predict_proba(timing_X[: min(1024, len(timing_X))])
    samples = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        _ = model.predict_proba(timing_X)
        elapsed = time.perf_counter() - start
        samples.append(1000.0 * elapsed / len(timing_X))
    return float(np.mean(samples)), float(np.std(samples)), len(timing_X)


def mean_absolute_input_gradient(model, X_test, max_samples=2048):
    sample = tf.convert_to_tensor(X_test[: min(len(X_test), max_samples)])
    with tf.GradientTape() as tape:
        tape.watch(sample)
        probability = model(sample, training=False)
    gradients = tape.gradient(probability, sample)
    importance = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()
    return {
        node: float(value) for node, value in zip(NODE_ORDER, importance)
    }


def train_neural_model(
    builder, X_train, y_train, X_validation, y_validation, seed=SEED
):
    tf.keras.backend.clear_session()
    set_global_seed(seed)
    model = builder(input_dim=X_train.shape[1])
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=2,
    )
    return model, history


def fit_xgboost_with_validation(
    X_train, y_train, X_validation, y_validation, seed
):
    """Select a small, prespecified XGBoost grid using validation AP only."""
    parameter_grid = [
        (n_estimators, max_depth, learning_rate)
        for n_estimators in (100, 200)
        for max_depth in (4, 6)
        for learning_rate in (0.05, 0.10)
    ]
    best_model = None
    best_params = None
    best_validation_ap = -np.inf
    for n_estimators, max_depth, learning_rate in parameter_grid:
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(X_train, y_train)
        validation_probability = model.predict_proba(X_validation)[:, 1]
        validation_ap = average_precision_score(
            y_validation, validation_probability
        )
        if validation_ap > best_validation_ap:
            best_model = model
            best_validation_ap = float(validation_ap)
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "selection_metric": "average_precision_on_balanced_validation",
            }
    return best_model, best_params, best_validation_ap


def save_history(history, dataset_name, model_name, training_seed):
    path = OUTPUT_DIR / (
        f"{dataset_name}_{model_name}_seed{training_seed}_history.csv"
    )
    pd.DataFrame(history.history).to_csv(path, index=False)


def run_benchmark(
    dataset_name,
    X_raw,
    y,
    source_path,
    training_seed=SEED,
    include_xgboost=True,
):
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        metadata,
    ) = prepare_dataset(X_raw, y, dataset_name)
    metadata["source_path"] = source_path
    metadata["training_seed"] = training_seed
    metadata["training_run_count_for_this_invocation"] = 1
    metadata["epochs"] = EPOCHS
    metadata["batch_size"] = BATCH_SIZE
    metadata["learning_rate"] = LEARNING_RATE
    metadata["timing_repeats"] = TIMING_REPEATS
    metadata["timing_max_samples"] = TIMING_MAX_SAMPLES

    result_rows = []
    test_probabilities = {"y_true": np.asarray(y_test, dtype=np.int8)}

    plain_model, plain_history = train_neural_model(
        build_plain_mlp,
        X_train,
        y_train,
        X_validation,
        y_validation,
        seed=training_seed,
    )
    plain_probability = plain_model.predict(X_test, batch_size=1024, verbose=0)
    test_probabilities["plain_mlp_attack_probability"] = np.asarray(
        plain_probability, dtype=np.float32
    ).reshape(-1)
    plain_result = classification_metrics(y_test, plain_probability)
    mean_ms, std_ms, timing_n = time_keras_model(plain_model, X_test)
    plain_result.update(
        {
            "dataset": dataset_name,
            "model": "Plain-MLP",
            "training_seed": training_seed,
            "latency_ms_per_flow_mean": mean_ms,
            "latency_ms_per_flow_std": std_ms,
            "latency_timing_samples": timing_n,
            "learned_alpha": np.nan,
        }
    )
    result_rows.append(plain_result)
    save_history(plain_history, dataset_name, "Plain_MLP", training_seed)
    plain_model.save(
        OUTPUT_DIR / f"{dataset_name}_Plain_MLP_seed{training_seed}.keras"
    )

    causal_model, causal_history = train_neural_model(
        build_causal_mlp,
        X_train,
        y_train,
        X_validation,
        y_validation,
        seed=training_seed,
    )
    causal_probability = causal_model.predict(X_test, batch_size=1024, verbose=0)
    test_probabilities["causal_mlp_attack_probability"] = np.asarray(
        causal_probability, dtype=np.float32
    ).reshape(-1)
    causal_result = classification_metrics(y_test, causal_probability)
    mean_ms, std_ms, timing_n = time_keras_model(causal_model, X_test)
    learned_alpha = float(
        causal_model.get_layer("causal_propagation").alpha.numpy()
    )
    causal_result.update(
        {
            "dataset": dataset_name,
            "model": "Causal-MLP",
            "training_seed": training_seed,
            "latency_ms_per_flow_mean": mean_ms,
            "latency_ms_per_flow_std": std_ms,
            "latency_timing_samples": timing_n,
            "learned_alpha": learned_alpha,
        }
    )
    result_rows.append(causal_result)
    save_history(causal_history, dataset_name, "Causal_MLP", training_seed)
    causal_model.save(
        OUTPUT_DIR / f"{dataset_name}_Causal_MLP_seed{training_seed}.keras"
    )

    gradient_importance = mean_absolute_input_gradient(causal_model, X_test)
    pd.DataFrame(
        {
            "semantic_node": list(gradient_importance.keys()),
            "mean_absolute_input_gradient": list(gradient_importance.values()),
        }
    ).sort_values(
        "mean_absolute_input_gradient", ascending=False
    ).to_csv(
        OUTPUT_DIR
        / f"{dataset_name}_seed{training_seed}_causal_gradient_importance.csv",
        index=False,
    )

    if include_xgboost:
        set_global_seed(training_seed)
        xgb_model, xgb_params, xgb_validation_ap = fit_xgboost_with_validation(
            X_train,
            y_train,
            X_validation,
            y_validation,
            seed=training_seed,
        )
        xgb_probability = xgb_model.predict_proba(X_test)[:, 1]
        test_probabilities["xgboost_attack_probability"] = np.asarray(
            xgb_probability, dtype=np.float32
        ).reshape(-1)
        xgb_result = classification_metrics(y_test, xgb_probability)
        mean_ms, std_ms, timing_n = time_xgboost_model(xgb_model, X_test)
        xgb_result.update(
            {
                "dataset": dataset_name,
                "model": "XGBoost",
                "training_seed": training_seed,
                "validation_average_precision": xgb_validation_ap,
                "latency_ms_per_flow_mean": mean_ms,
                "latency_ms_per_flow_std": std_ms,
                "latency_timing_samples": timing_n,
                "learned_alpha": np.nan,
            }
        )
        result_rows.append(xgb_result)
        xgb_model.save_model(
            OUTPUT_DIR / f"{dataset_name}_XGBoost_seed{training_seed}.json"
        )
        metadata["xgboost_best_params"] = xgb_params
        metadata["xgboost_validation_average_precision"] = xgb_validation_ap

    np.savez_compressed(
        OUTPUT_DIR / f"{dataset_name}_seed{training_seed}_test_predictions.npz",
        **test_probabilities,
    )
    probability_columns = {
        name.removesuffix("_attack_probability"): values
        for name, values in test_probabilities.items()
        if name != "y_true"
    }
    save_benign_precision_recall_curve(
        y_test, probability_columns, dataset_name, training_seed
    )

    metadata["causal_gradient_importance"] = gradient_importance
    metadata["learned_alpha"] = learned_alpha
    with open(
        OUTPUT_DIR / f"{dataset_name}_seed{training_seed}_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)

    results = pd.DataFrame(result_rows)
    results.to_csv(
        OUTPUT_DIR / f"{dataset_name}_seed{training_seed}_results.csv",
        index=False,
    )
    print("\nValidated results:")
    print(results.to_string(index=False))
    print("Causal input-gradient importance:", gradient_importance)
    print("Learned alpha:", learned_alpha)
    return results, metadata


# %% [Cell 3] Controlled synthetic SCM, DoWhy estimate/refutation, and RF F1
# These equations intentionally match the original synthetic experiment.  The
# outputs are recomputed here; no result is hard-coded.
set_global_seed()
np.random.seed(SEED)
n_samples = 2000

burst_duration = np.random.exponential(scale=0.5, size=n_samples)
normal_traffic = np.random.normal(loc=50.0, scale=10.0, size=n_samples)
tcp_timeout = (
    1.0
    + 0.8 * burst_duration
    + np.random.normal(0.0, 0.1, n_samples)
)
queue_drop = (
    0.3 * tcp_timeout
    + 0.02 * normal_traffic
    + np.random.normal(0.0, 0.05, n_samples)
)
attack_probability = 1.0 / (
    1.0 + np.exp(-(2.0 * burst_duration + 1.5 * queue_drop - 1.0))
)
attack_label = np.random.binomial(1, attack_probability)

synthetic = pd.DataFrame(
    {
        "Normal_Traffic": normal_traffic,
        "Burst_Duration": burst_duration,
        "TCP_Timeout": tcp_timeout,
        "Queue_Drop": queue_drop,
        "Attack_Label": attack_label,
    }
)

causal_graph = """
digraph {
    Normal_Traffic -> Queue_Drop;
    Normal_Traffic -> Attack_Label;
    Burst_Duration -> TCP_Timeout;
    Burst_Duration -> Attack_Label;
    TCP_Timeout -> Queue_Drop;
    Queue_Drop -> Attack_Label;
}
"""
causal_model = CausalModel(
    data=synthetic,
    treatment="Burst_Duration",
    outcome="Attack_Label",
    graph=causal_graph,
)
identified_estimand = causal_model.identify_effect(
    proceed_when_unidentifiable=True
)
estimate = causal_model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
    confidence_intervals=True,
)
refutation = causal_model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="random_common_cause",
    num_simulations=100,
    random_state=SEED,
)

X_synthetic = synthetic[
    ["Normal_Traffic", "Burst_Duration", "TCP_Timeout", "Queue_Drop"]
]
y_synthetic = synthetic["Attack_Label"]
X_syn_train, X_syn_test, y_syn_train, y_syn_test = train_test_split(
    X_synthetic,
    y_synthetic,
    test_size=TEST_SIZE,
    random_state=SEED,
)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=SEED,
    n_jobs=-1,
)
rf_model.fit(X_syn_train, y_syn_train)
rf_prediction = rf_model.predict(X_syn_test)

synthetic_results = pd.DataFrame(
    [
        {
            "seed": SEED,
            "n_samples": n_samples,
            "treatment": "Burst_Duration",
            "outcome": "Attack_Label",
            "control_value": float(estimate.control_value),
            "treatment_value": float(estimate.treatment_value),
            "ate": float(estimate.value),
            "refuted_ate": float(refutation.new_effect),
            "refutation_method": "random_common_cause",
            "refutation_simulations": 100,
            "rf_f1": f1_score(y_syn_test, rf_prediction),
        }
    ]
)
synthetic_results.to_csv(
    OUTPUT_DIR / "validated_synthetic_results.csv",
    index=False,
)
synthetic.to_csv(OUTPUT_DIR / "validated_synthetic_data.csv", index=False)
print(synthetic_results.to_string(index=False))
print("Identified estimand:")
print(identified_estimand)
print("Refutation:")
print(refutation)


# %% [Cell 4] Leakage-safe CIC-DDoS2019 MSSQL benchmark
# For a quick pipeline test, set MSSQL_NROWS in Cell 1.  For paper results,
# keep MSSQL_NROWS=None and restart/run all so the full file is evaluated.
X_mssql, y_mssql, mssql_source = load_mssql()
print("MSSQL source:", mssql_source)
print("MSSQL shape:", X_mssql.shape)
mssql_run_results = []
for training_seed in TRAINING_SEEDS:
    run_results, mssql_metadata = run_benchmark(
        dataset_name="CIC_DDoS2019_MSSQL",
        X_raw=X_mssql,
        y=y_mssql,
        source_path=mssql_source,
        training_seed=training_seed,
        include_xgboost=True,
    )
    mssql_run_results.append(run_results)
mssql_results = pd.concat(mssql_run_results, ignore_index=True)
mssql_results.to_csv(
    OUTPUT_DIR / "CIC_DDoS2019_MSSQL_results.csv", index=False
)
save_mean_benign_precision_recall_curve(
    [
        OUTPUT_DIR
        / f"CIC_DDoS2019_MSSQL_seed{seed}_test_predictions.npz"
        for seed in TRAINING_SEEDS
    ],
    "CIC_DDoS2019_MSSQL",
    ("plain_mlp", "causal_mlp", "xgboost"),
)
del X_mssql, y_mssql


# %% [Cell 5] Leakage-safe third-party CICFlowMeter Bot-IoT benchmark
# This cell consumes the published dhoogla/cicbotiot parquet variant.  It does
# NOT claim that the paper authors re-extracted the original PCAP files.
X_bot, y_bot, bot_source = load_bot_iot()
print("Bot-IoT source:", bot_source)
print("Bot-IoT shape:", X_bot.shape)
bot_run_results = []
for training_seed in TRAINING_SEEDS:
    run_results, bot_metadata = run_benchmark(
        dataset_name="CIC_Bot_IoT",
        X_raw=X_bot,
        y=y_bot,
        source_path=bot_source,
        training_seed=training_seed,
        include_xgboost=True,
    )
    bot_run_results.append(run_results)
bot_results = pd.concat(bot_run_results, ignore_index=True)
bot_results.to_csv(OUTPUT_DIR / "CIC_Bot_IoT_results.csv", index=False)
save_mean_benign_precision_recall_curve(
    [
        OUTPUT_DIR / f"CIC_Bot_IoT_seed{seed}_test_predictions.npz"
        for seed in TRAINING_SEEDS
    ],
    "CIC_Bot_IoT",
    ("plain_mlp", "causal_mlp", "xgboost"),
)
del X_bot, y_bot


# %% [Cell 6] Combine validated artifacts and print a paper-ready summary
benchmark_files = [
    OUTPUT_DIR / "CIC_DDoS2019_MSSQL_results.csv",
    OUTPUT_DIR / "CIC_Bot_IoT_results.csv",
]
available_files = [path for path in benchmark_files if path.exists()]
if not available_files:
    raise FileNotFoundError("Run the MSSQL and Bot-IoT benchmark cells first.")

all_results = pd.concat(
    [pd.read_csv(path) for path in available_files],
    ignore_index=True,
)
all_results.to_csv(OUTPUT_DIR / "validated_all_results.csv", index=False)

paper_columns = [
    "dataset",
    "model",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "false_positive_rate",
    "latency_ms_per_flow_mean",
    "latency_ms_per_flow_std",
    "learned_alpha",
]
print("\nPaper-ready benchmark table")
print(all_results[paper_columns].round(6).to_string(index=False))
print("\nSaved artifacts:")
for artifact in sorted(OUTPUT_DIR.iterdir()):
    print(" -", artifact.name)
print("\nUse only validated_synthetic_results.csv and validated_all_results.csv when updating the manuscript.")


# %% [Cell 7] Validate, archive, and expose all reproducibility artifacts
import hashlib
import os
import shutil
from IPython.display import FileLink, display


source_dir = Path("/kaggle/working/validated_outputs")
archive_base = Path("/kaggle/working/causal_ddos_validated_outputs")
zip_path = archive_base.with_suffix(".zip")

if not source_dir.is_dir():
    raise FileNotFoundError(
        f"Output directory not found: {source_dir}. "
        "Run experiment Cells 1--6 in the current Kaggle session first."
    )

artifact_files = sorted(
    path
    for path in source_dir.rglob("*")
    if path.is_file() and path.name != "SHA256SUMS.txt"
)
if not artifact_files:
    raise FileNotFoundError(f"No result files were found inside {source_dir}")

print(f"Source directory: {source_dir}")
print(f"Files found: {len(artifact_files)}")
for artifact in artifact_files:
    print(" -", artifact.relative_to(source_dir))


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# Save checksums before creating the archive.
checksum_path = source_dir / "SHA256SUMS.txt"
with open(checksum_path, "w", encoding="utf-8") as handle:
    for artifact in artifact_files:
        relative_name = artifact.relative_to(source_dir).as_posix()
        handle.write(f"{sha256_file(artifact)}  {relative_name}\n")

if zip_path.exists():
    zip_path.unlink()

created_path = Path(
    shutil.make_archive(
        base_name=str(archive_base),
        format="zip",
        root_dir=str(source_dir.parent),
        base_dir=source_dir.name,
    )
)

if not created_path.is_file():
    raise RuntimeError(f"ZIP creation failed: {created_path}")

size_mb = created_path.stat().st_size / (1024 ** 2)
print("\nZIP created successfully")
print("Path:", created_path)
print(f"Size: {size_mb:.2f} MB")
print("SHA-256:", sha256_file(created_path))
print("Working-directory ZIP files:", sorted(Path("/kaggle/working").glob("*.zip")))

# FileLink is more reliable in Kaggle when it is relative to /kaggle/working.
current_directory = Path.cwd()
try:
    os.chdir("/kaggle/working")
    display(FileLink(created_path.name, result_html_prefix="Download ZIP: "))
finally:
    os.chdir(current_directory)

print(
    "If the link is not clickable, refresh the Kaggle Files panel and download "
    f"{created_path.name} from /kaggle/working."
)
