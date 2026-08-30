from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# GraphGuard AI
# Phase 2 — XGBoost Feature-Only Baseline
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2"

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"

MODEL_FILE = OUTPUT_DIR / "xgboost.joblib"
METRICS_FILE = OUTPUT_DIR / "xgboost_metrics.json"


# ------------------------------------------------------------
# Temporal split
# ------------------------------------------------------------

TRAIN_END = 34

VAL_START = 35
VAL_END = 39

TEST_START = 40
TEST_END = 49


def load_data():
    """Load the feature and class datasets."""

    print("\nLoading features...")

    feature_columns = (
        ["txId", "time_step"]
        + [f"feature_{i}" for i in range(1, 166)]
    )

    features = pd.read_csv(
        FEATURES_FILE,
        header=None,
        names=feature_columns,
    )

    print(
        f"Features loaded: {len(features):,}"
    )

    print("\nLoading classes...")

    classes = pd.read_csv(
        CLASSES_FILE
    )

    print(
        f"Classes loaded: {len(classes):,}"
    )

    data = features.merge(
        classes,
        on="txId",
        how="left",
        validate="one_to_one",
    )

    return data


def prepare_labeled_data(data):
    """
    Keep only known labels.

    class 1 = illicit
    class 2 = licit
    """

    labeled = data[
        data["class"].isin(["1", "2"])
    ].copy()

    labeled["target"] = (
        labeled["class"] == "1"
    ).astype(int)

    return labeled


def create_temporal_split(data):
    """Create chronological train/validation/test sets."""

    train = data[
        data["time_step"] <= TRAIN_END
    ].copy()

    validation = data[
        (data["time_step"] >= VAL_START)
        & (data["time_step"] <= VAL_END)
    ].copy()

    test = data[
        (data["time_step"] >= TEST_START)
        & (data["time_step"] <= TEST_END)
    ].copy()

    return train, validation, test


def get_feature_columns(data):
    """Return only the 165 predictive features."""

    return [
        column
        for column in data.columns
        if column.startswith("feature_")
    ]


def calculate_scale_pos_weight(y):
    """
    Calculate the class imbalance ratio used by XGBoost.

    scale_pos_weight = negative / positive
    """

    positives = int(y.sum())
    negatives = int((y == 0).sum())

    return negatives / positives


def evaluate_model(
    model,
    X,
    y,
    split_name,
):
    """Evaluate predictions using threshold 0.5."""

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "split": split_name,
        "samples": int(len(y)),
        "positive_samples": int(y.sum()),
        "negative_samples": int(
            (y == 0).sum()
        ),
        "positive_rate": float(
            y.mean()
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                probabilities,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y,
                predictions,
            ).tolist()
        ),
    }

    print(
        f"\n{split_name.upper()} RESULTS"
    )

    print(
        f"Samples: "
        f"{metrics['samples']:,}"
    )

    print(
        f"Positive rate: "
        f"{metrics['positive_rate']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print("Confusion matrix:")

    print(
        np.array(
            metrics["confusion_matrix"]
        )
    )

    return metrics


def main():

    print("\n" + "=" * 70)
    print(
        "GraphGuard AI — "
        "XGBoost Feature-Only Baseline"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = load_data()

    # --------------------------------------------------------
    # Remove unknown labels
    # --------------------------------------------------------

    labeled = prepare_labeled_data(
        data
    )

    print(
        f"\nLabeled transactions: "
        f"{len(labeled):,}"
    )

    # --------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------

    train, validation, test = (
        create_temporal_split(
            labeled
        )
    )

    print("\nTemporal split:")

    print(
        f"Train: "
        f"timesteps 1–34 → "
        f"{len(train):,} transactions"
    )

    print(
        f"Validation: "
        f"timesteps 35–39 → "
        f"{len(validation):,} transactions"
    )

    print(
        f"Test: "
        f"timesteps 40–49 → "
        f"{len(test):,} transactions"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_columns = (
        get_feature_columns(
            labeled
        )
    )

    print(
        f"\nNumber of model features: "
        f"{len(feature_columns)}"
    )

    X_train = train[
        feature_columns
    ]

    y_train = train["target"]

    X_validation = validation[
        feature_columns
    ]

    y_validation = validation[
        "target"
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test["target"]

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    scale_pos_weight = (
        calculate_scale_pos_weight(
            y_train
        )
    )

    print(
        f"\nXGBoost scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nTraining XGBoost..."
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    print(
        "Training completed."
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    train_metrics = evaluate_model(
        model,
        X_train,
        y_train,
        "train",
    )

    validation_metrics = (
        evaluate_model(
            model,
            X_validation,
            y_validation,
            "validation",
        )
    )

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "test",
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "train_end": TRAIN_END,
        "validation_start": VAL_START,
        "validation_end": VAL_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
    }

    joblib.dump(
        artifact,
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results = {
        "model": "XGBoost",
        "features": len(
            feature_columns
        ),
        "scale_pos_weight": (
            scale_pos_weight
        ),
        "parameters": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "temporal_split": {
            "train": "1-34",
            "validation": "35-39",
            "test": "40-49",
        },
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        f"\nModel saved to: "
        f"{MODEL_FILE}"
    )

    print(
        f"Metrics saved to: "
        f"{METRICS_FILE}"
    )

    print("\n" + "=" * 70)
    print(
        "XGBOOST BASELINE COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()