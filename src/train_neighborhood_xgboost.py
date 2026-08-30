from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
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
# Phase 5 — Leakage-Safe Neighborhood XGBoost
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase5"

FEATURES_FILE = (
    DATA_DIR / "elliptic_txs_features.csv"
)

CLASSES_FILE = (
    DATA_DIR / "elliptic_txs_classes.csv"
)

NEIGHBORHOOD_FILE = (
    OUTPUT_DIR / "leakage_safe_neighborhood_risk.csv"
)

MODEL_FILE = (
    OUTPUT_DIR / "neighborhood_xgboost.joblib"
)

METRICS_FILE = (
    OUTPUT_DIR / "neighborhood_xgboost_metrics.json"
)


# ============================================================
# Feature definitions
# ============================================================

ORIGINAL_FEATURES = [
    f"feature_{i}"
    for i in range(1, 166)
]

NEIGHBORHOOD_FEATURES = [
    "model_risk",
    "incoming_mean_risk",
    "incoming_max_risk",
    "incoming_median_risk",
    "incoming_std_risk",
    "incoming_risk_neighbor_count",
    "outgoing_mean_risk",
    "outgoing_max_risk",
    "outgoing_median_risk",
    "outgoing_std_risk",
    "outgoing_risk_neighbor_count",
    "neighborhood_mean_risk",
    "neighborhood_max_risk",
    "neighborhood_median_risk",
    "neighborhood_neighbor_count",
    "incoming_high_risk_neighbors",
    "outgoing_high_risk_neighbors",
    "high_risk_neighbor_count",
    "high_risk_neighbor_fraction",
    "neighborhood_vs_self_risk",
]

MODEL_FEATURES = (
    ORIGINAL_FEATURES
    + NEIGHBORHOOD_FEATURES
)


# ============================================================
# Evaluation helper
# ============================================================

def evaluate_model(
    name,
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    cm = confusion_matrix(
        y_true,
        predictions,
    )

    print(
        f"\n{name} RESULTS"
    )

    print(
        f"Samples: {len(y_true):,}"
    )

    print(
        f"Positive rate: "
        f"{np.mean(y_true):.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )

    print(
        f"PR-AUC: {pr_auc:.4f}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        "Confusion matrix:"
    )

    print(cm)

    return {
        "samples": int(len(y_true)),
        "positive_rate": float(
            np.mean(y_true)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# Load data
# ============================================================

def load_data():

    print(
        "\nLoading original transaction features..."
    )

    feature_columns = [
        "txId",
        "time_step",
    ] + ORIGINAL_FEATURES

    features = pd.read_csv(
        FEATURES_FILE,
        header=None,
        names=feature_columns,
    )

    print(
        f"Features loaded: "
        f"{len(features):,}"
    )

    print(
        "\nLoading classes..."
    )

    classes = pd.read_csv(
        CLASSES_FILE
    )

    print(
        f"Classes loaded: "
        f"{len(classes):,}"
    )

    print(
        "\nLoading leakage-safe neighborhood features..."
    )

    neighborhood = pd.read_csv(
        NEIGHBORHOOD_FILE
    )

    print(
        f"Neighborhood rows loaded: "
        f"{len(neighborhood):,}"
    )

    return (
        features,
        classes,
        neighborhood,
    )


# ============================================================
# Prepare merged dataset
# ============================================================

def prepare_data(
    features,
    classes,
    neighborhood,
):

    data = features.merge(
        classes[
            [
                "txId",
                "class",
            ]
        ],
        on="txId",
        how="left",
        validate="one_to_one",
    )

    data["target"] = (
        data["class"] == "1"
    ).astype(int)

    data = data.merge(
        neighborhood[
            [
                "txId",
                *NEIGHBORHOOD_FEATURES,
            ]
        ],
        on="txId",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Only transactions with an actual leakage-safe model
    # prediction can be used by this experiment.
    # --------------------------------------------------------

    data = data[
        data["model_risk"].notna()
    ].copy()

    print(
        "\nRows with leakage-safe predictions:"
    )

    print(
        f"{len(data):,}"
    )

    # Only labeled transactions can be used for supervised
    # training/evaluation.

    data = data[
        data["class"].isin(
            ["1", "2"]
        )
    ].copy()

    print(
        "Labeled rows with leakage-safe predictions:"
    )

    print(
        f"{len(data):,}"
    )

    # --------------------------------------------------------
    # Verify no missing model features.
    # --------------------------------------------------------

    missing = (
        data[MODEL_FEATURES]
        .isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ]

    if len(missing) > 0:

        print(
            "\nMissing values detected:"
        )

        print(
            missing.to_string()
        )

        raise ValueError(
            "Model features contain missing values."
        )

    if (
        np.isinf(
            data[
                MODEL_FEATURES
            ]
            .to_numpy()
        )
        .sum()
        != 0
    ):

        raise ValueError(
            "Infinite values detected."
        )

    return data


# ============================================================
# Train model
# ============================================================

def train_model(
    train_data,
):

    X_train = train_data[
        MODEL_FEATURES
    ]

    y_train = train_data[
        "target"
    ]

    positives = (
        y_train == 1
    ).sum()

    negatives = (
        y_train == 0
    ).sum()

    scale_pos_weight = (
        negatives / positives
    )

    print(
        f"\nTraining positives: "
        f"{positives:,}"
    )

    print(
        f"Training negatives: "
        f"{negatives:,}"
    )

    print(
        f"Scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    print(
        "\nTraining Neighborhood XGBoost..."
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    print(
        "Training completed."
    )

    return model


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "GraphGuard AI — "
        "Leakage-Safe Neighborhood XGBoost"
    )

    print(
        "=" * 70
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        features,
        classes,
        neighborhood,
    ) = load_data()

    data = prepare_data(
        features,
        classes,
        neighborhood,
    )

    # --------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------

    train = data[
        data["time_step"].between(
            11,
            34,
        )
    ].copy()

    validation = data[
        data["time_step"].between(
            35,
            39,
        )
    ].copy()

    test = data[
        data["time_step"].between(
            40,
            49,
        )
    ].copy()

    print(
        "\nTemporal split:"
    )

    print(
        f"Train: timesteps 11–34 → "
        f"{len(train):,} transactions"
    )

    print(
        f"Validation: timesteps 35–39 → "
        f"{len(validation):,} transactions"
    )

    print(
        f"Test: timesteps 40–49 → "
        f"{len(test):,} transactions"
    )

    # --------------------------------------------------------
    # Feature summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL FEATURES"
    )

    print(
        "=" * 70
    )

    print(
        f"Original features: "
        f"{len(ORIGINAL_FEATURES)}"
    )

    print(
        f"Neighborhood features: "
        f"{len(NEIGHBORHOOD_FEATURES)}"
    )

    print(
        f"Total features: "
        f"{len(MODEL_FEATURES)}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    train_probabilities = (
        model.predict_proba(
            train[
                MODEL_FEATURES
            ]
        )[:, 1]
    )

    validation_probabilities = (
        model.predict_proba(
            validation[
                MODEL_FEATURES
            ]
        )[:, 1]
    )

    test_probabilities = (
        model.predict_proba(
            test[
                MODEL_FEATURES
            ]
        )[:, 1]
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    metrics = {}

    metrics[
        "train"
    ] = evaluate_model(
        "TRAIN",
        train["target"].to_numpy(),
        train_probabilities,
    )

    metrics[
        "validation"
    ] = evaluate_model(
        "VALIDATION",
        validation["target"].to_numpy(),
        validation_probabilities,
    )

    metrics[
        "test"
    ] = evaluate_model(
        "TEST",
        test["target"].to_numpy(),
        test_probabilities,
    )

    # --------------------------------------------------------
    # Baseline comparison
    # --------------------------------------------------------

    baseline_pr_auc = 0.6738
    graph_xgb_pr_auc = 0.6756

    test_pr_auc = metrics[
        "test"
    ][
        "pr_auc"
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "BASELINE COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        f"Feature-only XGBoost PR-AUC: "
        f"{baseline_pr_auc:.4f}"
    )

    print(
        f"Graph-enhanced XGBoost PR-AUC: "
        f"{graph_xgb_pr_auc:.4f}"
    )

    print(
        f"Leakage-safe Neighborhood XGBoost PR-AUC: "
        f"{test_pr_auc:.4f}"
    )

    print(
        f"\nImprovement vs feature-only: "
        f"{test_pr_auc - baseline_pr_auc:+.4f}"
    )

    print(
        f"Improvement vs graph-enhanced: "
        f"{test_pr_auc - graph_xgb_pr_auc:+.4f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print(
        f"\nModel saved to:"
    )

    print(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    output_metrics = {
        "model": (
            "leakage_safe_neighborhood_xgboost"
        ),
        "original_feature_count": len(
            ORIGINAL_FEATURES
        ),
        "neighborhood_feature_count": len(
            NEIGHBORHOOD_FEATURES
        ),
        "total_feature_count": len(
            MODEL_FEATURES
        ),
        "train_timesteps": "11-34",
        "validation_timesteps": "35-39",
        "test_timesteps": "40-49",
        "baseline_feature_only_pr_auc": (
            baseline_pr_auc
        ),
        "graph_enhanced_xgboost_pr_auc": (
            graph_xgb_pr_auc
        ),
        "metrics": metrics,
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output_metrics,
            file,
            indent=4,
        )

    print(
        "Metrics saved to:"
    )

    print(
        METRICS_FILE
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "LEAKAGE-SAFE NEIGHBORHOOD XGBOOST COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()