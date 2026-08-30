from pathlib import Path
import json

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
# Phase 4 — Graph-Enhanced XGBoost
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

FEATURES_FILE = (
    DATA_DIR / "elliptic_txs_features.csv"
)

CLASSES_FILE = (
    DATA_DIR / "elliptic_txs_classes.csv"
)

GRAPH_FEATURES_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "phase4"
    / "graph_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase4"
)

MODEL_FILE = (
    OUTPUT_DIR
    / "graph_xgboost.joblib"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "graph_xgboost_metrics.json"
)


# ============================================================
# Feature definitions
# ============================================================

ORIGINAL_FEATURES = [
    f"feature_{i}"
    for i in range(1, 166)
]


GRAPH_FEATURES = [
    "in_degree",
    "out_degree",
    "total_degree",
    "degree_imbalance",
    "absolute_degree_imbalance",
    "unique_out_neighbors",
    "unique_in_neighbors",
    "unique_neighbors",
    "has_incoming",
    "has_outgoing",
    "is_source",
    "is_sink",
    "is_isolated",
    "in_degree_ratio",
    "out_degree_ratio",
]


ALL_MODEL_FEATURES = (
    ORIGINAL_FEATURES
    + GRAPH_FEATURES
)


# ============================================================
# Load data
# ============================================================

def load_data():

    print("\nLoading original features...")

    columns = (
        ["txId", "time_step"]
        + ORIGINAL_FEATURES
    )

    features = pd.read_csv(
        FEATURES_FILE,
        header=None,
        names=columns,
    )

    print(
        f"Original feature rows: "
        f"{len(features):,}"
    )

    print("\nLoading classes...")

    classes = pd.read_csv(
        CLASSES_FILE
    )

    print("\nLoading graph features...")

    graph_features = pd.read_csv(
        GRAPH_FEATURES_FILE
    )

    print(
        f"Graph feature rows: "
        f"{len(graph_features):,}"
    )

    # --------------------------------------------------------
    # Merge graph features
    # --------------------------------------------------------

    data = features.merge(
        graph_features,
        on=["txId", "time_step"],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Merge labels
    # --------------------------------------------------------

    data = data.merge(
        classes,
        on="txId",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Keep labeled transactions
    # --------------------------------------------------------

    data = data[
        data["class"].isin(
            ["1", "2"]
        )
    ].copy()

    data["target"] = (
        data["class"] == "1"
    ).astype(int)

    print(
        f"\nLabeled transactions: "
        f"{len(data):,}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    missing_graph_values = (
        data[GRAPH_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    if missing_graph_values != 0:

        raise ValueError(
            "Missing graph feature values "
            f"found: {missing_graph_values}"
        )

    print(
        "Graph feature merge validated."
    )

    return data


# ============================================================
# Temporal split
# ============================================================

def create_splits(data):

    train = data[
        data["time_step"].between(
            1,
            34,
        )
    ]

    validation = data[
        data["time_step"].between(
            35,
            39,
        )
    ]

    test = data[
        data["time_step"].between(
            40,
            49,
        )
    ]

    print("\nTemporal split:")

    print(
        f"Train: "
        f"{len(train):,}"
    )

    print(
        f"Validation: "
        f"{len(validation):,}"
    )

    print(
        f"Test: "
        f"{len(test):,}"
    )

    return (
        train,
        validation,
        test,
    )


# ============================================================
# Train model
# ============================================================

def train_model(
    train,
    validation,
    test,
):

    print("\n" + "=" * 70)

    print(
        "Training Graph-Enhanced XGBoost"
    )

    print("=" * 70)

    X_train = train[
        ALL_MODEL_FEATURES
    ]

    y_train = train[
        "target"
    ]

    X_validation = validation[
        ALL_MODEL_FEATURES
    ]

    y_validation = validation[
        "target"
    ]

    X_test = test[
        ALL_MODEL_FEATURES
    ]

    y_test = test[
        "target"
    ]

    print(
        f"Total model features: "
        f"{len(ALL_MODEL_FEATURES)}"
    )

    print(
        f"Original features: "
        f"{len(ORIGINAL_FEATURES)}"
    )

    print(
        f"Graph features: "
        f"{len(GRAPH_FEATURES)}"
    )

    # --------------------------------------------------------
    # Class weight
    # --------------------------------------------------------

    positive_count = (
        y_train == 1
    ).sum()

    negative_count = (
        y_train == 0
    ).sum()

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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

    print("\nTraining...")

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

    return (
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    X,
    y,
    split_name,
):

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "samples": int(
            len(y)
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

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "GraphGuard AI — "
        "Graph-Enhanced XGBoost"
    )

    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data()

    (
        train,
        validation,
        test,
    ) = create_splits(
        data
    )

    (
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = train_model(
        train,
        validation,
        test,
    )

    train_metrics = evaluate(
        model,
        X_train,
        y_train,
        "train",
    )

    validation_metrics = evaluate(
        model,
        X_validation,
        y_validation,
        "validation",
    )

    test_metrics = evaluate(
        model,
        X_test,
        y_test,
        "test",
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    import joblib

    joblib.dump(
        {
            "model": model,
            "feature_columns":
                ALL_MODEL_FEATURES,
            "original_features":
                ORIGINAL_FEATURES,
            "graph_features":
                GRAPH_FEATURES,
        },
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results = {
        "model":
            "Graph-Enhanced XGBoost",

        "feature_count":
            len(ALL_MODEL_FEATURES),

        "original_feature_count":
            len(ORIGINAL_FEATURES),

        "graph_feature_count":
            len(GRAPH_FEATURES),

        "temporal_split": {
            "train": "1-34",
            "validation": "35-39",
            "test": "40-49",
        },

        "train":
            train_metrics,

        "validation":
            validation_metrics,

        "test":
            test_metrics,
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
        f"\nModel saved to:"
    )

    print(MODEL_FILE)

    print(
        f"Metrics saved to:"
    )

    print(METRICS_FILE)

    print("\n" + "=" * 70)

    print(
        "GRAPH-ENHANCED XGBOOST COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()