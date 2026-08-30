from pathlib import Path
import json

import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# GraphGuard AI
# Feature Group Ablation
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

FEATURES_FILE = (
    DATA_DIR / "elliptic_txs_features.csv"
)

CLASSES_FILE = (
    DATA_DIR / "elliptic_txs_classes.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "phase2"
)

METRICS_FILE = (
    OUTPUT_DIR / "feature_ablation_metrics.json"
)


# ------------------------------------------------------------
# Feature groups
# ------------------------------------------------------------

LOCAL_FEATURES = [
    f"feature_{i}"
    for i in range(1, 95)
]

AGGREGATED_FEATURES = [
    f"feature_{i}"
    for i in range(95, 166)
]

ALL_FEATURES = (
    LOCAL_FEATURES
    + AGGREGATED_FEATURES
)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

def load_data():

    print("\nLoading features...")

    columns = (
        ["txId", "time_step"]
        + ALL_FEATURES
    )

    features = pd.read_csv(
        FEATURES_FILE,
        header=None,
        names=columns,
    )

    print(
        f"Features loaded: "
        f"{len(features):,}"
    )

    print("\nLoading classes...")

    classes = pd.read_csv(
        CLASSES_FILE
    )

    data = features.merge(
        classes,
        on="txId",
        how="left",
        validate="one_to_one",
    )

    # Only labeled transactions
    data = data[
        data["class"].isin(["1", "2"])
    ].copy()

    # 1 = illicit
    # 2 = licit
    data["target"] = (
        data["class"] == "1"
    ).astype(int)

    print(
        f"Labeled transactions: "
        f"{len(data):,}"
    )

    return data


# ------------------------------------------------------------
# Temporal split
# ------------------------------------------------------------

def create_splits(data):

    train = data[
        data["time_step"].between(1, 34)
    ]

    validation = data[
        data["time_step"].between(35, 39)
    ]

    test = data[
        data["time_step"].between(40, 49)
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

    return train, validation, test


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

def train_model(
    train,
    validation,
    test,
    feature_columns,
    group_name,
):

    print("\n" + "=" * 70)

    print(
        f"Training XGBoost — "
        f"{group_name}"
    )

    print("=" * 70)

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
        f"Features: "
        f"{len(feature_columns)}"
    )

    print(
        f"scale_pos_weight: "
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

    print("\nTraining...")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation)
        ],
        verbose=False,
    )

    print("Training completed.")

    results = {}

    for (
        split_name,
        X_split,
        y_split,
    ) in [
        (
            "train",
            X_train,
            y_train,
        ),
        (
            "validation",
            X_validation,
            y_validation,
        ),
        (
            "test",
            X_test,
            y_test,
        ),
    ]:

        probabilities = (
            model.predict_proba(
                X_split
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        metrics = {
            "samples": int(
                len(y_split)
            ),

            "positive_rate": float(
                y_split.mean()
            ),

            "precision": float(
                precision_score(
                    y_split,
                    predictions,
                    zero_division=0,
                )
            ),

            "recall": float(
                recall_score(
                    y_split,
                    predictions,
                    zero_division=0,
                )
            ),

            "f1": float(
                f1_score(
                    y_split,
                    predictions,
                    zero_division=0,
                )
            ),

            "pr_auc": float(
                average_precision_score(
                    y_split,
                    probabilities,
                )
            ),

            "roc_auc": float(
                roc_auc_score(
                    y_split,
                    probabilities,
                )
            ),

            "confusion_matrix": (
                confusion_matrix(
                    y_split,
                    predictions,
                ).tolist()
            ),
        }

        results[split_name] = metrics

        print(
            f"\n{split_name.upper()} RESULTS"
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

    return results


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("\n" + "=" * 70)

    print(
        "GraphGuard AI — "
        "Feature Group Ablation"
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
    ) = create_splits(data)

    # --------------------------------------------------------
    # Local features
    # --------------------------------------------------------

    local_results = train_model(
        train,
        validation,
        test,
        LOCAL_FEATURES,
        "LOCAL FEATURES",
    )

    # --------------------------------------------------------
    # Aggregated features
    # --------------------------------------------------------

    aggregated_results = train_model(
        train,
        validation,
        test,
        AGGREGATED_FEATURES,
        "AGGREGATED FEATURES",
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "feature_groups": {
            "local": {
                "count": len(
                    LOCAL_FEATURES
                ),
                "range": "feature_1-feature_94",
            },

            "aggregated": {
                "count": len(
                    AGGREGATED_FEATURES
                ),
                "range": "feature_95-feature_165",
            },
        },

        "temporal_split": {
            "train": "1-34",
            "validation": "35-39",
            "test": "40-49",
        },

        "local": local_results,

        "aggregated": aggregated_results,
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
        f"\nResults saved to:"
    )

    print(METRICS_FILE)

    print("\n" + "=" * 70)

    print(
        "FEATURE ABLATION COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()