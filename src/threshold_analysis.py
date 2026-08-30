from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


# ============================================================
# GraphGuard AI
# Phase 6B — Threshold Analysis
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase5"

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"

NEIGHBORHOOD_FILE = (
    OUTPUT_DIR / "leakage_safe_neighborhood_risk.csv"
)

MODEL_FILE = (
    OUTPUT_DIR / "neighborhood_xgboost.joblib"
)

RESULTS_FILE = (
    OUTPUT_DIR / "threshold_analysis.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "threshold_analysis_summary.json"
)


# ============================================================
# Features
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
# Load test data
# ============================================================

def load_test_data():

    print(
        "\nLoading transaction features..."
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

    classes["target"] = (
        classes["class"] == "1"
    ).astype(int)

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

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    data = features.merge(
        classes[
            [
                "txId",
                "class",
                "target",
            ]
        ],
        on="txId",
        how="left",
        validate="one_to_one",
    )

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
    # Test period only
    # --------------------------------------------------------

    data = data[
        data["time_step"].between(
            40,
            49,
        )
    ].copy()

    # Only labeled test transactions.

    data = data[
        data["class"].isin(
            ["1", "2"]
        )
    ].copy()

    print(
        "\nTest transactions:"
    )

    print(
        f"{len(data):,}"
    )

    # --------------------------------------------------------
    # Verify features
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
            "\nMissing model features:"
        )

        print(
            missing.to_string()
        )

        raise ValueError(
            "Missing values found in "
            "test model features."
        )

    return data


# ============================================================
# Threshold evaluation
# ============================================================

def evaluate_threshold(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
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

    cm = confusion_matrix(
        y_true,
        predictions,
    )

    tn, fp, fn, tp = cm.ravel()

    alert_count = (
        predictions == 1
    ).sum()

    alert_rate = (
        alert_count
        / len(predictions)
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alerts": int(alert_count),
        "alert_rate": float(
            alert_rate
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "GraphGuard AI — "
        "Threshold Analysis"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_test_data()

    print(
        "\nLoading trained model..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Model loaded."
    )

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    X_test = data[
        MODEL_FEATURES
    ]

    y_test = data[
        "target"
    ].to_numpy()

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    pr_auc = (
        average_precision_score(
            y_test,
            probabilities,
        )
    )

    print(
        f"\nTest PR-AUC: "
        f"{pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    results = []

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "\n"
        "Threshold | Precision | Recall | F1 | "
        "Alerts | FP | FN"
    )

    print(
        "-" * 70
    )

    for threshold in thresholds:

        result = evaluate_threshold(
            y_test,
            probabilities,
            threshold,
        )

        results.append(
            result
        )

        print(
            f"{threshold:9.2f} | "
            f"{result['precision']:.4f}    | "
            f"{result['recall']:.4f} | "
            f"{result['f1']:.4f} | "
            f"{result['alerts']:6d} | "
            f"{result['false_positives']:2d} | "
            f"{result['false_negatives']:3d}"
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Best thresholds
    # --------------------------------------------------------

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_precision = results_df.loc[
        results_df["precision"].idxmax()
    ]

    # Best recall subject to at least
    # 80% precision, if possible.

    precision_constrained = (
        results_df[
            results_df["precision"] >= 0.80
        ]
    )

    if len(precision_constrained) > 0:

        best_recall_at_80_precision = (
            precision_constrained.loc[
                precision_constrained[
                    "recall"
                ].idxmax()
            ]
        )

    else:

        best_recall_at_80_precision = None

    # --------------------------------------------------------
    # Print recommendations
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD RECOMMENDATIONS"
    )

    print(
        "=" * 70
    )

    print(
        "\nBest F1 threshold:"
    )

    print(
        f"Threshold: "
        f"{best_f1['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_f1['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_f1['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best_f1['f1']:.4f}"
    )

    print(
        f"False positives: "
        f"{int(best_f1['false_positives'])}"
    )

    print(
        f"False negatives: "
        f"{int(best_f1['false_negatives'])}"
    )

    print(
        "\nHighest precision threshold:"
    )

    print(
        f"Threshold: "
        f"{best_precision['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_precision['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_precision['recall']:.4f}"
    )

    if (
        best_recall_at_80_precision
        is not None
    ):

        print(
            "\nBest recall with precision >= 80%:"
        )

        print(
            f"Threshold: "
            f"{best_recall_at_80_precision['threshold']:.2f}"
        )

        print(
            f"Precision: "
            f"{best_recall_at_80_precision['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{best_recall_at_80_precision['recall']:.4f}"
        )

    # --------------------------------------------------------
    # Save table
    # --------------------------------------------------------

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print(
        "\nThreshold results saved to:"
    )

    print(
        RESULTS_FILE
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "test_pr_auc": float(
            pr_auc
        ),

        "best_f1": {
            key: (
                float(value)
                if isinstance(
                    value,
                    (np.floating, float)
                )
                else int(value)
                if isinstance(
                    value,
                    (np.integer, int)
                )
                else value
            )
            for key, value
            in best_f1.to_dict().items()
        },

        "highest_precision": {
            key: (
                float(value)
                if isinstance(
                    value,
                    (np.floating, float)
                )
                else int(value)
                if isinstance(
                    value,
                    (np.integer, int)
                )
                else value
            )
            for key, value
            in best_precision.to_dict().items()
        },
    }

    if (
        best_recall_at_80_precision
        is not None
    ):

        summary[
            "best_recall_at_80_precision"
        ] = {
            key: (
                float(value)
                if isinstance(
                    value,
                    (np.floating, float)
                )
                else int(value)
                if isinstance(
                    value,
                    (np.integer, int)
                )
                else value
            )
            for key, value
            in best_recall_at_80_precision
            .to_dict()
            .items()
        }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        "Threshold summary saved to:"
    )

    print(
        SUMMARY_FILE
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD ANALYSIS COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()