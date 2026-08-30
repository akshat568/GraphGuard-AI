from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# GraphGuard AI
# Phase 5 — Leakage-Safe Neighborhood Risk Features
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

FEATURES_FILE = (
    DATA_DIR / "elliptic_txs_features.csv"
)

CLASSES_FILE = (
    DATA_DIR / "elliptic_txs_classes.csv"
)

EDGES_FILE = (
    DATA_DIR / "elliptic_txs_edgelist.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "phase2"
    / "xgboost.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase5"
)

RISK_FEATURES_FILE = (
    OUTPUT_DIR
    / "neighborhood_risk_features.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "neighborhood_risk_summary.json"
)


# ============================================================
# Original model features
# ============================================================

ORIGINAL_FEATURES = [
    f"feature_{i}"
    for i in range(1, 166)
]


# ============================================================
# Load transaction features
# ============================================================

def load_features():

    print("\nLoading transaction features...")

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
        f"Transactions loaded: "
        f"{len(features):,}"
    )

    return features


# ============================================================
# Load edges
# ============================================================

def load_edges():

    print("\nLoading transaction graph...")

    edges = pd.read_csv(
        EDGES_FILE
    )

    print(
        f"Edges loaded: "
        f"{len(edges):,}"
    )

    return edges


# ============================================================
# Generate XGBoost risk predictions
# ============================================================

def generate_risk_predictions(
    features,
):

    print(
        "\nLoading baseline XGBoost model..."
    )

    saved = joblib.load(
        MODEL_FILE
    )

    # Handle both the older direct-model format
    # and the saved dictionary format.
    if isinstance(saved, dict):

        if "model" in saved:
            model = saved["model"]
        else:
            model = saved

        if "feature_columns" in saved:
            model_features = saved[
                "feature_columns"
            ]
        else:
            model_features = ORIGINAL_FEATURES

    else:

        model = saved
        model_features = ORIGINAL_FEATURES

    print(
        f"Model features: "
        f"{len(model_features)}"
    )

    print(
        "\nGenerating transaction risk scores..."
    )

    X = features[
        model_features
    ]

    risk = model.predict_proba(
        X
    )[:, 1]

    predictions = pd.DataFrame(
        {
            "txId": features["txId"],
            "time_step": features["time_step"],
            "model_risk": risk,
        }
    )

    print(
        "Risk prediction completed."
    )

    print(
        f"Minimum risk: "
        f"{risk.min():.6f}"
    )

    print(
        f"Maximum risk: "
        f"{risk.max():.6f}"
    )

    print(
        f"Mean risk: "
        f"{risk.mean():.6f}"
    )

    return predictions


# ============================================================
# Calculate neighborhood risk
# ============================================================

def calculate_neighborhood_risk(
    predictions,
    edges,
):

    print(
        "\nCalculating neighborhood risk..."
    )

    # --------------------------------------------------------
    # Create risk lookup
    # --------------------------------------------------------

    risk_lookup = predictions.set_index(
        "txId"
    )["model_risk"]

    # --------------------------------------------------------
    # Source and target risk
    # --------------------------------------------------------

    edge_data = edges.copy()

    edge_data[
        "source_risk"
    ] = edge_data[
        "txId1"
    ].map(
        risk_lookup
    )

    edge_data[
        "target_risk"
    ] = edge_data[
        "txId2"
    ].map(
        risk_lookup
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    missing_source = (
        edge_data["source_risk"]
        .isna()
        .sum()
    )

    missing_target = (
        edge_data["target_risk"]
        .isna()
        .sum()
    )

    if missing_source != 0:
        raise ValueError(
            "Missing source risk values: "
            f"{missing_source}"
        )

    if missing_target != 0:
        raise ValueError(
            "Missing target risk values: "
            f"{missing_target}"
        )

    # --------------------------------------------------------
    # Incoming neighborhood
    #
    # For transaction T:
    # incoming neighbors are txId1
    # where txId2 == T
    # --------------------------------------------------------

    incoming = (
        edge_data
        .groupby("txId2")[
            "source_risk"
        ]
        .agg(
            incoming_mean_risk="mean",
            incoming_max_risk="max",
            incoming_median_risk="median",
            incoming_std_risk="std",
        )
    )

    incoming_count = (
        edge_data
        .groupby("txId2")
        .size()
        .rename(
            "incoming_risk_neighbor_count"
        )
    )

    incoming = incoming.join(
        incoming_count
    )

    # --------------------------------------------------------
    # Outgoing neighborhood
    #
    # For transaction T:
    # outgoing neighbors are txId2
    # where txId1 == T
    # --------------------------------------------------------

    outgoing = (
        edge_data
        .groupby("txId1")[
            "target_risk"
        ]
        .agg(
            outgoing_mean_risk="mean",
            outgoing_max_risk="max",
            outgoing_median_risk="median",
            outgoing_std_risk="std",
        )
    )

    outgoing_count = (
        edge_data
        .groupby("txId1")
        .size()
        .rename(
            "outgoing_risk_neighbor_count"
        )
    )

    outgoing = outgoing.join(
        outgoing_count
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    risk_features = predictions[
        [
            "txId",
            "time_step",
            "model_risk",
        ]
    ].copy()

    risk_features = risk_features.merge(
        incoming,
        left_on="txId",
        right_index=True,
        how="left",
    )

    risk_features = risk_features.merge(
        outgoing,
        left_on="txId",
        right_index=True,
        how="left",
    )

    # --------------------------------------------------------
    # Replace missing values for nodes without neighbors
    # --------------------------------------------------------

    mean_columns = [
        "incoming_mean_risk",
        "incoming_max_risk",
        "incoming_median_risk",
        "incoming_std_risk",
        "outgoing_mean_risk",
        "outgoing_max_risk",
        "outgoing_median_risk",
        "outgoing_std_risk",
    ]

    count_columns = [
        "incoming_risk_neighbor_count",
        "outgoing_risk_neighbor_count",
    ]

    risk_features[
        mean_columns
    ] = risk_features[
        mean_columns
    ].fillna(0.0)

    risk_features[
        count_columns
    ] = risk_features[
        count_columns
    ].fillna(0).astype(
        "int32"
    )

    # --------------------------------------------------------
    # Combined neighborhood statistics
    # --------------------------------------------------------

    combined_mean = (
        risk_features[
            [
                "incoming_mean_risk",
                "outgoing_mean_risk",
            ]
        ]
        .replace(
            0,
            np.nan,
        )
        .mean(
            axis=1
        )
        .fillna(0.0)
    )

    risk_features[
        "neighborhood_mean_risk"
    ] = combined_mean

    risk_features[
        "neighborhood_max_risk"
    ] = risk_features[
        [
            "incoming_max_risk",
            "outgoing_max_risk",
        ]
    ].max(
        axis=1
    )

    risk_features[
        "neighborhood_median_risk"
    ] = risk_features[
        [
            "incoming_median_risk",
            "outgoing_median_risk",
        ]
    ].max(
        axis=1
    )

    risk_features[
        "neighborhood_neighbor_count"
    ] = (
        risk_features[
            "incoming_risk_neighbor_count"
        ]
        +
        risk_features[
            "outgoing_risk_neighbor_count"
        ]
    )

    # --------------------------------------------------------
    # High-risk neighbor counts
    # --------------------------------------------------------

    high_risk_threshold = 0.5

    incoming_high_risk = (
        edge_data[
            "source_risk"
        ] >= high_risk_threshold
    )

    outgoing_high_risk = (
        edge_data[
            "target_risk"
        ] >= high_risk_threshold
    )

    incoming_high_count = (
        edge_data.loc[
            incoming_high_risk
        ]
        .groupby("txId2")
        .size()
        .rename(
            "incoming_high_risk_neighbors"
        )
    )

    outgoing_high_count = (
        edge_data.loc[
            outgoing_high_risk
        ]
        .groupby("txId1")
        .size()
        .rename(
            "outgoing_high_risk_neighbors"
        )
    )

    risk_features = risk_features.merge(
        incoming_high_count,
        left_on="txId",
        right_index=True,
        how="left",
    )

    risk_features = risk_features.merge(
        outgoing_high_count,
        left_on="txId",
        right_index=True,
        how="left",
    )

    risk_features[
        [
            "incoming_high_risk_neighbors",
            "outgoing_high_risk_neighbors",
        ]
    ] = risk_features[
        [
            "incoming_high_risk_neighbors",
            "outgoing_high_risk_neighbors",
        ]
    ].fillna(0).astype(
        "int32"
    )

    risk_features[
        "high_risk_neighbor_count"
    ] = (
        risk_features[
            "incoming_high_risk_neighbors"
        ]
        +
        risk_features[
            "outgoing_high_risk_neighbors"
        ]
    )

    # --------------------------------------------------------
    # High-risk neighbor fraction
    # --------------------------------------------------------

    risk_features[
        "high_risk_neighbor_fraction"
    ] = (
        risk_features[
            "high_risk_neighbor_count"
        ]
        /
        risk_features[
            "neighborhood_neighbor_count"
        ].replace(
            0,
            1,
        )
    )

    # --------------------------------------------------------
    # Risk difference
    # --------------------------------------------------------

    risk_features[
        "neighborhood_vs_self_risk"
    ] = (
        risk_features[
            "neighborhood_mean_risk"
        ]
        -
        risk_features[
            "model_risk"
        ]
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print(
        "\nValidating neighborhood features..."
    )

    feature_columns = [
        column
        for column in risk_features.columns
        if column not in [
            "txId",
            "time_step",
        ]
    ]

    if (
        risk_features[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
        != 0
    ):

        raise ValueError(
            "NaN values found in "
            "neighborhood risk features."
        )

    if (
        risk_features["txId"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate transaction IDs found."
        )

    print(
        "Neighborhood feature validation passed."
    )

    return risk_features


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "GraphGuard AI — "
        "Neighborhood Risk Feature Engineering"
    )

    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features = load_features()

    edges = load_edges()

    predictions = generate_risk_predictions(
        features
    )

    risk_features = (
        calculate_neighborhood_risk(
            predictions,
            edges,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\nSaving neighborhood risk features..."
    )

    risk_features.to_csv(
        RISK_FEATURES_FILE,
        index=False,
    )

    print(
        f"Saved to:\n"
        f"{RISK_FEATURES_FILE}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in risk_features.columns
        if column not in [
            "txId",
            "time_step",
        ]
    ]

    summary = {
        "transactions": int(
            len(risk_features)
        ),

        "risk_features": int(
            len(feature_columns)
        ),

        "features": feature_columns,

        "high_risk_threshold": 0.5,
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

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\nGenerated features:"
    )

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )

    print(
        "\nSample:"
    )

    print(
        risk_features.head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)

    print(
        "NEIGHBORHOOD RISK FEATURE ENGINEERING COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()