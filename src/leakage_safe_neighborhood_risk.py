from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# GraphGuard AI
# Phase 5 — Leakage-Safe Neighborhood Risk
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"
EDGES_FILE = DATA_DIR / "elliptic_txs_edgelist.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase5"

OUTPUT_FILE = (
    OUTPUT_DIR / "leakage_safe_neighborhood_risk.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "leakage_safe_neighborhood_risk_summary.json"
)


# ============================================================
# Configuration
# ============================================================

FEATURE_COLUMNS = [
    f"feature_{i}"
    for i in range(1, 166)
]

# Temporal folds for out-of-fold training predictions.
# Every validation fold is predicted by a model that did not
# train on that fold's labels.
TRAIN_FOLDS = [
    (11, 15),
    (16, 20),
    (21, 27),
    (28, 34),
]

TRAIN_START = 1
TRAIN_END = 34

VALIDATION_START = 35
VALIDATION_END = 39

TEST_START = 40
TEST_END = 49


# ============================================================
# Load data
# ============================================================

def load_data():

    print("\nLoading transaction features...")

    columns = (
        ["txId", "time_step"]
        + FEATURE_COLUMNS
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

    print("\nLoading classes...")

    classes = pd.read_csv(
        CLASSES_FILE
    )

    classes["target"] = (
        classes["class"] == "1"
    ).astype(int)

    print(
        f"Class records loaded: "
        f"{len(classes):,}"
    )

    print("\nLoading edges...")

    edges = pd.read_csv(
        EDGES_FILE
    )

    print(
        f"Edges loaded: "
        f"{len(edges):,}"
    )

    return features, classes, edges


# ============================================================
# Train one XGBoost model
# ============================================================

def train_xgboost(train_data):

    X_train = train_data[
        FEATURE_COLUMNS
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

    if positives == 0:
        raise ValueError(
            "Training fold contains no positive examples."
        )

    scale_pos_weight = (
        negatives / positives
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

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    return model


# ============================================================
# Generate temporal out-of-fold predictions
# ============================================================

def generate_training_predictions(
    data,
):

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATING LEAKAGE-SAFE TRAINING PREDICTIONS"
    )

    print(
        "=" * 70
    )

    predictions = {}

    for fold_number, (
        validation_start,
        validation_end,
    ) in enumerate(
        TRAIN_FOLDS,
        start=1,
    ):

        # ----------------------------------------------------
        # Training data comes only from earlier timesteps.
        # ----------------------------------------------------

        if fold_number == 1:

            train_end = validation_start - 1

        else:

            train_end = validation_start - 1

        train_data = data[
            data["time_step"].between(
                TRAIN_START,
                train_end,
            )
            &
            data["class"].isin(
                ["1", "2"]
            )
        ]

        validation_data = data[
            data["time_step"].between(
                validation_start,
                validation_end,
            )
            &
            data["class"].isin(
                ["1", "2"]
            )
        ]

        print(
            f"\nFold {fold_number}"
        )

        print(
            f"Training: "
            f"timesteps {TRAIN_START}-{train_end}"
        )

        print(
            f"Prediction: "
            f"timesteps "
            f"{validation_start}-{validation_end}"
        )

        print(
            f"Training samples: "
            f"{len(train_data):,}"
        )

        print(
            f"Prediction samples: "
            f"{len(validation_data):,}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model = train_xgboost(
            train_data
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                validation_data[
                    FEATURE_COLUMNS
                ]
            )[:, 1]
        )

        for tx_id, probability in zip(
            validation_data["txId"],
            probabilities,
        ):

            predictions[
                int(tx_id)
            ] = float(
                probability
            )

        print(
            "Fold completed."
        )

    return predictions


# ============================================================
# Generate validation/test predictions
# ============================================================

def generate_future_predictions(
    data,
):

    print(
        "\n" + "=" * 70
    )

    print(
        "GENERATING FUTURE-TIMESTEP PREDICTIONS"
    )

    print(
        "=" * 70
    )

    predictions = {}

    # --------------------------------------------------------
    # Validation: train only on 1-34
    # --------------------------------------------------------

    train_data = data[
        data["time_step"].between(
            TRAIN_START,
            TRAIN_END,
        )
        &
        data["class"].isin(
            ["1", "2"]
        )
    ]

    validation_data = data[
        data["time_step"].between(
            VALIDATION_START,
            VALIDATION_END,
        )
    ]

    print(
        "\nValidation prediction"
    )

    print(
        "Training: timesteps 1-34"
    )

    print(
        "Prediction: timesteps 35-39"
    )

    model = train_xgboost(
        train_data
    )

    probabilities = (
        model.predict_proba(
            validation_data[
                FEATURE_COLUMNS
            ]
        )[:, 1]
    )

    for tx_id, probability in zip(
        validation_data["txId"],
        probabilities,
    ):

        predictions[
            int(tx_id)
        ] = float(
            probability
        )

    print(
        "Validation predictions completed."
    )

    # --------------------------------------------------------
    # Test: train on labeled data through 39
    # --------------------------------------------------------

    test_train_data = data[
        data["time_step"].between(
            TRAIN_START,
            VALIDATION_END,
        )
        &
        data["class"].isin(
            ["1", "2"]
        )
    ]

    test_data = data[
        data["time_step"].between(
            TEST_START,
            TEST_END,
        )
    ]

    print(
        "\nTest prediction"
    )

    print(
        "Training: timesteps 1-39"
    )

    print(
        "Prediction: timesteps 40-49"
    )

    test_model = train_xgboost(
        test_train_data
    )

    test_probabilities = (
        test_model.predict_proba(
            test_data[
                FEATURE_COLUMNS
            ]
        )[:, 1]
    )

    for tx_id, probability in zip(
        test_data["txId"],
        test_probabilities,
    ):

        predictions[
            int(tx_id)
        ] = float(
            probability
        )

    print(
        "Test predictions completed."
    )

    return predictions


# ============================================================
# Build prediction table
# ============================================================

def build_prediction_table(
    features,
    training_predictions,
    future_predictions,
):

    all_predictions = {}

    all_predictions.update(
        training_predictions
    )

    all_predictions.update(
        future_predictions
    )

    result = features[
        ["txId", "time_step"]
    ].copy()

    result[
        "model_risk"
    ] = result["txId"].map(
        all_predictions
    )

    # Unknown transactions that are not part of the
    # labeled/evaluation periods remain without a prediction.
    # We deliberately do not invent labels or predictions for them.

    print(
        "\nPrediction coverage:"
    )

    print(
        f"Predicted transactions: "
        f"{result['model_risk'].notna().sum():,}"
    )

    print(
        f"Without prediction: "
        f"{result['model_risk'].isna().sum():,}"
    )

    return result


# ============================================================
# Neighborhood aggregation
# ============================================================

def calculate_neighborhood_features(
    predictions,
    edges,
):

    print(
        "\n" + "=" * 70
    )

    print(
        "CALCULATING LEAKAGE-SAFE NEIGHBORHOOD FEATURES"
    )

    print(
        "=" * 70
    )

    risk_lookup = predictions.set_index(
        "txId"
    )["model_risk"]

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
    # IMPORTANT:
    # We only aggregate neighbors for which a valid,
    # leakage-safe prediction exists.
    # --------------------------------------------------------

    valid_incoming = edge_data[
        edge_data["source_risk"].notna()
    ]

    valid_outgoing = edge_data[
        edge_data["target_risk"].notna()
    ]

    # --------------------------------------------------------
    # Incoming neighbor risk
    # --------------------------------------------------------

    incoming = (
        valid_incoming
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
        valid_incoming
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
    # Outgoing neighbor risk
    # --------------------------------------------------------

    outgoing = (
        valid_outgoing
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
        valid_outgoing
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
    # Base result
    # --------------------------------------------------------

    result = predictions.copy()

    result = result.merge(
        incoming,
        left_on="txId",
        right_index=True,
        how="left",
    )

    result = result.merge(
        outgoing,
        left_on="txId",
        right_index=True,
        how="left",
    )

    # --------------------------------------------------------
    # Fill nodes without predicted neighbors
    # --------------------------------------------------------

    risk_columns = [
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

    result[
        risk_columns
    ] = result[
        risk_columns
    ].fillna(0.0)

    result[
        count_columns
    ] = result[
        count_columns
    ].fillna(0).astype(
        "int32"
    )

    # --------------------------------------------------------
    # Combined neighborhood risk
    # --------------------------------------------------------

    result[
        "neighborhood_mean_risk"
    ] = result[
        [
            "incoming_mean_risk",
            "outgoing_mean_risk",
        ]
    ].mean(
        axis=1
    )

    result[
        "neighborhood_max_risk"
    ] = result[
        [
            "incoming_max_risk",
            "outgoing_max_risk",
        ]
    ].max(
        axis=1
    )

    result[
        "neighborhood_median_risk"
    ] = result[
        [
            "incoming_median_risk",
            "outgoing_median_risk",
        ]
    ].max(
        axis=1
    )

    result[
        "neighborhood_neighbor_count"
    ] = (
        result[
            "incoming_risk_neighbor_count"
        ]
        +
        result[
            "outgoing_risk_neighbor_count"
        ]
    )

    # --------------------------------------------------------
    # High-risk neighbor counts
    # --------------------------------------------------------

    threshold = 0.5

    incoming_high = valid_incoming[
        valid_incoming[
            "source_risk"
        ] >= threshold
    ]

    outgoing_high = valid_outgoing[
        valid_outgoing[
            "target_risk"
        ] >= threshold
    ]

    incoming_high_count = (
        incoming_high
        .groupby("txId2")
        .size()
        .rename(
            "incoming_high_risk_neighbors"
        )
    )

    outgoing_high_count = (
        outgoing_high
        .groupby("txId1")
        .size()
        .rename(
            "outgoing_high_risk_neighbors"
        )
    )

    result = result.merge(
        incoming_high_count,
        left_on="txId",
        right_index=True,
        how="left",
    )

    result = result.merge(
        outgoing_high_count,
        left_on="txId",
        right_index=True,
        how="left",
    )

    result[
        [
            "incoming_high_risk_neighbors",
            "outgoing_high_risk_neighbors",
        ]
    ] = result[
        [
            "incoming_high_risk_neighbors",
            "outgoing_high_risk_neighbors",
        ]
    ].fillna(0).astype(
        "int32"
    )

    result[
        "high_risk_neighbor_count"
    ] = (
        result[
            "incoming_high_risk_neighbors"
        ]
        +
        result[
            "outgoing_high_risk_neighbors"
        ]
    )

    result[
        "high_risk_neighbor_fraction"
    ] = (
        result[
            "high_risk_neighbor_count"
        ]
        /
        result[
            "neighborhood_neighbor_count"
        ].replace(
            0,
            1,
        )
    )

    # --------------------------------------------------------
    # Difference between self and neighborhood risk
    # --------------------------------------------------------

    result[
        "neighborhood_vs_self_risk"
    ] = (
        result[
            "neighborhood_mean_risk"
        ]
        -
        result[
            "model_risk"
        ]
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    # model_risk is intentionally allowed to be NaN for transactions
    # outside the leakage-safe prediction periods. These are NOT zero-risk
    # transactions; they simply have no leakage-safe model prediction.
    # All neighborhood aggregation features must be fully numeric.

    feature_columns = [
        column
        for column in result.columns
        if column not in ["txId", "time_step"]
    ]

    neighborhood_columns = [
        column
        for column in feature_columns
        if column not in ["model_risk", "neighborhood_vs_self_risk"]
    ]

    remaining_neighborhood_nan = (
        result[neighborhood_columns]
        .isna()
        .sum()
        .sum()
    )

    if remaining_neighborhood_nan != 0:
        raise ValueError(
            "Unexpected NaN values remain in neighborhood features."
        )

    # neighborhood_vs_self_risk is undefined when model_risk is unavailable.
    # Keep it as NaN in exactly those rows.
    result.loc[
        result["model_risk"].isna(),
        "neighborhood_vs_self_risk"
    ] = np.nan

    print("\nModel-risk coverage:")
    print(
        f"Available predictions: "
        f"{result['model_risk'].notna().sum():,}"
    )
    print(
        f"Unavailable predictions: "
        f"{result['model_risk'].isna().sum():,}"
    )

    numeric_columns = result[feature_columns].select_dtypes(
        include=[np.number]
    )

    if np.isinf(numeric_columns.to_numpy()).sum() != 0:
        raise ValueError(
            "Infinite values found in leakage-safe neighborhood features."
        )

    print("Neighborhood feature validation passed.")

    return result


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "GraphGuard AI — "
        "Leakage-Safe Neighborhood Risk Pipeline"
    )

    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features, classes, edges = load_data()

    # --------------------------------------------------------
    # Merge labels
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

    # --------------------------------------------------------
    # Training OOF predictions
    # --------------------------------------------------------

    training_predictions = (
        generate_training_predictions(
            data
        )
    )

    # --------------------------------------------------------
    # Future predictions
    # --------------------------------------------------------

    future_predictions = (
        generate_future_predictions(
            data
        )
    )

    # --------------------------------------------------------
    # Prediction table
    # --------------------------------------------------------

    predictions = build_prediction_table(
        features,
        training_predictions,
        future_predictions,
    )

    # --------------------------------------------------------
    # Neighborhood aggregation
    # --------------------------------------------------------

    result = calculate_neighborhood_features(
        predictions,
        edges,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved leakage-safe features to:"
    )

    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    risk_feature_columns = [
        column
        for column in result.columns
        if column not in [
            "txId",
            "time_step",
        ]
    ]

    summary = {
        "transactions": int(
            len(result)
        ),

        "feature_count": int(
            len(risk_feature_columns)
        ),

        "features": risk_feature_columns,

        "training_prediction_folds": [
            {
                "train":
                    f"{TRAIN_START}-{end}",
                "predict":
                    f"{start}-{end}",
            }
            for start, end in TRAIN_FOLDS
        ],

        "validation_prediction": {
            "train": "1-34",
            "predict": "35-39",
        },

        "test_prediction": {
            "train": "1-39",
            "predict": "40-49",
        },

        "high_risk_threshold": 0.5,

        "predicted_transactions": int(
            result[
                "model_risk"
            ].notna().sum()
        ),

        "unpredicted_transactions": int(
            result[
                "model_risk"
            ].isna().sum()
        ),
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
        f"Summary saved to:"
    )

    print(SUMMARY_FILE)

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print(
        "\nGenerated feature columns:"
    )

    for feature in risk_feature_columns:
        print(
            f"  - {feature}"
        )

    print(
        "\nSample:"
    )

    print(
        result.head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)

    print(
        "LEAKAGE-SAFE NEIGHBORHOOD RISK COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()