from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ============================================================
# GraphGuard AI
# Phase 6A — Neighborhood XGBoost Feature Importance
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "phase5"
    / "neighborhood_xgboost.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase5"
)

RANKING_FILE = (
    OUTPUT_DIR
    / "neighborhood_xgboost_feature_importance.csv"
)

PLOT_FILE = (
    OUTPUT_DIR
    / "neighborhood_xgboost_feature_importance.png"
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

ALL_FEATURES = (
    ORIGINAL_FEATURES
    + NEIGHBORHOOD_FEATURES
)


def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "GraphGuard AI — "
        "Neighborhood XGBoost Feature Importance"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading Neighborhood XGBoost model..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    importances = (
        model.feature_importances_
    )

    if len(importances) != len(
        ALL_FEATURES
    ):

        raise ValueError(
            f"Feature count mismatch. "
            f"Model has {len(importances)} "
            f"features but expected "
            f"{len(ALL_FEATURES)}."
        )

    print(
        f"Total features: "
        f"{len(importances)}"
    )

    # --------------------------------------------------------
    # Create ranking
    # --------------------------------------------------------

    ranking = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "importance": importances,
        }
    )

    ranking[
        "group"
    ] = ranking[
        "feature"
    ].apply(
        lambda x:
            "neighborhood"
            if x in NEIGHBORHOOD_FEATURES
            else "original"
    )

    ranking = ranking.sort_values(
        "importance",
        ascending=False,
    ).reset_index(
        drop=True
    )

    ranking[
        "rank"
    ] = ranking.index + 1

    ranking = ranking[
        [
            "rank",
            "feature",
            "importance",
            "group",
        ]
    ]

    # --------------------------------------------------------
    # Save full ranking
    # --------------------------------------------------------

    ranking.to_csv(
        RANKING_FILE,
        index=False,
    )

    print(
        "\nFull ranking saved to:"
    )

    print(
        RANKING_FILE
    )

    # --------------------------------------------------------
    # Top 20
    # --------------------------------------------------------

    print(
        "\nTop 20 features:"
    )

    print(
        ranking.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Neighborhood feature importance
    # --------------------------------------------------------

    neighborhood_ranking = ranking[
        ranking["group"]
        == "neighborhood"
    ].copy()

    print(
        "\n" + "=" * 70
    )

    print(
        "NEIGHBORHOOD FEATURE IMPORTANCE"
    )

    print(
        "=" * 70
    )

    print(
        neighborhood_ranking.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Group contribution
    # --------------------------------------------------------

    original_importance = ranking[
        ranking["group"] == "original"
    ]["importance"].sum()

    neighborhood_importance = ranking[
        ranking["group"] == "neighborhood"
    ]["importance"].sum()

    total_importance = (
        original_importance
        + neighborhood_importance
    )

    original_percentage = (
        original_importance
        / total_importance
        * 100
    )

    neighborhood_percentage = (
        neighborhood_importance
        / total_importance
        * 100
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE GROUP CONTRIBUTION"
    )

    print(
        "=" * 70
    )

    print(
        f"Original feature importance: "
        f"{original_importance:.6f}"
    )

    print(
        f"Neighborhood feature importance: "
        f"{neighborhood_importance:.6f}"
    )

    print(
        f"Neighborhood feature importance (%): "
        f"{neighborhood_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Plot top 20
    # --------------------------------------------------------

    top = ranking.head(20).sort_values(
        "importance"
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top["feature"],
        top["importance"],
    )

    plt.xlabel(
        "XGBoost Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 20 Features — "
        "Leakage-Safe Neighborhood XGBoost"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=150,
    )

    plt.close()

    print(
        "\nPlot saved to:"
    )

    print(
        PLOT_FILE
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE IMPORTANCE ANALYSIS COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()