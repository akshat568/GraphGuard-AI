from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# GraphGuard AI
# Graph-Enhanced XGBoost Feature Importance
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "phase4"
    / "graph_xgboost.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase4"
)

RANKING_FILE = (
    OUTPUT_DIR
    / "graph_xgboost_feature_importance.csv"
)

PLOT_FILE = (
    OUTPUT_DIR
    / "graph_xgboost_feature_importance.png"
)


# ============================================================
# Feature groups
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


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "GraphGuard AI — "
        "Graph-Enhanced XGBoost Feature Importance"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading graph-enhanced XGBoost model...")

    saved = joblib.load(
        MODEL_FILE
    )

    model = saved["model"]

    feature_columns = saved[
        "feature_columns"
    ]

    print(
        f"Total features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Importance
    # --------------------------------------------------------

    importances = model.feature_importances_

    ranking = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    )

    ranking = ranking.sort_values(
        "importance",
        ascending=False,
    ).reset_index(
        drop=True
    )

    ranking["rank"] = (
        ranking.index + 1
    )

    ranking = ranking[
        [
            "rank",
            "feature",
            "importance",
        ]
    ]

    # --------------------------------------------------------
    # Feature group
    # --------------------------------------------------------

    ranking["group"] = ranking[
        "feature"
    ].apply(
        lambda x:
        "graph"
        if x in GRAPH_FEATURES
        else "original"
    )

    # --------------------------------------------------------
    # Save complete ranking
    # --------------------------------------------------------

    ranking.to_csv(
        RANKING_FILE,
        index=False,
    )

    print(
        f"\nFull ranking saved to:"
    )

    print(RANKING_FILE)

    # --------------------------------------------------------
    # Top 20
    # --------------------------------------------------------

    print("\nTop 20 features:")

    print(
        ranking.head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Graph feature importance
    # --------------------------------------------------------

    graph_ranking = ranking[
        ranking["group"] == "graph"
    ].copy()

    graph_ranking = graph_ranking.sort_values(
        "importance",
        ascending=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "GRAPH FEATURE IMPORTANCE"
    )

    print(
        "=" * 70
    )

    print(
        graph_ranking.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Importance totals
    # --------------------------------------------------------

    original_importance = ranking[
        ranking["group"] == "original"
    ]["importance"].sum()

    graph_importance = ranking[
        ranking["group"] == "graph"
    ]["importance"].sum()

    total_importance = (
        original_importance
        + graph_importance
    )

    graph_percentage = (
        graph_importance
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
        f"Graph feature importance: "
        f"{graph_importance:.6f}"
    )

    print(
        f"Graph feature importance (%): "
        f"{graph_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Plot top 20
    # --------------------------------------------------------

    top = ranking.head(20).copy()

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top["feature"][::-1],
        top["importance"][::-1],
    )

    plt.xlabel(
        "XGBoost Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Graph-Enhanced XGBoost — "
        "Top 20 Features"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=150,
    )

    plt.close()

    print(
        f"\nPlot saved to:"
    )

    print(PLOT_FILE)

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