from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# GraphGuard AI
# XGBoost Feature Importance Analysis
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "phase2"
    / "xgboost.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase2"
)

CSV_FILE = (
    OUTPUT_DIR
    / "xgboost_feature_importance.csv"
)

PLOT_FILE = (
    OUTPUT_DIR
    / "xgboost_feature_importance.png"
)


def main():

    print("\n" + "=" * 70)
    print(
        "GraphGuard AI — "
        "XGBoost Feature Importance"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load saved model
    # --------------------------------------------------------

    print("\nLoading XGBoost model...")

    artifact = joblib.load(
        MODEL_FILE
    )

    model = artifact["model"]

    feature_columns = artifact[
        "feature_columns"
    ]

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Extract importance
    # --------------------------------------------------------

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_df[
        "rank"
    ] = (
        importance_df.index + 1
    )

    importance_df = importance_df[
        [
            "rank",
            "feature",
            "importance",
        ]
    ]

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    importance_df.to_csv(
        CSV_FILE,
        index=False,
    )

    print(
        f"\nSaved full feature ranking to:"
    )

    print(CSV_FILE)

    # --------------------------------------------------------
    # Display top features
    # --------------------------------------------------------

    print(
        "\nTop 20 features:"
    )

    print(
        importance_df
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Plot top 20
    # --------------------------------------------------------

    top = (
        importance_df
        .head(20)
        .sort_values(
            "importance"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top["feature"],
        top["importance"],
    )

    plt.xlabel(
        "XGBoost Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 20 XGBoost Features"
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

    print("=" * 70)


if __name__ == "__main__":
    main()