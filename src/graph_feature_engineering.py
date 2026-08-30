from pathlib import Path
import json

import pandas as pd


# ============================================================
# GraphGuard AI
# Phase 4 — Graph Structural Feature Engineering
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
EDGES_FILE = DATA_DIR / "elliptic_txs_edgelist.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase4"

GRAPH_FEATURES_FILE = OUTPUT_DIR / "graph_features.csv"
SUMMARY_FILE = OUTPUT_DIR / "graph_feature_summary.json"


# ============================================================
# Load transaction metadata
# ============================================================

def load_nodes():

    print("\nLoading transaction metadata...")

    columns = ["txId", "time_step"]

    nodes = pd.read_csv(
        FEATURES_FILE,
        header=None,
        usecols=[0, 1],
        names=columns,
    )

    print(
        f"Transactions loaded: {len(nodes):,}"
    )

    return nodes


# ============================================================
# Load edges
# ============================================================

def load_edges():

    print("\nLoading transaction edges...")

    edges = pd.read_csv(
        EDGES_FILE
    )

    print(
        f"Edges loaded: {len(edges):,}"
    )

    return edges


# ============================================================
# Build structural features
# ============================================================

def create_graph_features(
    nodes,
    edges,
):

    print("\nCreating graph structural features...")

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    graph_features = nodes[
        ["txId", "time_step"]
    ].copy()

    # --------------------------------------------------------
    # In-degree
    # --------------------------------------------------------

    print("Calculating in-degree...")

    in_degree = (
        edges
        .groupby("txId2")
        .size()
        .rename("in_degree")
    )

    graph_features = graph_features.merge(
        in_degree,
        left_on="txId",
        right_index=True,
        how="left",
    )

    # --------------------------------------------------------
    # Out-degree
    # --------------------------------------------------------

    print("Calculating out-degree...")

    out_degree = (
        edges
        .groupby("txId1")
        .size()
        .rename("out_degree")
    )

    graph_features = graph_features.merge(
        out_degree,
        left_on="txId",
        right_index=True,
        how="left",
    )

    # --------------------------------------------------------
    # Fill isolated nodes
    # --------------------------------------------------------

    graph_features[
        "in_degree"
    ] = graph_features[
        "in_degree"
    ].fillna(0).astype("int32")

    graph_features[
        "out_degree"
    ] = graph_features[
        "out_degree"
    ].fillna(0).astype("int32")

    # --------------------------------------------------------
    # Total degree
    # --------------------------------------------------------

    graph_features[
        "total_degree"
    ] = (
        graph_features["in_degree"]
        + graph_features["out_degree"]
    )

    # --------------------------------------------------------
    # Degree imbalance
    # --------------------------------------------------------

    graph_features[
        "degree_imbalance"
    ] = (
        graph_features["out_degree"]
        - graph_features["in_degree"]
    )

    # --------------------------------------------------------
    # Absolute degree imbalance
    # --------------------------------------------------------

    graph_features[
        "absolute_degree_imbalance"
    ] = (
        graph_features[
            "degree_imbalance"
        ].abs()
    )

    # --------------------------------------------------------
    # Unique neighbors
    # --------------------------------------------------------

    print(
        "Calculating unique neighbor count..."
    )

    outgoing_neighbors = (
        edges[
            ["txId1", "txId2"]
        ]
        .drop_duplicates()
        .groupby("txId1")
        .size()
        .rename("unique_out_neighbors")
    )

    incoming_neighbors = (
        edges[
            ["txId1", "txId2"]
        ]
        .drop_duplicates()
        .groupby("txId2")
        .size()
        .rename("unique_in_neighbors")
    )

    graph_features = graph_features.merge(
        outgoing_neighbors,
        left_on="txId",
        right_index=True,
        how="left",
    )

    graph_features = graph_features.merge(
        incoming_neighbors,
        left_on="txId",
        right_index=True,
        how="left",
    )

    graph_features[
        "unique_out_neighbors"
    ] = (
        graph_features[
            "unique_out_neighbors"
        ]
        .fillna(0)
        .astype("int32")
    )

    graph_features[
        "unique_in_neighbors"
    ] = (
        graph_features[
            "unique_in_neighbors"
        ]
        .fillna(0)
        .astype("int32")
    )

    # --------------------------------------------------------
    # Total unique neighbors
    # --------------------------------------------------------

    graph_features[
        "unique_neighbors"
    ] = (
        graph_features[
            "unique_out_neighbors"
        ]
        + graph_features[
            "unique_in_neighbors"
        ]
    )

    # --------------------------------------------------------
    # Has incoming / outgoing transaction
    # --------------------------------------------------------

    graph_features[
        "has_incoming"
    ] = (
        graph_features[
            "in_degree"
        ] > 0
    ).astype("int8")

    graph_features[
        "has_outgoing"
    ] = (
        graph_features[
            "out_degree"
        ] > 0
    ).astype("int8")

    # --------------------------------------------------------
    # Source / sink indicators
    # --------------------------------------------------------

    graph_features[
        "is_source"
    ] = (
        (
            graph_features[
                "out_degree"
            ] > 0
        )
        &
        (
            graph_features[
                "in_degree"
            ] == 0
        )
    ).astype("int8")

    graph_features[
        "is_sink"
    ] = (
        (
            graph_features[
                "in_degree"
            ] > 0
        )
        &
        (
            graph_features[
                "out_degree"
            ] == 0
        )
    ).astype("int8")

    # --------------------------------------------------------
    # Isolated node
    # --------------------------------------------------------

    graph_features[
        "is_isolated"
    ] = (
        graph_features[
            "total_degree"
        ] == 0
    ).astype("int8")

    # --------------------------------------------------------
    # Degree ratios
    # --------------------------------------------------------

    graph_features[
        "in_degree_ratio"
    ] = (
        graph_features["in_degree"]
        /
        graph_features["total_degree"].replace(
            0,
            1,
        )
    )

    graph_features[
        "out_degree_ratio"
    ] = (
        graph_features["out_degree"]
        /
        graph_features["total_degree"].replace(
            0,
            1,
        )
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print(
        "\nValidating graph features..."
    )

    assert (
        len(graph_features)
        == len(nodes)
    )

    assert (
        graph_features["txId"]
        .is_unique
    )

    feature_columns = [
        column
        for column in graph_features.columns
        if column not in [
            "txId",
            "time_step",
        ]
    ]

    assert len(feature_columns) > 0

    assert (
        graph_features[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
        == 0
    )

    print(
        "Graph feature validation passed."
    )

    return graph_features


# ============================================================
# Summary
# ============================================================

def create_summary(
    graph_features,
):

    feature_columns = [
        column
        for column in graph_features.columns
        if column not in [
            "txId",
            "time_step",
        ]
    ]

    summary = {
        "transactions": int(
            len(graph_features)
        ),

        "graph_features": int(
            len(feature_columns)
        ),

        "features": feature_columns,

        "statistics": {},
    }

    for feature in feature_columns:

        series = graph_features[
            feature
        ]

        summary[
            "statistics"
        ][feature] = {
            "min": float(
                series.min()
            ),

            "max": float(
                series.max()
            ),

            "mean": float(
                series.mean()
            ),

            "median": float(
                series.median()
            ),
        }

    return summary


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "GraphGuard AI — "
        "Graph Structural Feature Engineering"
    )

    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    nodes = load_nodes()

    edges = load_edges()

    graph_features = create_graph_features(
        nodes,
        edges,
    )

    # --------------------------------------------------------
    # Save graph features
    # --------------------------------------------------------

    print(
        "\nSaving graph features..."
    )

    graph_features.to_csv(
        GRAPH_FEATURES_FILE,
        index=False,
    )

    print(
        f"Saved to:\n"
        f"{GRAPH_FEATURES_FILE}"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = create_summary(
        graph_features
    )

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
        f"\nSummary saved to:\n"
        f"{SUMMARY_FILE}"
    )

    # --------------------------------------------------------
    # Display sample
    # --------------------------------------------------------

    print(
        "\nGraph feature sample:"
    )

    print(
        graph_features.head(10)
        .to_string(index=False)
    )

    print(
        "\nGraph feature columns:"
    )

    for feature in [
        column
        for column in graph_features.columns
        if column not in [
            "txId",
            "time_step",
        ]
    ]:
        print(
            f"  - {feature}"
        )

    print("\n" + "=" * 70)

    print(
        "GRAPH FEATURE ENGINEERING COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()