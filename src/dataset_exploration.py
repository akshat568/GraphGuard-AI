from pathlib import Path
import json

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# GraphGuard AI
# Phase 1 — Dataset Exploration & Audit
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase1"
PLOT_DIR = OUTPUT_DIR / "plots"

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"
EDGES_FILE = DATA_DIR / "elliptic_txs_edgelist.csv"

CHUNK_SIZE = 50_000


def check_files():
    """Check that all required dataset files exist."""

    print("\n" + "=" * 70)
    print("1. DATASET FILE CHECK")
    print("=" * 70)

    files = [
        FEATURES_FILE,
        CLASSES_FILE,
        EDGES_FILE,
    ]

    for file_path in files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {file_path}"
            )

        size_mb = file_path.stat().st_size / (1024 * 1024)

        print(
            f"[OK] {file_path.name:<35} "
            f"{size_mb:>10.2f} MB"
        )


def load_classes():
    """Load transaction labels."""

    print("\n" + "=" * 70)
    print("2. LOADING CLASSES")
    print("=" * 70)

    classes = pd.read_csv(CLASSES_FILE)

    print(f"Rows: {len(classes):,}")
    print(f"Columns: {classes.columns.tolist()}")

    print("\nClass distribution:")
    print(classes["class"].value_counts(dropna=False))

    return classes


def load_node_metadata():
    """
    Load only txId and time_step from the large feature file.

    The original feature CSV has no header.
    """

    print("\n" + "=" * 70)
    print("3. LOADING NODE METADATA")
    print("=" * 70)

    node_metadata = pd.read_csv(
        FEATURES_FILE,
        header=None,
        usecols=[0, 1],
        names=["txId", "time_step"],
    )

    print(f"Number of transactions: {len(node_metadata):,}")
    print(
        f"Unique transaction IDs: "
        f"{node_metadata['txId'].nunique():,}"
    )

    duplicate_count = (
        node_metadata["txId"].duplicated().sum()
    )

    print(f"Duplicate transaction IDs: {duplicate_count:,}")

    print("\nTime-step distribution:")
    print(
        node_metadata["time_step"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return node_metadata


def analyze_temporal_labels(node_metadata, classes):
    """Analyze labels across time."""

    print("\n" + "=" * 70)
    print("4. TEMPORAL LABEL DISTRIBUTION")
    print("=" * 70)

    merged = node_metadata.merge(
        classes,
        on="txId",
        how="left",
        validate="one_to_one",
    )

    missing_labels = merged["class"].isna().sum()

    print(f"Transactions without a class entry: {missing_labels:,}")

    temporal = (
        merged
        .groupby(["time_step", "class"])
        .size()
        .unstack(fill_value=0)
    )

    for required_class in ["1", "2", "unknown"]:
        if required_class not in temporal.columns:
            temporal[required_class] = 0

    temporal = temporal[["1", "2", "unknown"]]

    temporal.columns = [
        "illicit",
        "licit",
        "unknown",
    ]

    temporal["labeled"] = (
        temporal["illicit"] + temporal["licit"]
    )

    temporal["illicit_rate"] = np.where(
        temporal["labeled"] > 0,
        temporal["illicit"] / temporal["labeled"],
        0,
    )

    print("\n")
    print(temporal.to_string())

    return merged, temporal


def analyze_edges(node_metadata):
    """Analyze edge validity, direction and temporal consistency."""

    print("\n" + "=" * 70)
    print("5. EDGE ANALYSIS")
    print("=" * 70)

    edges = pd.read_csv(EDGES_FILE)

    print(f"Number of edges: {len(edges):,}")
    print(f"Columns: {edges.columns.tolist()}")

    # --------------------------------------------------------
    # Endpoint validity
    # --------------------------------------------------------

    node_ids = set(node_metadata["txId"])

    source_valid = edges["txId1"].isin(node_ids)
    target_valid = edges["txId2"].isin(node_ids)

    invalid_sources = (~source_valid).sum()
    invalid_targets = (~target_valid).sum()

    print("\nEndpoint validation:")
    print(f"Invalid source transaction IDs: {invalid_sources:,}")
    print(f"Invalid target transaction IDs: {invalid_targets:,}")

    # --------------------------------------------------------
    # Self-loops
    # --------------------------------------------------------

    self_loops = (
        edges["txId1"] == edges["txId2"]
    ).sum()

    print(f"Self-loop edges: {self_loops:,}")

    # --------------------------------------------------------
    # Temporal edge audit
    # --------------------------------------------------------

    source_times = node_metadata.rename(
        columns={
            "txId": "txId1",
            "time_step": "source_time",
        }
    )

    target_times = node_metadata.rename(
        columns={
            "txId": "txId2",
            "time_step": "target_time",
        }
    )

    edge_times = edges.merge(
        source_times,
        on="txId1",
        how="left",
    )

    edge_times = edge_times.merge(
        target_times,
        on="txId2",
        how="left",
    )

    missing_source_time = edge_times["source_time"].isna().sum()
    missing_target_time = edge_times["target_time"].isna().sum()

    print("\nTemporal endpoint validation:")
    print(
        f"Edges with missing source timestep: "
        f"{missing_source_time:,}"
    )
    print(
        f"Edges with missing target timestep: "
        f"{missing_target_time:,}"
    )

    same_time = (
        edge_times["source_time"]
        == edge_times["target_time"]
    )

    cross_time = (~same_time).sum()

    print(f"\nEdges connecting different timesteps: {cross_time:,}")

    if cross_time > 0:
        print("\nCross-timestep edge examples:")
        print(
            edge_times.loc[
                ~same_time,
                [
                    "txId1",
                    "txId2",
                    "source_time",
                    "target_time",
                ],
            ]
            .head(20)
            .to_string(index=False)
        )

    temporal_summary = (
        edge_times
        .groupby(
            ["source_time", "target_time"]
        )
        .size()
        .reset_index(name="edge_count")
    )

    return edges, edge_times, temporal_summary


def analyze_degree_distribution(edges, node_metadata):
    """Calculate in-degree, out-degree and total degree."""

    print("\n" + "=" * 70)
    print("6. DEGREE ANALYSIS")
    print("=" * 70)

    in_degree = (
        edges.groupby("txId2")
        .size()
        .rename("in_degree")
    )

    out_degree = (
        edges.groupby("txId1")
        .size()
        .rename("out_degree")
    )

    degree = node_metadata[["txId"]].copy()

    degree = degree.merge(
        in_degree,
        left_on="txId",
        right_index=True,
        how="left",
    )

    degree = degree.merge(
        out_degree,
        left_on="txId",
        right_index=True,
        how="left",
    )

    degree = degree.fillna(0)

    degree["total_degree"] = (
        degree["in_degree"]
        + degree["out_degree"]
    )

    print("\nDegree statistics:")
    print(
        degree[
            [
                "in_degree",
                "out_degree",
                "total_degree",
            ]
        ].describe()
    )

    return degree


def analyze_connected_components(edges, node_metadata):
    """Analyze weakly connected components."""

    print("\n" + "=" * 70)
    print("7. CONNECTED COMPONENT ANALYSIS")
    print("=" * 70)

    graph = nx.DiGraph()

    graph.add_nodes_from(
        node_metadata["txId"].tolist()
    )

    graph.add_edges_from(
        edges[["txId1", "txId2"]]
        .itertuples(index=False, name=None)
    )

    weak_components = list(
        nx.weakly_connected_components(graph)
    )

    component_sizes = np.array(
        [len(component) for component in weak_components]
    )

    print(
        f"Number of weakly connected components: "
        f"{len(weak_components):,}"
    )

    print("\nComponent size statistics:")

    print(
        pd.Series(component_sizes).describe()
    )

    return graph, component_sizes


def analyze_features():
    """
    Process the large feature CSV in chunks.

    We check:
    - row count
    - missing values
    - infinite values
    - numeric conversion problems
    """

    print("\n" + "=" * 70)
    print("8. FEATURE DATA QUALITY")
    print("=" * 70)

    feature_names = (
        ["txId", "time_step"]
        + [
            f"feature_{i}"
            for i in range(1, 166)
        ]
    )

    total_rows = 0
    missing_values = 0
    infinite_values = 0
    invalid_numeric_values = 0

    timestep_counts = {}

    reader = pd.read_csv(
        FEATURES_FILE,
        header=None,
        names=feature_names,
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        total_rows += len(chunk)

        # ----------------------------------------------------
        # Time-step distribution
        # ----------------------------------------------------

        counts = (
            chunk["time_step"]
            .value_counts()
        )

        for timestep, count in counts.items():
            timestep_counts[timestep] = (
                timestep_counts.get(timestep, 0)
                + int(count)
            )

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        missing_values += int(
            chunk.isna().sum().sum()
        )

        # ----------------------------------------------------
        # Numeric feature validation
        # ----------------------------------------------------

        numeric_features = chunk.iloc[:, 2:]

        numeric_array = numeric_features.to_numpy(
            dtype=np.float64,
            copy=False,
        )

        infinite_values += int(
            np.isinf(numeric_array).sum()
        )

        if chunk_number % 2 == 0:
            print(
                f"Processed approximately "
                f"{total_rows:,} feature rows..."
            )

    print("\nFeature quality results:")
    print(f"Total feature rows: {total_rows:,}")
    print(f"Missing values: {missing_values:,}")
    print(f"Infinite values: {infinite_values:,}")
    print(
        f"Invalid numeric values: "
        f"{invalid_numeric_values:,}"
    )

    print("\nFeature timestep distribution:")
    print(
        pd.Series(timestep_counts)
        .sort_index()
        .to_string()
    )

    return {
        "total_rows": total_rows,
        "missing_values": missing_values,
        "infinite_values": infinite_values,
        "invalid_numeric_values": invalid_numeric_values,
    }


def create_plots(
    classes,
    temporal,
    degree,
):
    """Create Phase 1 visualizations."""

    print("\n" + "=" * 70)
    print("9. CREATING PLOTS")
    print("=" * 70)

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_counts = (
        classes["class"]
        .value_counts()
        .reindex(
            ["1", "2", "unknown"],
            fill_value=0,
        )
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        ["Illicit", "Licit", "Unknown"],
        class_counts.values,
    )

    plt.title("GraphGuard AI — Class Distribution")
    plt.ylabel("Number of Transactions")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "class_distribution.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Transactions per timestep
    # --------------------------------------------------------

    timestep_total = (
        temporal[
            ["illicit", "licit", "unknown"]
        ]
        .sum(axis=1)
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        timestep_total.index,
        timestep_total.values,
        marker="o",
        markersize=3,
    )

    plt.title(
        "GraphGuard AI — Transactions per Time Step"
    )

    plt.xlabel("Time Step")
    plt.ylabel("Transactions")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "transactions_by_time.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Illicit rate over time
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        temporal.index,
        temporal["illicit_rate"] * 100,
        marker="o",
        markersize=3,
    )

    plt.title(
        "GraphGuard AI — Illicit Rate by Time Step"
    )

    plt.xlabel("Time Step")
    plt.ylabel("Illicit Rate (%)")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "illicit_rate_by_time.png",
        dpi=150,
    )

    plt.close()

    # --------------------------------------------------------
    # Degree distribution
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.hist(
        degree["total_degree"],
        bins=50,
    )

    plt.title(
        "GraphGuard AI — Total Degree Distribution"
    )

    plt.xlabel("Total Degree")
    plt.ylabel("Number of Transactions")

    plt.yscale("log")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "degree_distribution.png",
        dpi=150,
    )

    plt.close()

    print(f"Plots saved to: {PLOT_DIR}")


def save_outputs(
    classes,
    temporal,
    edge_times,
    temporal_summary,
    degree,
    component_sizes,
    feature_quality,
):
    """Save Phase 1 analysis results."""

    print("\n" + "=" * 70)
    print("10. SAVING RESULTS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Class distribution

    class_distribution = (
        classes["class"]
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )

    class_distribution.to_csv(
        OUTPUT_DIR / "class_distribution.csv",
        index=False,
    )

    # Temporal labels

    temporal.to_csv(
        OUTPUT_DIR / "temporal_label_distribution.csv"
    )

    # Edge time analysis

    edge_times.to_csv(
        OUTPUT_DIR / "edge_time_audit.csv",
        index=False,
    )

    temporal_summary.to_csv(
        OUTPUT_DIR / "edge_temporal_summary.csv",
        index=False,
    )

    # Degree statistics

    degree.to_csv(
        OUTPUT_DIR / "degree_statistics.csv",
        index=False,
    )

    # Component statistics

    component_df = pd.DataFrame(
        {
            "component_size": component_sizes
        }
    )

    component_df.to_csv(
        OUTPUT_DIR / "connected_component_sizes.csv",
        index=False,
    )

    # Dataset summary

    summary = {
        "transactions": int(len(degree)),
        "edges": int(len(edge_times)),
        "unique_transaction_ids": int(
            degree["txId"].nunique()
        ),
        "time_steps": int(
            temporal.index.nunique()
        ),
        "illicit_transactions": int(
            classes["class"].eq("1").sum()
        ),
        "licit_transactions": int(
            classes["class"].eq("2").sum()
        ),
        "unknown_transactions": int(
            classes["class"].eq("unknown").sum()
        ),
        "weakly_connected_components": int(
            len(component_sizes)
        ),
        "largest_component_size": int(
            component_sizes.max()
        ),
        "feature_quality": feature_quality,
    }

    with open(
        OUTPUT_DIR / "dataset_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        f"Results saved to: {OUTPUT_DIR}"
    )


def main():

    print("\n" + "=" * 70)
    print(
        "GraphGuard AI — Phase 1: "
        "Dataset Exploration & Audit"
    )
    print("=" * 70)

    check_files()

    classes = load_classes()

    node_metadata = load_node_metadata()

    merged, temporal = analyze_temporal_labels(
        node_metadata,
        classes,
    )

    edges, edge_times, temporal_summary = (
        analyze_edges(node_metadata)
    )

    degree = analyze_degree_distribution(
        edges,
        node_metadata,
    )

    graph, component_sizes = (
        analyze_connected_components(
            edges,
            node_metadata,
        )
    )

    feature_quality = analyze_features()

    create_plots(
        classes,
        temporal,
        degree,
    )

    save_outputs(
        classes,
        temporal,
        edge_times,
        temporal_summary,
        degree,
        component_sizes,
        feature_quality,
    )

    print("\n" + "=" * 70)
    print("PHASE 1 DATASET AUDIT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()