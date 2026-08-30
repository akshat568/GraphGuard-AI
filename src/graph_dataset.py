from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# ============================================================
# GraphGuard AI
# Phase 3 — Graph Dataset Construction
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"
EDGES_FILE = DATA_DIR / "elliptic_txs_edgelist.csv"


def load_features():
    """Load node IDs, timesteps and 165 node features."""

    print("\nLoading feature data...")

    feature_columns = (
        ["txId", "time_step"]
        + [
            f"feature_{i}"
            for i in range(1, 166)
        ]
    )

    features = pd.read_csv(
        FEATURES_FILE,
        header=None,
        names=feature_columns,
    )

    print(
        f"Transactions loaded: "
        f"{len(features):,}"
    )

    print(
        f"Feature dimensions: "
        f"{len(feature_columns) - 2}"
    )

    return features


def load_classes():
    """Load transaction labels."""

    print("\nLoading classes...")

    classes = pd.read_csv(
        CLASSES_FILE
    )

    print(
        f"Class records loaded: "
        f"{len(classes):,}"
    )

    return classes


def create_node_mapping(tx_ids):
    """
    Convert transaction IDs into consecutive integer indices.

    PyTorch Geometric expects node indices such as:

    0, 1, 2, ..., N-1
    """

    return {
        tx_id: index
        for index, tx_id in enumerate(tx_ids)
    }


def create_node_features(features):
    """Create the X matrix containing only the 165 features."""

    feature_columns = [
        f"feature_{i}"
        for i in range(1, 166)
    ]

    x = torch.tensor(
        features[feature_columns].values,
        dtype=torch.float32,
    )

    return x


def create_labels(features, classes):
    """
    Create node labels.

    class 1 = illicit -> 1
    class 2 = licit   -> 0
    unknown           -> -1
    """

    merged = features[
        ["txId", "time_step"]
    ].merge(
        classes,
        on="txId",
        how="left",
        validate="one_to_one",
    )

    y = torch.full(
        (len(merged),),
        -1,
        dtype=torch.long,
    )

    y[
        merged["class"] == "1"
    ] = 1

    y[
        merged["class"] == "2"
    ] = 0

    known_label_mask = y != -1

    return (
        y,
        known_label_mask,
        merged,
    )


def create_edges(
    edges,
    tx_to_index,
):
    """
    Convert transaction IDs in the edge list into
    integer node indices.

    The original graph is directed:

        txId1 -> txId2
    """

    source_indices = edges[
        "txId1"
    ].map(tx_to_index)

    target_indices = edges[
        "txId2"
    ].map(tx_to_index)

    if source_indices.isna().any():
        raise ValueError(
            "Some source transaction IDs "
            "are missing from the node mapping."
        )

    if target_indices.isna().any():
        raise ValueError(
            "Some target transaction IDs "
            "are missing from the node mapping."
        )

    edge_index = torch.tensor(
        np.vstack(
            [
                source_indices.to_numpy(
                    dtype=np.int64
                ),
                target_indices.to_numpy(
                    dtype=np.int64
                ),
            ]
        ),
        dtype=torch.long,
    )

    return edge_index


def create_temporal_masks(
    time_steps,
    known_label_mask,
):
    """
    Create chronological train/validation/test masks.

    Train       = timesteps 1-34
    Validation  = timesteps 35-39
    Test        = timesteps 40-49

    Unknown transactions are excluded from each mask.
    """

    train_mask = (
        (time_steps <= 34)
        & known_label_mask
    )

    validation_mask = (
        (time_steps >= 35)
        & (time_steps <= 39)
        & known_label_mask
    )

    test_mask = (
        (time_steps >= 40)
        & (time_steps <= 49)
        & known_label_mask
    )

    return (
        train_mask,
        validation_mask,
        test_mask,
    )


def build_graph():
    """Build the complete PyTorch Geometric graph."""

    print("\n" + "=" * 70)
    print(
        "GraphGuard AI — "
        "Building PyTorch Geometric Graph"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    features = load_features()

    classes = load_classes()

    edges = pd.read_csv(
        EDGES_FILE
    )

    # --------------------------------------------------------
    # Node mapping
    # --------------------------------------------------------

    print("\nCreating transaction ID mapping...")

    tx_to_index = create_node_mapping(
        features["txId"].tolist()
    )

    # --------------------------------------------------------
    # Node features
    # --------------------------------------------------------

    print("\nCreating node feature matrix...")

    x = create_node_features(
        features
    )

    print(
        f"Node feature matrix: "
        f"{tuple(x.shape)}"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    print("\nCreating node labels...")

    (
        y,
        known_label_mask,
        merged,
    ) = create_labels(
        features,
        classes,
    )

    print(
        f"Known labels: "
        f"{int(known_label_mask.sum()):,}"
    )

    print(
        f"Unknown labels: "
        f"{int((~known_label_mask).sum()):,}"
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    print("\nCreating edge index...")

    edge_index = create_edges(
        edges,
        tx_to_index,
    )

    print(
        f"Edge index: "
        f"{tuple(edge_index.shape)}"
    )

    # --------------------------------------------------------
    # Time steps
    # --------------------------------------------------------

    time_steps = torch.tensor(
        features["time_step"].values,
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Temporal masks
    # --------------------------------------------------------

    print(
        "\nCreating temporal masks..."
    )

    (
        train_mask,
        validation_mask,
        test_mask,
    ) = create_temporal_masks(
        time_steps,
        known_label_mask,
    )

    print(
        f"Train nodes: "
        f"{int(train_mask.sum()):,}"
    )

    print(
        f"Validation nodes: "
        f"{int(validation_mask.sum()):,}"
    )

    print(
        f"Test nodes: "
        f"{int(test_mask.sum()):,}"
    )

    # --------------------------------------------------------
    # Build PyG Data object
    # --------------------------------------------------------

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )

    data.time_step = time_steps

    data.known_label_mask = (
        known_label_mask
    )

    data.train_mask = train_mask
    data.validation_mask = (
        validation_mask
    )
    data.test_mask = test_mask

    # --------------------------------------------------------
    # Validation checks
    # --------------------------------------------------------

    print(
        "\nRunning graph validation..."
    )

    assert data.num_nodes == 203769

    assert data.num_edges == 234355

    assert data.num_node_features == 165

    assert int(
        data.train_mask.sum()
    ) == 29894

    assert int(
        data.validation_mask.sum()
    ) == 5486

    assert int(
        data.test_mask.sum()
    ) == 11184

    assert (
        data.train_mask
        & data.validation_mask
    ).sum() == 0

    assert (
        data.train_mask
        & data.test_mask
    ).sum() == 0

    assert (
        data.validation_mask
        & data.test_mask
    ).sum() == 0

    print(
        "All graph validation checks passed."
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GRAPH SUMMARY")
    print("=" * 70)

    print(
        f"Nodes: "
        f"{data.num_nodes:,}"
    )

    print(
        f"Edges: "
        f"{data.num_edges:,}"
    )

    print(
        f"Node features: "
        f"{data.num_node_features}"
    )

    print(
        f"Known labels: "
        f"{int(data.known_label_mask.sum()):,}"
    )

    print(
        f"Train: "
        f"{int(data.train_mask.sum()):,}"
    )

    print(
        f"Validation: "
        f"{int(data.validation_mask.sum()):,}"
    )

    print(
        f"Test: "
        f"{int(data.test_mask.sum()):,}"
    )

    print(
        "\nGraph dataset construction "
        "completed successfully."
    )

    return data


if __name__ == "__main__":
    build_graph()