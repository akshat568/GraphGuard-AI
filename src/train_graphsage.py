from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from graph_dataset import build_graph
from graph_model import GraphSAGEClassifier


# ============================================================
# GraphGuard AI
# Phase 3 — GraphSAGE Training
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
)

MODEL_FILE = (
    OUTPUT_DIR
    / "graphsage.pt"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "graphsage_metrics.json"
)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42


def set_seed(seed):
    """Set random seeds."""

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate(
    model,
    data,
    mask,
    split_name,
):
    """Evaluate GraphSAGE on a selected mask."""

    model.eval()

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).long()

    y_true = data.y[mask].cpu().numpy()

    y_prob = (
        probabilities[mask]
        .cpu()
        .numpy()
    )

    y_pred = (
        predictions[mask]
        .cpu()
        .numpy()
    )

    metrics = {
        "split": split_name,
        "samples": int(len(y_true)),
        "positive_samples": int(
            y_true.sum()
        ),
        "negative_samples": int(
            (y_true == 0).sum()
        ),
        "positive_rate": float(
            y_true.mean()
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_prob,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_true,
                y_pred,
            ).tolist()
        ),
    }

    print(
        f"\n{split_name.upper()} RESULTS"
    )

    print(
        f"Samples: "
        f"{metrics['samples']:,}"
    )

    print(
        f"Positive rate: "
        f"{metrics['positive_rate']:.4f}"
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

    print("Confusion matrix:")

    print(
        np.array(
            metrics[
                "confusion_matrix"
            ]
        )
    )

    return metrics


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("\n" + "=" * 70)

    print(
        "GraphGuard AI — "
        "GraphSAGE Training"
    )

    print("=" * 70)

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: "
        f"{device}"
    )

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    data = build_graph()

    data = data.to(device)

    print(
        "\nGraph moved to device."
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = GraphSAGEClassifier(
        input_dim=165,
        hidden_dim=128,
        embedding_dim=64,
        dropout=0.3,
    ).to(device)

    print(
        "\nGraphSAGE model:"
    )

    print(model)

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    train_labels = data.y[
        data.train_mask
    ]

    positive_count = (
        train_labels == 1
    ).sum().item()

    negative_count = (
        train_labels == 0
    ).sum().item()

    pos_weight = (
        negative_count
        / positive_count
    )

    print(
        f"\nTraining positives: "
        f"{positive_count:,}"
    )

    print(
        f"Training negatives: "
        f"{negative_count:,}"
    )

    print(
        f"Positive class weight: "
        f"{pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    class_weights = torch.tensor(
        [
            1.0,
            pos_weight,
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    epochs = 100

    best_val_pr_auc = -1.0

    best_state = None

    print(
        f"\nTraining for "
        f"{epochs} epochs..."
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        optimizer.zero_grad()

        logits = model(
            data.x,
            data.edge_index,
        )

        loss = criterion(
            logits[
                data.train_mask
            ],
            data.y[
                data.train_mask
            ],
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                data.x,
                data.edge_index,
            )

            val_probabilities = (
                torch.softmax(
                    val_logits,
                    dim=1,
                )[:, 1]
            )

        val_y = data.y[
            data.validation_mask
        ].cpu().numpy()

        val_prob = (
            val_probabilities[
                data.validation_mask
            ]
            .cpu()
            .numpy()
        )

        val_pr_auc = (
            average_precision_score(
                val_y,
                val_prob,
            )
        )

        if val_pr_auc > best_val_pr_auc:

            best_val_pr_auc = (
                val_pr_auc
            )

            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Epoch "
                f"{epoch:03d} | "
                f"Loss: "
                f"{loss.item():.4f} | "
                f"Val PR-AUC: "
                f"{val_pr_auc:.4f}"
            )

    print(
        "\nTraining completed."
    )

    print(
        f"Best validation PR-AUC: "
        f"{best_val_pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    train_metrics = evaluate(
        model,
        data,
        data.train_mask,
        "train",
    )

    validation_metrics = evaluate(
        model,
        data,
        data.validation_mask,
        "validation",
    )

    test_metrics = evaluate(
        model,
        data,
        data.test_mask,
        "test",
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_dim": 165,

            "hidden_dim": 128,

            "embedding_dim": 64,

            "dropout": 0.3,

            "temporal_split": {
                "train": "1-34",
                "validation": "35-39",
                "test": "40-49",
            },
        },
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results = {
        "model": "GraphSAGE",

        "architecture": {
            "input_dim": 165,
            "hidden_dim": 128,
            "embedding_dim": 64,
            "dropout": 0.3,
        },

        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
        },

        "epochs": epochs,

        "best_validation_pr_auc":
            best_val_pr_auc,

        "temporal_split": {
            "train": "1-34",
            "validation": "35-39",
            "test": "40-49",
        },

        "train": train_metrics,

        "validation":
            validation_metrics,

        "test": test_metrics,
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
        f"\nModel saved to: "
        f"{MODEL_FILE}"
    )

    print(
        f"Metrics saved to: "
        f"{METRICS_FILE}"
    )

    print("\n" + "=" * 70)

    print(
        "GRAPHSAGE TRAINING COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()