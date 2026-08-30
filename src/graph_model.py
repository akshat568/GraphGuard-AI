import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv


class GraphSAGEClassifier(nn.Module):
    """
    Two-layer GraphSAGE node classifier.

    Architecture:

        165
         ↓
       SAGEConv
         ↓
       128
         ↓
       ReLU
         ↓
       Dropout
         ↓
       SAGEConv
         ↓
        64
         ↓
       ReLU
         ↓
       Linear
         ↓
         2 classes
    """

    def __init__(
        self,
        input_dim=165,
        hidden_dim=128,
        embedding_dim=64,
        dropout=0.3,
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim,
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            embedding_dim,
        )

        self.classifier = nn.Linear(
            embedding_dim,
            2,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        edge_index,
    ):
        """
        Forward pass.

        Returns:
            logits: class logits for every node
        """

        x = self.conv1(
            x,
            edge_index,
        )

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index,
        )

        x = torch.relu(x)

        logits = self.classifier(x)

        return logits

    def get_embeddings(
        self,
        x,
        edge_index,
    ):
        """
        Return the learned node embeddings
        before the final classification layer.
        """

        x = self.conv1(
            x,
            edge_index,
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index,
        )

        x = torch.relu(x)

        return x