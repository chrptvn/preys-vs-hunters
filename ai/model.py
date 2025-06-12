import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

class SwarmBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Input node features:
        # Channel 0: normalized index (to prevent symmetry during training)
        # Channel 1: observer type (e.g., hunter or prey)
        # Channel 2: entity type (e.g., hunter, prey, wall, etc.)
        # Channel 3: normalized distance in x-direction
        # Channel 4: normalized distance in y-direction
        self.conv1 = GCNConv(5, 32, add_self_loops=False)

        # Two additional GCN layers to deepen feature extraction
        self.conv2 = GCNConv(32, 32, add_self_loops=False)
        self.conv3 = GCNConv(32, 32, add_self_loops=False)

        # Output heads for multi-task predictions:

        # Predicts distance to the nearest target (scalar per node)
        self.distance_head = nn.Linear(32, 1)

        # Predicts how much each node attention the observer should pay to it
        self.attention_head = nn.Linear(32, 1)

        # Predicts the (x, y) location of the best target to chase
        self.target_location_head = nn.Linear(32, 2)

        # Predicts movement logits: 8 directions + 1 for "stay"
        self.action_head = nn.Linear(32, 9)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        # Apply GCN layers with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Predict distance to target for each node
        distance_scores = self.distance_head(x).squeeze(-1)

        # Predict attention scores for each node
        attention_scores = self.attention_head(x).squeeze(-1)

        # Select the most promising target (node with max chase score)
        target_idx = torch.argmax(attention_scores)

        # Predict the target's location based on selected node
        target_location_score = self.target_location_head(x[target_idx])

        # Predict action logits (movement direction) based on selected node
        action_logits = self.action_head(x[target_idx])

        # Return all outputs for downstream decision-making
        return distance_scores, attention_scores, target_location_score, action_logits
