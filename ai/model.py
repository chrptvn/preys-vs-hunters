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

        # Predicts how attractive a target is to chase (scalar per node)
        self.chase_head = nn.Linear(32, 1)

        # Predicts the (x, y) location of the best target to chase
        self.target_location_head = nn.Linear(32, 2)

        # Predicts movement logits: 8 directions + 1 for "stay"
        self.action_head = nn.Linear(32, 9)

    def forward(self, data):
        # Apply GCN layers with ReLU activation
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.relu(self.conv3(x, data.edge_index))

        # Predict distance to target for each node
        distance_scores = self.distance_head(x).squeeze(-1)

        # Predict chase preference scores for each node
        chase_scores = self.chase_head(x).squeeze(-1)

        # Select the most promising target (node with max chase score)
        target_idx = torch.argmax(chase_scores)

        # Predict the target's location based on selected node
        target_location_score = self.target_location_head(x[target_idx])

        # Predict action logits (movement direction) based on selected node
        action_logits = self.action_head(x[target_idx])

        # Return all outputs for downstream decision-making
        return distance_scores, chase_scores, target_location_score, action_logits
