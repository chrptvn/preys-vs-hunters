import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from enum import Enum


# Replace with your actual ActionType import
class ActionType(Enum):
    IDLE = 0
    MOVE = 1
    FOLLOW = 2
    ESCAPE = 3


class Model(nn.Module):
    def __init__(self, model_file, device):
        super().__init__()
        self.device = device
        if model_file and os.path.exists(model_file):
            self.checkpoint = torch.load(model_file, map_location=device)
        else:
            self.checkpoint = {}

    def load_checkpoint(self, checkpoint: str):
        if checkpoint not in self.checkpoint:
            raise ValueError(f"Checkpoint '{checkpoint}' not found.")
        self.load_state_dict(self.checkpoint[checkpoint])
        self.to(self.device)
        self.eval()


class HunterModel(Model):
    def __init__(self, model_file: str, device, max_target_entities: int = 10):
        super().__init__(model_file, device)

        input_dim = 4  # [node_type, type/goal_type, distance, extra (optional)]
        hidden_dim = 64
        self.max_target_entities = max_target_entities

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.3)

        self.pre_head = nn.Linear(hidden_dim, hidden_dim)

        # Multi-head outputs
        self.heads = nn.ModuleDict({
            "action": nn.Linear(hidden_dim, len(ActionType)),
            "move": nn.Linear(hidden_dim, 2),  # dx, dy
            "goal": nn.Linear(hidden_dim, 3),  # CHASE, ESCAPE, STAY
            "target": nn.Linear(hidden_dim, max_target_entities)
        })

    def forward(self, x, edge_index, node_index: int = 0):
        x = self.dropout(F.relu(self.conv1(x, edge_index)))
        x = self.dropout(F.relu(self.conv2(x, edge_index)))
        x = self.dropout(F.relu(self.conv3(x, edge_index)))

        node = x[node_index]  # by default, assume node 0 is the observer
        node = F.relu(self.pre_head(node))

        return (
            self.heads["action"](node),  # logits over ActionType
            self.heads["move"](node),    # movement vector
            self.heads["goal"](node),    # logits over goals
            self.heads["target"](node)   # logits over entity indexes
        )
