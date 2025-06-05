from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F


class SwarmBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 32, add_self_loops=False)
        self.conv2 = GCNConv(32, 32, add_self_loops=False)

        self.chase_head = nn.Linear(32, 1)
        self.flee_head = nn.Linear(32, 1)
        self.action_head = nn.Linear(32, 8)  # 8 directions

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))

        chase_scores = self.chase_head(x).squeeze(-1)
        flee_scores = self.flee_head(x).squeeze(-1)
        action_logits = self.action_head(x[-1])  # observer node

        return chase_scores, flee_scores, action_logits
