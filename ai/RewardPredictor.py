import torch
import torch.nn as nn


class RewardPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # (4, 11, 11) → (16, 11, 11)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (16, 11, 11) → (32, 11, 11)
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 11 * 11 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output = predicted reward
        )

    def forward(self, grid, movement_onehot):
        grid_features = self.conv(grid)
        x = torch.cat([grid_features, movement_onehot], dim=1)
        return self.fc(x)
