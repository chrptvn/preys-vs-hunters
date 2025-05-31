import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from enum import Enum


# Define Enums for readability
class Animal(Enum):
    RED_CAT = 0
    BLUE_CAT = 1
    RED_BIRD = 2
    BLUE_BIRD = 3


class Color(Enum):
    RED = 0
    BLUE = 1


# Features: [legs, color]
x = torch.tensor([
    [4.0, Color.RED.value],    # RED_CAT
    [4.0, Color.BLUE.value],   # BLUE_CAT
    [2.0, Color.RED.value],    # RED_BIRD
    [2.0, Color.BLUE.value],   # BLUE_BIRD
], dtype=torch.float)

# Labels: 0 = RedCat, 1 = BlueCat, 2 = RedBird, 3 = BlueBird
y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

# Connect nodes (arbitrary connections)
edge_index = torch.tensor([
    [0, 0, 1, 1, 2, 2, 3, 3],
    [1, 2, 0, 3, 0, 3, 1, 2]
], dtype=torch.long)

# Train on 3 nodes, test on 1
train_mask = torch.tensor([True, True, True, False])
test_mask = torch.tensor([False, False, False, True])

data = Data(x=x, edge_index=edge_index, y=y,
            train_mask=train_mask, test_mask=test_mask)


# Define a GCN with 4 output classes
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 8)
        self.conv2 = GCNConv(8, 4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Initialize and train
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Inference
model.eval()
out = model(data)
_, pred = out.max(dim=1)

# Print predictions
print("\nPredictions:")
for animal in Animal:
    true_class = animal.name
    predicted_class = list(Animal)[pred[animal.value].item()].name
    print(f"{true_class} → Predicted: {predicted_class}")

# Test accuracy
correct = pred[data.test_mask] == y[data.test_mask]
accuracy = int(correct.sum()) / int(data.test_mask.sum())
print("\nTest Accuracy:", accuracy)

# Custom input function
def classify(legs: int, color: str):
    color_val = Color[color.upper()].value
    input_tensor = torch.tensor([[float(legs), float(color_val)]], dtype=torch.float)
    data_with_input = Data(x=torch.cat([data.x, input_tensor], dim=0),
                           edge_index=data.edge_index)
    model.eval()
    out = model(data_with_input)
    predicted = out[-1].argmax().item()
    return list(Animal)[predicted].name

# Try new predictions
print("\n🔮 Custom Predictions:")
print("2 legs, Red →", classify(2, "red"))
print("4 legs, Blue →", classify(4, "blue"))
print("2 legs, Blue →", classify(2, "blue"))
print("4 legs, Red →", classify(4, "red"))
