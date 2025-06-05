import torch
import random
import torch.nn as nn
from torch_geometric.data import Data
from ai.brain import Movement, EntityType
from ai.model import SwarmBrain

# ----------------------------------------------------------------------
def get_action_from_delta(dx, dy):
    if dy < 0:
        if dx > 0:
            return Movement.UPPER_RIGHT
        elif dx < 0:
            return Movement.UPPER_LEFT
        else:
            return Movement.UP
    elif dy > 0:
        if dx > 0:
            return Movement.LOWER_RIGHT
        elif dx < 0:
            return Movement.LOWER_LEFT
        else:
            return Movement.DOWN
    else:
        return Movement.RIGHT if dx > 0 else Movement.LEFT

# ----------------------------------------------------------------------
def sample(n_entities=5):
    ox, oy = random.randint(0, 10), random.randint(0, 10)
    rows = []
    entities = []

    observer_type = random.choice([EntityType.PREY, EntityType.HUNTER])

    for _ in range(n_entities):
        ex, ey = random.randint(0, 10), random.randint(0, 10)
        dx, dy = (ex - ox) / 10, (ey - oy) / 10
        dist = (dx * dx + dy * dy) ** 0.5
        e_type = random.choice([EntityType.PREY, EntityType.HUNTER])
        rows.append([e_type.value, dx, dy])
        entities.append((e_type, dist, dx, dy))

    rows.append([observer_type.value, 0.0, 0.0])
    observer_idx = n_entities
    num_nodes = n_entities + 1

    edges = []
    for i in range(n_entities):
        edges.append([i, observer_idx])
        edges.append([observer_idx, i])
    for i in range(num_nodes):
        edges.append([i, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(rows, dtype=torch.float32)

    if observer_type == EntityType.PREY:
        hunters = [(i, dist) for i, (t, dist, _, _) in enumerate(entities) if t == EntityType.HUNTER]
        if not hunters:
            return None
        absolute_target = min(hunters, key=lambda x: x[1])[0]
    else:
        preys = [(i, dist) for i, (t, dist, _, _) in enumerate(entities) if t == EntityType.PREY]
        if not preys:
            return None
        absolute_target = min(preys, key=lambda x: x[1])[0]


    return Data(x=x, edge_index=edge_index), absolute_target, observer_type, entities

# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SwarmBrain().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

action_labels = [
    "UP", "UPPER_RIGHT", "RIGHT", "LOWER_RIGHT",
    "DOWN", "LOWER_LEFT", "LEFT", "UPPER_LEFT"
]

for epoch in range(20000):
    sample_result = sample()
    if sample_result is None:
        continue
    g, absolute_target, observer_type, entities = sample_result
    g = g.to(device)
    entity_types = g.x[:-1, 0]  # exclude observer

    chase_scores, flee_scores, action_logits = model(g)
    is_flee = observer_type == EntityType.PREY

    target_type = (EntityType.HUNTER if is_flee else EntityType.PREY).value
    mask = entity_types == target_type
    if mask.sum().item() == 0:
        continue

    logits = (flee_scores if is_flee else chase_scores)[:-1][mask]
    masked_indices = mask.nonzero(as_tuple=True)[0]
    if absolute_target not in masked_indices:
        continue

    relative_target = (masked_indices == absolute_target).nonzero(as_tuple=True)[0].item()
    target = torch.tensor([relative_target], dtype=torch.long, device=device)

    _, _, dx, dy = entities[absolute_target]
    action = get_action_from_delta(dx, dy)
    action_target = torch.tensor([action.value], dtype=torch.long, device=device)

    distance_weight = torch.tensor([entities[absolute_target][1]], dtype=torch.float32, device=device)
    loss_behavior = loss_fn(logits.unsqueeze(0), target) * distance_weight
    loss_action = loss_fn(action_logits.unsqueeze(0), action_target)
    total_loss = loss_behavior + loss_action

    opt.zero_grad()
    total_loss.backward()
    opt.step()

    if epoch % 1 == 0:
        pred_idx = logits.argmax().item()
        pred_absolute = masked_indices[pred_idx].item()
        pred_type = "PREY" if entities[pred_absolute][0] == EntityType.PREY else "HUNTER"
        tgt_type = "PREY" if entities[absolute_target][0] == EntityType.PREY else "HUNTER"
        type_str = "PREY" if observer_type == EntityType.PREY else "HUNTER"
        behavior = "FLEE" if is_flee else "CHASE"

        n_preys = sum(1 for e, _, _, _ in entities if e == EntityType.PREY)
        n_hunters = sum(1 for e, _, _, _ in entities if e == EntityType.HUNTER)

        pred_action_idx = action_logits.argmax().item()
        pred_action_str = action_labels[pred_action_idx]
        target_action_str = action_labels[action.value]

        print(f"{epoch:4d}  {behavior:5s}  observer={type_str:6s}  loss={total_loss.item():.4f}  "
              f"pred={pred_absolute} ({pred_type})  tgt={absolute_target} ({tgt_type})  "
              f"[preys={n_preys} hunters={n_hunters}]  "
              f"action={pred_action_str} (tgt={target_action_str})")

torch.save(model.state_dict(), "swarmbrain.pt")
print("Model saved to swarmbrain.pt")
