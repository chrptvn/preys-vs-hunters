import torch
from torch_geometric.data import Data
from enum import Enum

from preys_vs_hunters.goal.Goal import GoalType, Stay


class NodeType(Enum):
    OBSERVER = 0
    LOCALISATION = 1
    ACTION = 2
    AGGREGATION = 3
    ENTITY = 4
    GOAL = 5

# --- Entity & Goal Stubs ---
class Entity:
    def __init__(self, entity_id, entity_type, x, y):
        self.id = entity_id
        self.type = entity_type
        self.x = x
        self.y = y

# --- Helpers ---
def get_relative_location(entity, observer, grid_size):
    return (entity.x - observer.x, entity.y - observer.y)

def get_normalized_distance(relative_location, grid_size):
    dx, dy = relative_location
    max_dist = max(grid_size)
    return ((dx**2 + dy**2)**0.5) / max_dist

# --- DataBuilder ---
class DataBuilder:
    def __init__(self, observer, grid_size):
        self.observer = observer
        self.grid_size = grid_size
        self.rows, self.edges, self.entity_node_map = [], [], {}
        self.clear()

    def _create_row(self, row): self.rows.append(row); return len(self.rows) - 1

    def clear(self):
        self.rows, self.edges, self.entity_node_map = [], [], {}
        self.observer_idx = self._create_row([0, float(self.observer.type.value), 0.0, 0.0])
        self.loc_idx = self._create_row([1, 0.0, 0.0, 0.0])
        self.action_idx = self._create_row([2, 0.0, 0.0, 0.0])
        self.agg_idx = self._create_row([3, 0.0, 0.0, 0.0])
        self.edges += [[self.observer_idx, self.agg_idx],
                       [self.loc_idx, self.agg_idx],
                       [self.action_idx, self.agg_idx]]
        self.add_goal(Stay())

    def add_entity(self, entity):
        dx, dy = entity.x - self.observer.x, entity.y - self.observer.y
        dist = ((dx**2 + dy**2)**0.5) / max(self.grid_size)
        idx = self._create_row([4, float(entity.type.value), dist, 0.0])
        self.entity_node_map[entity.id] = idx
        self.edges.append([idx, self.loc_idx])

    def add_goal(self, goal):
        target_idx = self.entity_node_map.get(goal.target_id, -1)
        idx = self._create_row([5, float(goal.type.value), float(target_idx), 0.0])
        self.edges.append([idx, self.action_idx])

    def get_index_by_entity_id(self, entity_id):
        return self.entity_node_map[entity_id]

    def build(self):
        x = torch.tensor(self.rows, dtype=torch.float32)
        edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)