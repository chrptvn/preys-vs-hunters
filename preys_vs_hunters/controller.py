from enum import Enum
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

import torch
from torch_geometric.data import Data

from ai.model import SwarmBrain
from preys_vs_hunters.display import GridDisplay
from preys_vs_hunters.entities.entity import EntityType, Movement, Entity
from preys_vs_hunters.entities.pool import Pool
from preys_vs_hunters.utils import get_entities_at_location


class ActionType(Enum):
    SPAWN = 0
    DELETE = 1
    WATCH = 2


class PreysVsHunters:
    def __init__(self, model: SwarmBrain, device, size=(100, 100), interval=200):
        self.is_running = False
        self.spawn_type = None
        self.action = ActionType.WATCH
        self.grid = np.zeros(size, dtype=int)
        self.entities_pool = Pool()
        self.display = GridDisplay(self.grid)
        self.fig = self.display.get_figure()
        self.ax = self.display.ax
        self.ani = FuncAnimation(
            self.fig,
            self._update,
            interval=interval,
            blit=True,
            cache_frame_data=False
        )

        self._setup_events()
        self._setup_ui()

        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.size = size

    def build_graph_for_entity(self, observer: Entity, all_entities: List[Entity]):
        ox, oy = observer.location
        rows = []
        edges = []
        entities = []

        for e in all_entities:
            if e.id == observer.id:
                continue
            ex, ey = e.location
            dx, dy = (ex - ox) / self.grid.shape[0], (ey - oy) / self.grid.shape[1]
            e_type = e.type.value if hasattr(e.type, 'value') else e.type
            rows.append([e_type, dx, dy])
            entities.append((e_type, dx, dy))

        observer_type = observer.type.value if hasattr(observer.type, 'value') else observer.type
        rows.append([observer_type, 0.0, 0.0])
        observer_idx = len(rows) - 1
        num_nodes = len(rows)

        for i in range(num_nodes - 1):
            edges.append([i, observer_idx])
            edges.append([observer_idx, i])
        for i in range(num_nodes):
            edges.append([i, i])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(rows, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index)

    def _spawn(self, entity_type: EntityType, location: tuple[int, int]):
        x, y = location
        self.entities_pool.add(entity_type, location)
        self.grid[y, x] = self.display.get_color_index(self.entities_pool.entities[-1].color)

    def _remove(self, entity_id: int):
        entity = next((e for e in self.entities_pool.entities if e.id == entity_id), None)
        if entity:
            self.grid[entity.location] = 0
            self.entities_pool.remove(entity_id)

    def _run(self):
        plt.show()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x = int(event.xdata)
        y = int(event.ydata)

        entities = get_entities_at_location((x, y), self.entities_pool.entities)
        if self.action is ActionType.SPAWN and not entities and self.spawn_type is not None:
            self._spawn(self.spawn_type, (x, y))
            print(f"Spawned {self.spawn_type} at ({x}, {y})")
        elif self.action is ActionType.DELETE:
            if entities:
                entity = entities[0]
                self._remove(entity.id)
                print(f"Removed {entity.label} at ({x}, {y})")

    def _setup_ui(self):
        ax_button_play = plt.axes((0.8, 0.01, 0.1, 0.05))
        self.play_pause_button = Button(ax_button_play, 'Play')
        self.play_pause_button.on_clicked(self._toggle_play_pause)

        ax_button_reset = plt.axes((0.68, 0.01, 0.1, 0.05))
        self.reset_button = Button(ax_button_reset, 'Reset')
        self.reset_button.on_clicked(self._reset)

        ax_button_hunter = plt.axes((0.56, 0.01, 0.1, 0.05))
        self.button_hunter = Button(ax_button_hunter, 'Hunter')
        self.button_hunter.on_clicked(lambda event: self._set_spawn_type(EntityType.HUNTER))

        ax_button_prey = plt.axes((0.44, 0.01, 0.1, 0.05))
        self.button_prey = Button(ax_button_prey, 'Prey')
        self.button_prey.on_clicked(lambda event: self._set_spawn_type(EntityType.PREY))

        ax_button_wall = plt.axes((0.32, 0.01, 0.1, 0.05))
        self.button_wall = Button(ax_button_wall, 'Wall')
        self.button_wall.on_clicked(lambda event: self._set_spawn_type(EntityType.WALL))

        ax_button_delete = plt.axes((0.20, 0.01, 0.1, 0.05))
        self.button_delete = Button(ax_button_delete, 'Delete')
        self.button_delete.on_clicked(lambda event: self._set_action_type(ActionType.DELETE))

    def _setup_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _toggle_play_pause(self, _):
        self.is_running = not self.is_running
        self.play_pause_button.label.set_text('Play' if not self.is_running else 'Pause')
        self.fig.canvas.draw_idle()

    def _reset(self, _):
        self.grid.fill(0)
        self.display.update(self.grid)
        self.entities_pool.clear()
        self.fig.canvas.draw_idle()

    def _set_spawn_type(self, spawn_type: EntityType):
        self.action = ActionType.SPAWN
        self.spawn_type = spawn_type

    def _set_action_type(self, action_type: ActionType):
        self.action = action_type
        if action_type == ActionType.DELETE:
            self.spawn_type = None
        elif action_type == ActionType.WATCH:
            self.spawn_type = None

    def _update(self, _):
        if self.is_running:
            for entity in self.entities_pool.entities:
                x, y = entity.location
                self.grid[y, x] = 0
                g = self.build_graph_for_entity(entity, self.entities_pool.entities).to(self.device)
                with torch.no_grad():
                    _, _, action_logits = self.model(g)
                action_idx = action_logits.argmax().item()
                movement = list(Movement)[action_idx % len(Movement)]
                entity.move(movement, self.size, self.entities_pool.entities)
                x, y = entity.location
                self.grid[y, x] = self.display.get_color_index(entity.color)

        return self.display.update(self.grid)
