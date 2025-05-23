from enum import Enum
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from ai.Brain import decide_movement
from ai.HunterModel import HunterModel
from preys_vs_hunters.display import GridDisplay
from preys_vs_hunters.entities.entity import EntityType, Movement, Entity
from preys_vs_hunters.entities.pool import Pool
from preys_vs_hunters.utils import get_local_observation, get_entities_at_location


class ActionType(Enum):
    SPAWN = 0
    DELETE = 1
    WATCH = 2


class PreysVsHunters:
    def __init__(self, size=(100, 100), interval=200):
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HunterModel()
        self.model.load_state_dict(torch.load("C:\\Users\\cacpot\\projects\\perso\\preys-vs-hunters\\hunter.pt",
                                         map_location=torch.device('cpu')))
        self.model.eval()

    def _spawn(self, entity_type: EntityType, location: tuple[int, int]):
        self.entities_pool.add(entity_type, location)
        self.grid[location] = self.display.get_color_index(self.entities_pool.entities[-1].color)

    def _remove(self, entity_id: int):
        entity = next((e for e in self.entities_pool.entities if e.id == entity_id), None)
        if entity:
            self.grid[entity.location] = 0
            self.entities_pool.remove(entity_id)

    def _move_to(self, entity: Entity, movement: Movement):
        # Clear the old location on the grid
        self.grid[entity.location] = 0
        # Get the new location
        new_location = self._get_new_location(entity, movement)

        # Update the entity’s internal location
        entity.location = new_location

        # Paint the new location on the grid
        self.grid[new_location] = self.display.get_color_index(entity.color)

    def _get_new_location(self, entity: Entity, movement: Movement):
        direction = (0, 0)
        if movement is Movement.UP:
            direction = (-1, 0)
        elif movement is Movement.DOWN:
            direction = (1, 0)
        elif movement is Movement.LEFT:
            direction = (0, -1)
        elif movement is Movement.RIGHT:
            direction = (0, 1)
        elif movement is Movement.UPPER_LEFT:
            direction = (-1, -1)
        elif movement is Movement.UPPER_RIGHT:
            direction = (-1, 1)
        elif movement is Movement.LOWER_LEFT:
            direction = (1, -1)
        elif movement is Movement.LOWER_RIGHT:
            direction = (1, 1)

        # Compute new location with toroidal wrapping
        x_max, y_max = self.grid.shape
        dx, dy = direction
        new_x = (entity.location[0] + dx) % x_max
        new_y = (entity.location[1] + dy) % y_max
        return (new_x, new_y)

    def _run(self):
        plt.show()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x = int(event.ydata)
        y = int(event.xdata)

        entities = get_entities_at_location((x, y), self.entities_pool.entities)
        if self.action is ActionType.SPAWN and not entities and self.spawn_type:
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

    def _is_blocked(self, entityType: EntityType, entities_at_location: List[Entity]):
        if entityType is EntityType.HUNTER:
            for entity in entities_at_location:
                print(f"Entity {entity.id} blocked by {entity.type} at {entity.location}")
                if entity.type is EntityType.WALL or entity.type is EntityType.HUNTER:
                    return True
                elif entity.type is EntityType.PREY:
                    return False

        return False

    def _update(self, _):
        if self.is_running:

            for entity in self.entities_pool.entities:
                new_location = entity.location
                best_move = None

                if entity.type is EntityType.HUNTER:
                    local_observation = get_local_observation(entity.location, self.grid, self.entities_pool.entities)
                    best_move = decide_movement(entity.id, local_observation, self.model, self.device)

                    new_location = self._get_new_location(entity, best_move)

                    entities_at_new_location = get_entities_at_location(new_location, self.entities_pool.entities)

                    if not self._is_blocked(entity.type, entities_at_new_location):
                        self._move_to(entity, best_move)
                        for entity_at_new_location in entities_at_new_location:
                            self.entities_pool.entities.remove(entity_at_new_location)

                elif entity.type is EntityType.PREY:
                    pass


        return self.display.update(self.grid)
