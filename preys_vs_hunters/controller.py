from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from preys_vs_hunters.display import GridDisplay
from preys_vs_hunters.entities.entity import EntityType
from preys_vs_hunters.entities.pool import Pool
from preys_vs_hunters.utils import get_local_observation, get_entities_at_location, \
    get_entities_by_distance, calculate_distance_reward


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

    def _spawn(self, entity_type: EntityType, location: tuple[int, int]):
        self.entities_pool.add(entity_type, location)
        self.grid[location] = self.display.get_color_index(self.entities_pool.entities[-1].color)

    def _remove(self, entity_id: int):
        entity = next((e for e in self.entities_pool.entities if e.id == entity_id), None)
        if entity:
            self.grid[entity.location] = 0
            self.entities_pool.remove(entity_id)

    def _move(self, entity_id: int, direction: tuple[int, int]):
        # Find the entity by ID
        entity = next((e for e in self.entities_pool.entities if e.id == entity_id), None)
        if not entity:
            return  # No such entity, do nothing

        # Compute new location with toroidal wrapping
        x_max, y_max = self.grid.shape
        dx, dy = direction
        new_x = (entity.location[0] + dx) % x_max
        new_y = (entity.location[1] + dy) % y_max
        old_location = entity.location
        new_location = (new_x, new_y)

        if entity.type is EntityType.HUNTER:
            old_observation = get_local_observation(old_location, self.grid, self.entities_pool.entities)
            new_observation = get_local_observation(new_location, self.grid, self.entities_pool.entities)
            old_distances = get_entities_by_distance(old_observation, entity.id, [EntityType.PREY])
            new_distances = get_entities_by_distance(new_observation, entity.id, [EntityType.PREY])
            score = calculate_distance_reward(old_distances, new_distances)
            print(f"Score for {entity.label} moving from {old_location} to {new_location}: {score}")

        # Clear the old location on the grid
        self.grid[entity.location] = 0

        # Update the entity’s internal location
        entity.location = new_location

        # Paint the new location on the grid
        self.grid[new_location] = self.display.get_color_index(entity.color)

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

    def _update(self, _):
        if self.is_running:

            for entity in self.entities_pool.entities:
                if entity.type is EntityType.HUNTER:
                    self._move(entity.id, (np.random.randint(-1, 2), np.random.randint(-1, 2)))
                    # self._move(entity.id, (0,0))
                elif entity.type is EntityType.PREY:
                    # self._move(entity.id, (np.random.randint(-1, 2), np.random.randint(-1, 2)))
                    self._move(entity.id, (0, 0))

        return self.display.update(self.grid)
