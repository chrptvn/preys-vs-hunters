import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for interactive GUI windows

from matplotlib import colors
import matplotlib.pyplot as plt

class GridDisplay:
    """
    A simple wrapper around matplotlib to visualize a 2D grid using imshow.

    This class handles:
    - Setting up the figure and axis
    - Drawing the grid with a customizable color map
    - Updating the grid image during animation
    """

    def __init__(self, grid, color_map=None):
        """
        Initialize the grid display.

        :param grid: A 2D NumPy array representing the Game of Life grid.
        :param color_map: Optional matplotlib ListedColormap object. Defaults to black/white.
        """
        self.grid = grid
        self.color_map = color_map or colors.ListedColormap(['black', 'white', 'red', 'blue', 'yellow'])

        # Set up the figure and axis
        self.fig, self.ax = plt.subplots()

        # Ensure the grid initializes with a drawn state
        self.grid[0, 0] = 1  # Temporarily set a pixel to trigger initial draw
        self.im = self.ax.imshow(
            self.grid,
            cmap=self.color_map,
            norm=colors.NoNorm(),
            interpolation='nearest',
            extent=(0, self.grid.shape[1], self.grid.shape[0], 0),
            origin='upper'
        )
        self.grid[0, 0] = 0  # Reset pixel after initialization

        # Hide axis ticks and labels
        self.ax.axis('off')

    def get_color_index(self, color):
        """
        Get the index of a color in the color map.

        :param color: The color to find.
        :return: The index of the color in the color map.
        """
        return self.color_map.colors.index(color)

    def update(self, new_grid):
        """
        Update the displayed grid with a new state.

        :param new_grid: The new 2D NumPy array to display.
        :return: A list containing the updated image (for FuncAnimation blitting).
        """
        self.grid = new_grid
        self.im.set_data(self.grid)
        return [self.im]

    def get_figure(self):
        """
        Get the matplotlib figure associated with this display.

        :return: The matplotlib figure instance.
        """
        return self.fig
