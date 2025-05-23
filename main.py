from game_of_life.controller import GameOfLife
from preys_vs_hunters.controller import PreysVsHunters

# -----------------------------------------------------------
# Entry point for Conway's Game of Life
#
# This script initializes the GameOfLife controller and
# launches the simulation using Matplotlib for display.
#
# Grid size and animation interval can be adjusted as needed.
# -----------------------------------------------------------

if __name__ == '__main__':
    # Initialize the game with a 100x100 grid and 200ms update interval
    game = PreysVsHunters(size=(50, 50), interval=50)

    # Start the simulation
    game._run()
