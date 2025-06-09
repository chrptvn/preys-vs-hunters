# Preys vs Hunters Simulation

A grid-based simulation using PyTorch Geometric and Matplotlib where hunters chase preys, preys flee, and behavior is learned using a Graph Neural Network (GNN) model.

## Features

- Interactive GUI built with Matplotlib.
- GNN-based agent brain (`SwarmBrain`) using PyTorch Geometric.
- Toroidal grid (wraparound movement).
- Multiple entity types:
  - Preys
  - Hunters
  - Walls (static obstacles)
- Training script with curriculum-style entity scaling.
- Spawn, delete, and observe modes.

## Requirements

- Python 3.10+
- PyTorch
- PyTorch Geometric
- Matplotlib
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
preys-vs-hunters/
├── ai/
│   └── model.py             # GNN model: SwarmBrain
├── simulation/
│   ├── controller.py        # Simulation logic and GUI
│   ├── entities/            # Entity types (Hunter, Prey, Wall, etc.)
│   ├── utils.py             # Utility functions (distance, delta, graph building)
│   ├── display.py           # Grid display rendering logic
│   
├── main.py                  # Entry point to run the simulation
├── train.py                 # Training loop for SwarmBrain
├── requirements.txt
└── README.md
```

## Usage

### Training the model

Run the training loop:

```bash
python train.py
```

This trains the `SwarmBrain` model using progressively more complex environments and saves the final model as `swarm_brain.pt`.

### Running the simulation

After training, run the main simulation:

```bash
python main.py
```

### Controls

- Hunter / Prey / Wall: Select what to spawn
- Delete: Remove entities by clicking
- Play/Pause: Start or stop simulation
- Reset: Clear the grid

### Grid Behavior

- Grid wraps around (toroidal)
- Entities move simultaneously each frame
- Hunters learn to chase; preys learn to flee

---

Created by [Christian Potvin](https://www.linkedin.com/in/christian-potvin-62b27156/)

