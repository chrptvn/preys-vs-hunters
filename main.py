import torch

from ai.model import SwarmBrain
from simulation.simulation import Simulation

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwarmBrain()

    model.load_state_dict(torch.load("swarm_brain.pt", map_location=device))
    model.to(device)
    model.eval()

    game = Simulation(model=model, device=device, grid_size=(50, 50), interval=100)

    # Start the simulation
    game._run()
