import torch

from ai.model import SwarmBrain
from preys_vs_hunters.controller import PreysVsHunters

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwarmBrain()
    model.load_state_dict(torch.load("swarmbrain.pt", map_location=device))
    model.to(device)
    model.eval()  # optional, disables dropout/batchnorm if present

    game = PreysVsHunters(model=model, device=device, size=(50, 50), interval=100)

    # Start the simulation
    game._run()
