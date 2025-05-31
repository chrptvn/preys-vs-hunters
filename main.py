import torch

from ai.models import HunterModel
from preys_vs_hunters.controller import PreysVsHunters

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hunter_model = HunterModel(model_file="hunter.pt", device=device)

    game = PreysVsHunters(hunter_model=hunter_model, device=device, size=(50, 50), interval=500)

    # Start the simulation
    game._run()
