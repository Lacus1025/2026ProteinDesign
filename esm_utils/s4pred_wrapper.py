import sys
import os
import torch

S4PRED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.pardir, "s4pred")
S4PRED_DIR = os.path.abspath(S4PRED_DIR)
if S4PRED_DIR not in sys.path:
    sys.path.insert(0, S4PRED_DIR)

from network import S4PRED
from utilities import aas2int


class S4PredWrapper:
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = S4PRED().to(self.device)
        self.model.eval()

        weight_dir = os.path.join(S4PRED_DIR, "weights")
        for i in range(5):
            weight_file = os.path.join(weight_dir, f"weights_{i+1}.pt")
            sub_model = getattr(self.model, f"model_{i+1}")
            state_dict = torch.load(weight_file, map_location=self.device)
            sub_model.load_state_dict(state_dict)

        self.model.requires_grad_(False)

    @torch.no_grad()
    def predict(self, sequence: str) -> str:
        int_seq = aas2int(sequence)
        tensor = torch.tensor([int_seq], device=self.device)
        output = self.model(tensor)
        ss_indices = output.argmax(-1).cpu().numpy()
        ind2char = {0: "C", 1: "H", 2: "E"}
        return "".join(ind2char[i] for i in ss_indices)
