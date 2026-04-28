import torch
import torch.nn as nn

class SANA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, stress=0.0):
        noise = stress * torch.randn_like(x)
        return self.net(x + noise)
