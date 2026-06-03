import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScoreNet(nn.Module):
    """
    Small MLP that predicts the noise epsilon added at timestep t.
    Input: (x_t, t) — noisy 2D point and timestep
    Output: predicted noise epsilon (2D)
    """
    def __init__(self, d_hidden=256, n_layers=4, T=500):
        super().__init__()
        self.T = T
        d_time = 32

        self.time_mlp = nn.Sequential(
            nn.Linear(d_time, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.input_proj = nn.Linear(2, d_hidden)
        self.layers = nn.ModuleList([
            nn.Linear(d_hidden, d_hidden) for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_hidden, 2)

    def sinusoidal(self, t):
        half = 16
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, 32)

    def forward(self, x, t):
        h = self.input_proj(x) + self.time_mlp(self.sinusoidal(t))
        for layer in self.layers:
            h = F.silu(layer(h))
        return self.output(h)
