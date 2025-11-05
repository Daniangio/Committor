# model.py
# Defines the neural network architecture for learning g and q.

import torch
from torch import nn
import config as cfg

class SmallNet(nn.Module):
    """
    A small neural network that learns both a reaction coordinate 'g' and
    the committor 'q'. It uses a shared backbone and two separate heads.
    """
    def __init__(self, hidden=cfg.HIDDEN_UNITS):
        super().__init__()
        # Input normalization layer to scale inputs to [-1, 1]
        # We register them as buffers so they are moved to the correct device
        # automatically with .to(device) and are saved with the model state_dict.
        domain_min = torch.tensor([cfg.X_MIN, cfg.Y_MIN], dtype=torch.float32)
        domain_max = torch.tensor([cfg.X_MAX, cfg.Y_MAX], dtype=torch.float32)
        self.register_buffer('domain_min', domain_min)
        self.register_buffer('domain_max', domain_max)
        self.register_buffer('domain_range', domain_max - domain_min)

        self.net_g = nn.Sequential(
            nn.Linear(2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2)
        )

        self.net_z = nn.Sequential(
            nn.Linear(2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )

        # A learnable parameter to link g and q via a sigmoid
        self.alpha = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 2).

        Returns:
            tuple: (g, q, alpha)
                g (torch.Tensor): The reaction coordinate, shape (N,).
                q (torch.Tensor): The committor probability, shape (N,).
                alpha (torch.Tensor): The learnable linking parameter.
        """
        # Normalize input x from [min, max] to [-1, 1]
        x_normalized = 2 * (x - self.domain_min) / self.domain_range - 1

        g = self.net_g(x_normalized).squeeze(-1)
        gA = g[..., 0]
        gB = g[..., 1]

        z = self.net_z(x_normalized).squeeze(-1)
        q = torch.sigmoid(z)
        return gA, gB, z, q, self.alpha
