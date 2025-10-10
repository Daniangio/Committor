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
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )
        # g is the reaction coordinate or quasipotential (unbounded)
        self.g_head = nn.Linear(hidden, 1)
        # z is the logit for the committor q (bounded in [0, 1])
        self.z_head = nn.Linear(hidden, 1)

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
        h = self.net(x)
        g = self.g_head(h).squeeze(-1)
        # Pass the logit z through a sigmoid to ensure q is in [0, 1]
        z = self.z_head(h).squeeze(-1)
        q = torch.sigmoid(z)
        return g, z, q, self.alpha
