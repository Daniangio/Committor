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
    def __init__(self, domain_min=None, domain_max=None, periodicity_info=None, hidden=cfg.HIDDEN_UNITS):
        super().__init__()
        # --- Set defaults if no domain/periodicity info is provided ---
        if domain_min is None:
            domain_min = [cfg.X_MIN, cfg.Y_MIN]
        if domain_max is None:
            domain_max = [cfg.X_MAX, cfg.Y_MAX]
        if periodicity_info is None:
            # Default to non-periodic for all dimensions
            num_dims = len(domain_min)
            periodicity_info = [{'periodic': False, 'period': None} for _ in range(num_dims)]

        # Input normalization layer to scale inputs to [-1, 1]
        # We register them as buffers so they are moved to the correct device
        # automatically with .to(device) and are saved with the model state_dict.
        domain_min = torch.tensor(domain_min, dtype=torch.float32)
        domain_max = torch.tensor(domain_max, dtype=torch.float32)
        self.register_buffer('domain_min', domain_min)
        self.register_buffer('domain_max', domain_max)
        self.register_buffer('domain_range', domain_max - domain_min)
        self.periodicity_info = periodicity_info

        # Determine the input dimension based on periodicity
        input_dim = 0
        for dim_info in self.periodicity_info:
            input_dim += 2 if dim_info['periodic'] else 1

        self.net_g = nn.Sequential(
            nn.Linear(input_dim, hidden),
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
            nn.Linear(input_dim, hidden),
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
        transformed_inputs = []
        for i, dim_info in enumerate(self.periodicity_info):
            coord = x[:, i]
            if dim_info['periodic']:
                period = dim_info['period']
                if period is None:
                    raise ValueError(f"Period must be specified for periodic dimension {i}")
                # Scale to [0, 2*pi]
                scaled_coord = 2 * torch.pi * (coord - self.domain_min[i]) / period
                transformed_inputs.append(torch.sin(scaled_coord))
                transformed_inputs.append(torch.cos(scaled_coord))
            else:
                # For non-periodic dimensions, just normalize to [-1, 1]
                normalized_coord = 2 * (coord - self.domain_min[i]) / self.domain_range[i] - 1
                transformed_inputs.append(normalized_coord)

        # Concatenate along the feature dimension
        final_input = torch.stack(transformed_inputs, dim=1)

        g = self.net_g(final_input)
        gA = g[..., 0]
        gB = g[..., 1]

        z = self.net_z(final_input).squeeze(-1)
        q = torch.sigmoid(z)
        return gA, gB, z, q, self.alpha
