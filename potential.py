# potential.py
# Defines the potential energy surface.

import torch
import config as cfg

class MullerBrown:
    """
    Implements the Müller-Brown potential and its gradient using PyTorch.
    This allows for automatic differentiation and seamless integration with the sampler.
    """
    def __init__(self, device=cfg.DEVICE):
        # Move all parameters to the specified device
        self.A = torch.tensor(cfg.A, dtype=torch.float32, device=device)
        self.a = torch.tensor(cfg.a, dtype=torch.float32, device=device)
        self.b = torch.tensor(cfg.b, dtype=torch.float32, device=device)
        self.c = torch.tensor(cfg.c, dtype=torch.float32, device=device)
        self.x0 = torch.tensor(cfg.x0, dtype=torch.float32, device=device)
        self.y0 = torch.tensor(cfg.y0, dtype=torch.float32, device=device)
        self.offset = cfg.POTENTIAL_OFFSET
        self.device = device

    def potential(self, xy):
        """
        Calculates the Müller-Brown potential V(x, y).

        Args:
            xy (torch.Tensor): A tensor of shape (N, 2) with coordinates.

        Returns:
            torch.Tensor: A tensor of shape (N,) with potential energy values.
        """
        x = xy[..., 0]
        y = xy[..., 1]
        val = torch.zeros_like(x)
        for i in range(4):
            val += self.A[i] * torch.exp(
                self.a[i] * (x - self.x0[i])**2 +
                self.b[i] * (x - self.x0[i]) * (y - self.y0[i]) +
                self.c[i] * (y - self.y0[i])**2
            )
        return val + self.offset

    def gradient(self, xy):
        """
        Calculates the gradient of the Müller-Brown potential, -F.

        Args:
            xy (torch.Tensor): A tensor of shape (N, 2) with coordinates.
                               Must have requires_grad=True.

        Returns:
            torch.Tensor: A tensor of shape (N, 2) with the gradient [dV/dx, dV/dy].
        """
        xy_clone = xy.clone().requires_grad_(True)
        V = self.potential(xy_clone)
        # torch.autograd.grad computes sum(V), so we can pass it directly
        grad_V = torch.autograd.grad(
            outputs=V,
            inputs=xy_clone,
            grad_outputs=torch.ones_like(V),
            create_graph=False,
            retain_graph=False
        )[0]
        return grad_V
