# sampler.py
# Implements sampling from a potential using overdamped Langevin dynamics.

import math
import torch
import numpy as np
import config as cfg

class LangevinSampler:
    """
    Generates samples from a potential energy surface using overdamped Langevin dynamics.
    The update rule is:
    x_{t+1} = x_t - (grad(V(x_t)) + grad(V_bias(x_t))) * dt + sqrt(2*dt/beta) * noise
    """
    def __init__(self, potential, beta, dt, n_steps, n_walkers, record_stride, device):
        self.potential = potential
        self.beta = beta
        self.dt = dt
        self.n_steps = n_steps
        self.n_walkers = n_walkers
        self.record_stride = record_stride
        self.device = device
        print(f"Langevin Sampler initialized: {n_walkers} walkers, {n_steps} steps, dt={dt:.2e}")

    @torch.no_grad()
    def sample(self, n_samples, initial_pos, bias_grad_fn=None):
        """
        Runs the Langevin simulation to generate samples.

        Args:
            n_samples (int): The total number of samples to return.
            initial_pos (np.ndarray): The starting position (e.g., cfg.A_CENTER).
            bias_grad_fn (callable, optional): A function that takes a position tensor
                                              and returns the gradient of the biasing
                                              potential. Defaults to None.

        Returns:
            torch.Tensor: A tensor of shape (n_samples, 2) containing the samples.
        """
        # Ensure initial_pos is a list of arrays for consistent processing
        if not isinstance(initial_pos, (list, tuple)):
            initial_pos_list = [initial_pos]
        else:
            initial_pos_list = initial_pos

        num_initial_positions = len(initial_pos_list)
        
        # Distribute walkers among initial positions
        walkers_per_pos = [self.n_walkers // num_initial_positions] * num_initial_positions
        for i in range(self.n_walkers % num_initial_positions):
            walkers_per_pos[i] += 1

        # Initialize walkers around the starting position
        all_initial_positions_tensors = []
        for i, pos_center in enumerate(initial_pos_list):
            num_walkers_for_this_pos = walkers_per_pos[i]
            pos_tensor = torch.tensor(pos_center, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(num_walkers_for_this_pos, 1)
            pos_tensor += torch.randn_like(pos_tensor) * 0.1 # Add initial noise
            all_initial_positions_tensors.append(pos_tensor)
        positions = torch.cat(all_initial_positions_tensors, dim=0)

        samples = []
        # The entire sampling process should not track gradients.
        # We only need the final positions, not the history of how they were generated.
        for step in range(self.n_steps):
            # Force from the main potential
            # We need to temporarily enable grad to compute the potential's gradient.
            # The positions tensor itself should not track history across steps.
            with torch.enable_grad():
                force = -self.potential.gradient(positions)
                force = force.detach() # CRITICAL: Detach the force from its graph

            # Add force from the biasing potential if provided
            if bias_grad_fn is not None:
                with torch.enable_grad():
                    bias_force = bias_grad_fn(positions)
                    bias_force = bias_force.detach() # CRITICAL: Detach the bias_force
                    force += bias_force

            # Langevin dynamics update
            noise = torch.randn_like(positions)
            positions = positions + force * self.dt + math.sqrt(2 * self.dt / self.beta) * noise
            if step % self.record_stride == 0:
                samples.append(positions.cpu().clone())

        # Concatenate and select the desired number of samples
        all_samples = torch.cat(samples, dim=0)
        # Shuffle to ensure random selection
        indices = torch.randperm(all_samples.size(0))
        return all_samples[indices][:n_samples].to(self.device)
