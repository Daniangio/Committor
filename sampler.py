# sampler.py
# Implements sampling from a potential using overdamped Langevin dynamics.

from bias import OPESBias
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
    def __init__(self, potential, beta, dt, n_steps, N_WALKERS, record_stride, device):
        self.potential = potential
        self.beta = beta
        self.dt = dt
        self.n_steps = n_steps
        self.N_WALKERS = N_WALKERS
        self.record_stride = record_stride
        self.device = device
        print(f"Langevin Sampler initialized: {N_WALKERS} walkers, {n_steps} steps, dt={dt:.2e}")

    def sample(self, n_samples, initial_pos, bias_manager):
        """
        Runs the Langevin simulation to generate samples.

        Args:
            n_samples (int|None): The total number of samples to return.
            initial_pos (np.ndarray or list of np.ndarray): The starting position(s).
            bias_manager (BiasManager): The bias manager to calculate bias forces.

        Returns:
            torch.Tensor: A tensor of shape (n_samples, 2) containing the samples.
        """
        if not isinstance(initial_pos, (list, tuple)):
            initial_pos_list = [initial_pos]
        else:
            initial_pos_list = initial_pos

        num_initial_positions = len(initial_pos_list)
        walkers_per_pos = [self.N_WALKERS // num_initial_positions] * num_initial_positions
        for i in range(self.N_WALKERS % num_initial_positions):
            walkers_per_pos[i] += 1

        all_initial_positions_tensors = []
        for i, pos_center in enumerate(initial_pos_list):
            num_walkers_for_this_pos = walkers_per_pos[i]
            pos_tensor = torch.tensor(pos_center, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(num_walkers_for_this_pos, 1)
            pos_tensor += torch.randn_like(pos_tensor) * 0.1
            all_initial_positions_tensors.append(pos_tensor)
        positions = torch.cat(all_initial_positions_tensors, dim=0)

        # Find the on-the-fly OPES bias object, if it exists
        opes_bias = None
        if bias_manager:
            print("  Bias manager provided. Bias forces will be included in dynamics.")
            for bias in bias_manager.biases:
                if isinstance(bias, OPESBias):
                    opes_bias = bias
                    print("  On-the-fly OPES mode enabled in sampler.")
                    break

        production_started = False if opes_bias else True
        samples = []

        for step in range(self.n_steps):
            force = -self.potential.gradient(positions)

            force += bias_manager.calculate_bias_force(positions)

            noise = torch.randn_like(positions)
            positions += force * self.dt + math.sqrt(2 * self.dt / self.beta) * noise
            
            # --- On-the-fly OPES Logic ---
            if opes_bias and not opes_bias.is_converged:
                # Deposit kernels at specified stride
                if step > 0 and step % cfg.OPES_STRIDE == 0:
                    opes_bias.add_kernels(positions)
                    
                    # Check for convergence after depositing
                    if opes_bias.check_convergence():
                        production_started = True
                        print(f"    Production sampling started at step {step}.")
            
            # --- Sample Recording ---
            # Record samples only during production phase
            if production_started and step % self.record_stride == 0:
                samples.append(positions.cpu().clone())
        
        all_samples = torch.cat(samples, dim=0)
        indices = torch.randperm(all_samples.size(0))
        if n_samples is None:
            n_samples = all_samples.size(0)
        return all_samples[indices][::max(1, len(all_samples) // n_samples)].to(self.device).detach()
