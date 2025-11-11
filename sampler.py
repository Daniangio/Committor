# sampler.py
# Implements sampling from a potential using overdamped Langevin dynamics.
# REFACTORED:
# 1. Drives the on-the-fly OPES convergence process.
# 2. Manages two phases: equilibration (building bias) and production (recording samples).

from bias import OPESBias
import math
import torch
import numpy as np
import os
from plotting import plot_opes_convergence_step
import config as cfg

class LangevinSampler:
    """
    Generates samples from a potential energy surface using overdamped Langevin dynamics.
    Drives the on-the-fly convergence of an OPES bias if provided.
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

    def sample(self, n_samples, initial_pos, bias_manager, iteration, experiment_dir):
        """
        Runs the Langevin simulation, managing OPES convergence before sampling.
        """
        opes_plots_dir = os.path.join(experiment_dir, "opes_convergence", f"iter_{iteration:02d}")

        # --- Walker Initialization ---
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

        # --- Sampler State Setup ---
        opes_bias = next((b for b in bias_manager.biases if isinstance(b, OPESBias)), None)
        
        # Flag to track if OPES has converged in the equilibration phase
        opes_converged_in_equilibration = False
        last_opes_max_diff = float('inf') # To store the last convergence metric
        
        # Use a separate variable for current positions to avoid confusion with initial 'positions'
        current_positions = positions.clone() 

        samples = []
        
        bias_grad_fn = bias_manager.get_bias_grad_fn()
        
        print("  Beginning Langevin dynamics...")

        # --- Phase 1: Equilibration (OPES Convergence) ---
        if opes_bias:
            os.makedirs(opes_plots_dir, exist_ok=True)
            print(f"    Running equilibration for OPES bias (max {cfg.MAX_OPES_EQUILIBRATION_STEPS} steps)...")
            for eq_step in range(1, cfg.MAX_OPES_EQUILIBRATION_STEPS + 1):
                force = -self.potential.gradient(current_positions)
                # Bias force during equilibration still uses the current, evolving bias
                force += bias_grad_fn(current_positions)

                noise = torch.randn_like(current_positions)
                current_positions += force * self.dt + math.sqrt(2 * self.dt / self.beta) * noise

                if eq_step % cfg.OPES_STRIDE == 0:
                    opes_bias.add_kernels(current_positions)
                    conv_data = opes_bias.check_convergence()
                    last_opes_max_diff = conv_data["max_diff"]

                    # Call the new plotting function
                    plot_opes_convergence_step(eq_step=eq_step, iteration=iteration, output_dir=opes_plots_dir, **conv_data)

                    if last_opes_max_diff < cfg.OPES_CONV_TOL:
                        opes_converged_in_equilibration = True
                        print(f"    OPES bias converged after {eq_step} equilibration steps (max_diff: {last_opes_max_diff:.3e}).")
                        break
            
            if not opes_converged_in_equilibration:
                print(f"    Warning: OPES bias did not converge within {cfg.MAX_OPES_EQUILIBRATION_STEPS} equilibration steps (last max_diff: {last_opes_max_diff:.3e}).")
                print(f"    Proceeding to production run with current bias state.")
        else:
            print("    No OPES bias configured, skipping equilibration phase.")

        # --- Phase 2: Production (Record Samples) ---
        print(f"  Running production for {self.n_steps} steps...")
        
        for prod_step in range(1, self.n_steps + 1): # Loop for the specified number of production steps
            force = -self.potential.gradient(current_positions)
            # Bias force continues to be applied, and OPES continues to update (on-the-fly)
            force += bias_grad_fn(current_positions) 
            
            noise = torch.randn_like(current_positions)
            current_positions += force * self.dt + math.sqrt(2 * self.dt / self.beta) * noise
            
            # OPES continues to update its state even during production, as it's "on-the-fly"
            if opes_bias and prod_step % cfg.OPES_STRIDE == 0:
                opes_bias.add_kernels(current_positions)
                # We don't need to check convergence here to stop production,
                # as production runs for a fixed self.n_steps.
                # However, check_convergence updates v_bias_grid_old, so it's good to call it.
                opes_bias.check_convergence() 

            if prod_step % self.record_stride == 0:
                samples.append(current_positions.detach().cpu().clone())

        # --- Collate and Subsample Results ---
        if not samples:
            return torch.empty(0, 2, device=self.device)

        all_samples = torch.cat(samples, dim=0).requires_grad_(False)
        indices = torch.randperm(all_samples.size(0))
        
        if n_samples is None: # Return all collected production samples
            return all_samples[indices].to(self.device)
        else: # Subsample to the desired number
            return all_samples[indices][::max(1, len(all_samples) // n_samples)].to(self.device)
