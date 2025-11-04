# train.py
# Main script to orchestrate the iterative training and sampling process.
# REFACTORED: Unified training loop. The first iteration is an unbiased run
# that generates the boundary condition dataset and performs the first full training.

import torch
import time
import os
import numpy as np
import datetime

# Import project modules
import config as cfg
from potential import MullerBrown
from model import SmallNet
from sampler import LangevinSampler
from losses import calculate_variational_losses, calculate_boundary_loss
from plotting import plot_results, plot_iteration_feedback, plot_sampling_feedback, create_gif_from_plots
from bias import BiasManager, OPESBias, KolmogorovBias

def main():
    """Main training loop."""
    # --- Experiment Setup ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{timestamp}"
    experiment_dir = os.path.join("experiments", run_name)
    bias_output_dir = os.path.join(experiment_dir, "bias_potentials")
    iteration_plot_dir = os.path.join(experiment_dir, "iteration_plots")
    os.makedirs(iteration_plot_dir, exist_ok=True)
    print(f"Saving results to: {os.path.abspath(experiment_dir)}")

    # --- Initialization ---
    potential = MullerBrown(device=cfg.DEVICE)
    model = SmallNet().to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    sampler = LangevinSampler(
        potential=potential, beta=cfg.BETA, dt=cfg.LANGEVIN_DT,
        n_steps=cfg.LANGEVIN_N_STEPS, N_WALKERS=cfg.N_WALKERS,
        record_stride=cfg.LANGEVIN_RECORD_STRIDE, device=cfg.DEVICE
    )

    # --- Main Bias Manager (used for iterations > 0) ---
    bias_manager = BiasManager(model=model, device=cfg.DEVICE)
    opes_bias = OPESBias(
        model=model, bias_factor=cfg.OPES_BIAS_FACTOR,
        kernel_sigma=cfg.OPES_KERNEL_SIGMA, delta_E=cfg.OPES_DELTA_E,
        device=cfg.DEVICE, grid_min=cfg.OPES_CV_MIN, grid_max=cfg.OPES_CV_MAX, grid_bins=cfg.OPES_CV_BINS
    )
    bias_manager.add_bias(opes_bias)
    
    if cfg.W_KOLMOGOROV > 0.0:
        kolmogorov_bias = KolmogorovBias(
            lambda_k=cfg.W_KOLMOGOROV, device=cfg.DEVICE
        )
        bias_manager.add_bias(kolmogorov_bias)

    # --- Data Placeholders ---
    unbiased_loader = None
    latest_samples = None
    start_time = time.time()

    # --- Unified Iterative Training and Sampling Loop ---
    for iteration in range(cfg.N_ITERATIONS):
        iteration_num = iteration + 1
        print(f"\n--- Iteration {iteration_num}/{cfg.N_ITERATIONS} ---")
        model.eval()

        # --- 1. Sampling Step ---
        if iteration == 0:
            print("Running initial UNBIASED sampling...")
            # Use a dummy, empty bias manager for the first unbiased run
            sampling_bias_manager = BiasManager(model, cfg.DEVICE)
            weights = None # Will be set to ones after sampling
            n_samples_per_iter = cfg.N_SAMPLES_UNBIASED_INITIAL
            
        else: # Biased sampling for all subsequent iterations
            print("Running BIASED sampling (with on-the-fly OPES convergence)...")
            bias_manager.reset_all_states()
            sampling_bias_manager = bias_manager
            n_samples_per_iter = cfg.N_SAMPLES_PER_ITER
        
        new_samples = sampler.sample(
            n_samples_per_iter,
            initial_pos=[cfg.A_CENTER, cfg.B_CENTER], 
            bias_manager=sampling_bias_manager,
            iteration=iteration_num, experiment_dir=experiment_dir
        )

        if new_samples.shape[0] == 0:
            print("Warning: No samples collected in this iteration. Ending run.")
            break
        
        latest_samples = new_samples.detach().clone()

        # --- 2. Plot Sampling & Calculate Weights ---
        plot_sampling_feedback(model, potential, new_samples, iteration_num, sampling_bias_manager, output_dir=iteration_plot_dir)

        # Create a GIF of the OPES convergence from this iteration's sampling run
        if iteration > 0:
            opes_plots_dir = os.path.join(experiment_dir, "opes_convergence", f"iter_{iteration_num:02d}")
            create_gif_from_plots(opes_plots_dir, iteration_num)

        if iteration == 0:
            weights = torch.ones(new_samples.shape[0], device=cfg.DEVICE)
            # The first batch of samples becomes the fixed dataset for boundary conditions
            unbiased_dataset = torch.utils.data.TensorDataset(new_samples.detach().clone())
            unbiased_loader = torch.utils.data.DataLoader(unbiased_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
            print(f"Generated {new_samples.size(0)} unbiased samples, which will be used for all boundary loss calculations.")
        else:
            # We need to calculate gradients w.r.t. samples for the Kolmogorov bias
            samples_for_bias_calc = new_samples.detach().clone().requires_grad_(True)
            v_biases = bias_manager.calculate_bias_potential(samples_for_bias_calc)
            v_total_bias = v_biases['total'].detach()
            weights = torch.exp(cfg.BETA * v_total_bias)
            
            max_weight = torch.quantile(weights, 0.99) if weights.numel() > 10 else 1000.0
            weights.clamp_(max=max_weight)
            weights /= weights.mean()
            print(f"Collected {new_samples.size(0)} production samples. Mean weight: {weights.mean():.4f}")

            # Save the converged OPES bias potential for this iteration
            if opes_bias:
                os.makedirs(bias_output_dir, exist_ok=True)
                cv_grid, v_bias_grid = opes_bias.get_potential_on_grid()
                save_path = os.path.join(bias_output_dir, f"opes_bias_iter_{iteration_num:02d}.npz")
                np.savez(save_path, 
                         cv_grid=cv_grid.cpu().numpy(), 
                         v_bias_grid=v_bias_grid.cpu().detach().numpy())
                print(f"  Saved converged OPES bias to {save_path}")

        # --- 3. Unified Training Step ---
        print("Training the model...")
        biased_dataset = torch.utils.data.TensorDataset(new_samples, weights)
        biased_loader = torch.utils.data.DataLoader(biased_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
        
        for epoch in range(1, cfg.N_EPOCHS_PER_ITER + 1):
            model.train()
            unbiased_iter = iter(unbiased_loader)
            
            for batch_samples, batch_weights in biased_loader:
                try:
                    batch_unbiased, = next(unbiased_iter)
                except StopIteration:
                    unbiased_iter = iter(unbiased_loader)
                    batch_unbiased, = next(unbiased_iter)

                batch_samples.requires_grad_(True)
                v_batch = potential.potential(batch_samples)
                var_losses = calculate_variational_losses(model, batch_samples, v_batch, batch_weights)
                
                dist_A = torch.linalg.norm(batch_unbiased - torch.tensor(cfg.A_CENTER, device=cfg.DEVICE), axis=-1)
                dist_B = torch.linalg.norm(batch_unbiased - torch.tensor(cfg.B_CENTER, device=cfg.DEVICE), axis=-1)
                masks = {'A': dist_A < cfg.RADIUS, 'B': dist_B < cfg.RADIUS}
                boundary_loss = calculate_boundary_loss(model, batch_unbiased, masks)
                
                total_loss = var_losses['total_variational'] + cfg.W_UNBIASED_BOUND * boundary_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            if epoch % cfg.LOG_LOSS_EVER_N_EPOCHS == 0:
                print(f"  Epoch {epoch}/{cfg.N_EPOCHS_PER_ITER} | Total Loss: {total_loss.item():.3e} ")

        # --- 4. Plot Training Feedback ---
        plot_iteration_feedback(model, potential, iteration_num, output_dir=iteration_plot_dir)

    end_time = time.time()
    print(f"\nTraining complete in {end_time - start_time:.2f} seconds.")

    # --- Final Evaluation and Visualization ---
    plot_results(model, potential, latest_samples, bias_manager, output_dir=experiment_dir)

if __name__ == '__main__':
    main()
