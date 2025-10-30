# train.py
# Main script to orchestrate the iterative training and sampling process.
# REFACTORED: Implements static biasing, decoupled losses, and separate data handling.

import torch
import time
import os
import datetime

# Import project modules
import config as cfg
from potential import MullerBrown
from model import SmallNet
from sampler import LangevinSampler
from losses import calculate_variational_losses, calculate_boundary_loss
from plotting import plot_results, plot_iteration_feedback, plot_sampling_feedback
from bias import BiasManager, OPESBias, KolmogorovBias

def main():
    """Main training loop."""
    # --- Experiment Setup ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join("experiments", f"run_{timestamp}")
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

    # --- Bias Setup ---
    bias_manager = BiasManager(model=model, device=cfg.DEVICE)
    cv_func = lambda x: model(x)[1]
    
    if cfg.OPES_ENABLED:
        opes_bias = OPESBias(
            cv_func=cv_func, bias_factor=cfg.OPES_BIAS_FACTOR,
            kernel_sigma=cfg.OPES_KERNEL_SIGMA, device=cfg.DEVICE
        )
        bias_manager.add_bias(opes_bias)
        
    if cfg.LAMBDA_KOLMOGOROV > 0:
        kolmogorov_bias = KolmogorovBias(
            lambda_k=cfg.LAMBDA_KOLMOGOROV, device=cfg.DEVICE
        )
        bias_manager.add_bias(kolmogorov_bias)

    # --- Data Generation for Boundary Conditions ---
    print("\n--- Generating initial unbiased dataset for boundary conditions ---")
    unbiased_samples = sampler.sample(cfg.N_SAMPLES_UNBIASED_INITIAL, initial_pos=[cfg.A_CENTER, cfg.B_CENTER])
    unbiased_dataset = torch.utils.data.TensorDataset(unbiased_samples)
    unbiased_loader = torch.utils.data.DataLoader(unbiased_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    print(f"Generated {len(unbiased_dataset)} unbiased samples.")

    # --- Data Storage for Biased Data ---
    latest_biased_samples = None
    start_time = time.time()

    # --- Iterative Training and Sampling Loop ---
    for iteration in range(cfg.N_ITERATIONS):
        iteration_num = iteration + 1
        print(f"\n--- Iteration {iteration_num}/{cfg.N_ITERATIONS} ---")

        # 1. Update biases with samples from the *previous* iteration
        model.eval()
        print("Updating bias states for current iteration...")
        # For on-the-fly OPES, this just resets the state. For others, it might use samples.
        bias_manager.update_all_states(latest_biased_samples) 

        # 2. Generate new samples using the updated bias manager
        print("Running biased sampling...")
        new_samples = sampler.sample(cfg.N_SAMPLES_PER_ITER, initial_pos=[cfg.A_CENTER, cfg.B_CENTER], bias_manager=bias_manager)
        latest_biased_samples = new_samples.detach().clone()

        # 3. Plot sampling feedback
        plot_sampling_feedback(model, potential, new_samples, iteration_num, bias_manager, output_dir=iteration_plot_dir)

        # 4. Calculate importance weights for training
        x_grad = new_samples.clone().requires_grad_(True)
        v_biases = bias_manager.calculate_bias_potential(x_grad)
        v_total_bias = v_biases['total'].detach()
        new_weights = torch.exp(cfg.BETA * v_total_bias).detach()
        
        # Clip and normalize weights for stable training
        max_weight = torch.quantile(new_weights, 0.99) if new_weights.numel() > 10 else 1000.0
        new_weights.clamp_(max=max_weight)
        new_weights /= new_weights.mean()
        
        print(f"Generated {new_samples.size(0)} biased samples. Mean weight: {new_weights.mean():.4f}")

        # --- Inner Training Loop ---
        biased_dataset = torch.utils.data.TensorDataset(new_samples, new_weights)
        biased_loader = torch.utils.data.DataLoader(biased_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
        
        for epoch in range(1, cfg.N_EPOCHS_PER_ITER + 1):
            model.train()
            # Cycle through unbiased loader to match batches
            unbiased_iter = iter(unbiased_loader)
            
            for batch_samples, batch_weights in biased_loader:
                try:
                    batch_unbiased, = next(unbiased_iter)
                except StopIteration:
                    unbiased_iter = iter(unbiased_loader)
                    batch_unbiased, = next(unbiased_iter)

                # --- Variational Loss on Biased Data ---
                batch_samples.requires_grad_(True)
                v_batch = potential.potential(batch_samples)
                var_losses = calculate_variational_losses(model, batch_samples, v_batch, batch_weights)
                
                # --- Boundary Loss on Unbiased Data ---
                dist_A = torch.linalg.norm(batch_unbiased - torch.tensor(cfg.A_CENTER, device=cfg.DEVICE), axis=-1)
                dist_B = torch.linalg.norm(batch_unbiased - torch.tensor(cfg.B_CENTER, device=cfg.DEVICE), axis=-1)
                masks = {'A': dist_A < cfg.RADIUS, 'B': dist_B < cfg.RADIUS}
                boundary_loss = calculate_boundary_loss(model, batch_unbiased, masks)
                
                # --- Total Loss and Optimization Step ---
                total_loss = var_losses['total_variational'] + cfg.W_UNBIASED_BOUND * boundary_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            if epoch % cfg.LOG_LOSS_EVER_N_EPOCHS == 0:
                l_tot = total_loss.item()
                l_var = var_losses['total_variational'].item()
                l_bnd = boundary_loss.item()
                print(f"  Epoch {epoch}/{cfg.N_EPOCHS_PER_ITER} | Total Loss: {l_tot:.3e} "
                      f"[Variational: {l_var:.3e}, Boundary: {l_bnd:.3e}]")

        # 6. Plot training feedback
        plot_iteration_feedback(model, potential, iteration_num, output_dir=iteration_plot_dir)

    end_time = time.time()
    print(f"\nTraining complete in {end_time - start_time:.2f} seconds.")

    # --- Final Evaluation and Visualization ---
    plot_results(model, potential, latest_biased_samples, bias_manager, output_dir=experiment_dir)

if __name__ == '__main__':
    main()
