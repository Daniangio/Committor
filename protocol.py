# train.py
# protocol.py
# Orchestrates the iterative training and sampling process.

import torch
import time
import os
import numpy as np
import datetime
import argparse
import subprocess

# Import project modules
import config as cfg
from potential import MullerBrown
from model import SmallNet
from sampler import LangevinSampler
from plotting import plot_sampling_feedback, create_gif_from_plots, plot_iteration_feedback
from bias import BiasManager, OPESBias, KolmogorovBias

def main():
    """Main adaptive sampling and training loop."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the adaptive training protocol for the committor.")
    parser.add_argument('--beta', type=float, default=cfg.BETA,
                        help=f"Inverse temperature (1/kT). Default: {cfg.BETA}")
    parser.add_argument('--device', type=str, default=str(cfg.DEVICE),
                        help=f"Computation device ('cpu', 'cuda'). Default: {cfg.DEVICE}")
    args = parser.parse_args()

    # --- Update Config from Args ---
    cfg.BETA = args.beta
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: CUDA requested via '--device {args.device}' but not available. Falling back to 'cpu'.")
        cfg.DEVICE = torch.device('cpu')
    else:
        cfg.DEVICE = torch.device(args.device)
    print(f"Running on device: {cfg.DEVICE} with BETA: {cfg.BETA}")

    # --- Experiment Setup ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"beta_{cfg.BETA:.3f}_{timestamp}"
    experiment_dir = os.path.join("experiments", run_name)
    data_output_dir = os.path.join(experiment_dir, "training_data")
    iteration_plot_dir = os.path.join(experiment_dir, "iteration_plots")
    os.makedirs(iteration_plot_dir, exist_ok=True)
    print(f"Saving results to: {os.path.abspath(experiment_dir)}")

    # --- Initialization ---
    potential = MullerBrown(device=cfg.DEVICE)
    model = SmallNet().to(cfg.DEVICE)

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
    all_dataset_paths = []
    model_path = os.path.join(experiment_dir, "model_initial.pt")
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
            n_samples_per_iter = cfg.N_SAMPLES_UNBIASED_INITIAL
            
        else: # Biased sampling for all subsequent iterations
            print("Running BIASED sampling (with on-the-fly OPES convergence)...")
            bias_manager.reset_all_states()
            sampling_bias_manager = bias_manager
            n_samples_per_iter = cfg.N_SAMPLES_PER_ITER
            # Load the model from the previous training step
            model_path = os.path.join(experiment_dir, f"model_iter_{iteration:02d}.pt")
            model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
            print(f"Loaded model from {model_path} for sampling.")
        
        new_samples = sampler.sample(
            n_samples_per_iter,
            initial_pos=[cfg.A_CENTER, cfg.B_CENTER], 
            bias_manager=sampling_bias_manager,
            iteration=iteration_num, experiment_dir=experiment_dir
        )

        if new_samples.shape[0] == 0:
            print("Warning: No samples collected in this iteration. Ending run.")
            break
        
        # --- 2. Plot Sampling, Calculate Weights, and Save Data ---
        plot_sampling_feedback(model, potential, new_samples, iteration_num, sampling_bias_manager, output_dir=iteration_plot_dir)

        # Create a GIF of the OPES convergence from this iteration's sampling run
        if iteration > 0:
            opes_plots_dir = os.path.join(experiment_dir, "opes_convergence", f"iter_{iteration_num:02d}")
            create_gif_from_plots(opes_plots_dir, iteration_num)

        # --- Calculate weights and boundary types for the new samples ---
        if iteration == 0:
            # For the first iteration, weights are uniform.
            weights = torch.ones(new_samples.shape[0], device=cfg.DEVICE)
        else:
            v_biases = bias_manager.calculate_bias_potential(new_samples.requires_grad_(True))
            v_total_bias = v_biases['total'].detach()
            weights = torch.exp(cfg.BETA * v_total_bias)
            max_weight = torch.quantile(weights, 0.99) if weights.numel() > 10 else 1000.0
            weights.clamp_(max=max_weight)
            weights /= weights.mean()
            print(f"Collected {new_samples.size(0)} production samples. Mean weight: {weights.mean():.4f}")
        new_samples = new_samples.detach()

        # Calculate potential and unbiased forces for the new samples
        potential_values = potential.potential(new_samples)
        unbiased_forces = -potential.gradient(new_samples)

        # Determine boundary type for each sample (A=0, B=1, bulk=nan)
        dist_A = torch.linalg.norm(new_samples - torch.tensor(cfg.A_CENTER, device=cfg.DEVICE), axis=-1)
        dist_B = torch.linalg.norm(new_samples - torch.tensor(cfg.B_CENTER, device=cfg.DEVICE), axis=-1)
        boundary_type = np.full(new_samples.shape[0], np.nan, dtype=np.float32)
        # Use .detach() to avoid carrying gradients into numpy conversion
        boundary_type[dist_A.cpu().detach().numpy() < cfg.RADIUS] = 0.0
        boundary_type[dist_B.cpu().detach().numpy() < cfg.RADIUS] = 1.0
        print(f"  Identified {np.sum(boundary_type==0.0)} points in basin A, {np.sum(boundary_type==1.0)} in basin B.")

        # Save the collected samples and weights to a .npz file
        os.makedirs(data_output_dir, exist_ok=True)
        dataset_path = os.path.join(data_output_dir, f"data_iter_{iteration_num:02d}.npz")
        
        np.savez(dataset_path, 
                 samples=new_samples.cpu().numpy(), 
                 weights=weights.cpu().numpy(),
                 boundary_type=boundary_type,
                 potential=potential_values.cpu().detach().numpy(),
                 forces=unbiased_forces.cpu().detach().numpy())
        print(f"Saved training data to {dataset_path}")
        all_dataset_paths.append(dataset_path)

        # --- 3. Training Step (as a subprocess) ---
        print(f"--- Starting Training for Iteration {iteration_num} ---")
        output_model_path = os.path.join(experiment_dir, f"model_iter_{iteration_num:02d}.pt")
        
        # Construct the command to call the new train.py script
        train_command = [
            "python", "train.py"
        ]
        output_plot_path = os.path.join(iteration_plot_dir, f"training_data_feedback_{iteration_num:02d}.png")
        train_command.extend(["--dataset"] + all_dataset_paths)
        train_command.extend([
            "--output-model-path", output_model_path,
            "--output-plot-path", output_plot_path,
            "--iteration", str(iteration_num),
            "--device", str(cfg.DEVICE),
            "--ranges", str(cfg.X_MIN), str(cfg.X_MAX), str(cfg.Y_MIN), str(cfg.Y_MAX)
            # No --periodic argument is passed, so it defaults to False for all dimensions
        ])
        # For the first iteration, we train from scratch. For others, we load the previous model.
        if iteration > 0:
            train_command.extend(["--input-model-path", model_path])

        # Execute the training script
        subprocess.run(train_command, check=True)

        # --- 4. Plot Training Feedback (with potential context) ---
        # Load the model that was just saved by the training script
        model.load_state_dict(torch.load(output_model_path, map_location=cfg.DEVICE))
        plot_iteration_feedback(model, potential, iteration_num, output_dir=iteration_plot_dir)

    end_time = time.time()
    print(f"\nTraining complete in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
