# committor.py
# This script calculates the "ground truth" committor probability for a given
# potential energy surface (e.g., Müller-Brown) by running a large number of
# unbiased molecular dynamics simulations from each point on a grid.

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import time
import math
import argparse

# Import project modules
import config as cfg
from potential import MullerBrown

# --- Simulation Parameters ---
# Number of independent trajectories to start from each grid point.
N_TRAJECTORIES_PER_POINT = 5000
# Maximum number of simulation steps for any trajectory before it's timed out.
MAX_SIMULATION_STEPS = 100000
# Langevin dynamics timestep. Should be small enough for stability.
DT = 5e-6
# How often (in steps) to check if trajectories have reached a basin.
CHECK_STRIDE = 100
# How often (in steps) to plot the positions of the remaining trajectories.
PLOT_STRIDE = 10000
# Directory to save the results.
OUTPUT_DIR = "ground_truth"

def plot_active_walkers(active_positions, step, V_grid, XX, YY, trajectory_plot_dir):
    """
    Plots the current positions of active trajectories over the potential.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot potential contours
    ax.contourf(XX, YY, V_grid, levels=np.logspace(0, 3, 30), cmap='viridis_r', norm=Normalize(0, 200), alpha=0.8)
    
    # Plot active walker positions
    walkers_np = active_positions.cpu().numpy()
    ax.scatter(walkers_np[:, 0], walkers_np[:, 1], s=1, c='black', alpha=0.5, label=f'Active Walkers ({len(walkers_np)})')
    
    # Add basin circles
    circle_A = plt.Circle(cfg.A_CENTER, cfg.RADIUS, color='lime', fill=False, lw=2, label='Basin A')
    circle_B = plt.Circle(cfg.B_CENTER, cfg.RADIUS, color='yellow', fill=False, lw=2, label='Basin B')
    ax.add_patch(circle_A)
    ax.add_patch(circle_B)
    
    ax.set_title(f'Active Trajectories at Step {step}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(cfg.X_MIN, cfg.X_MAX)
    ax.set_ylim(cfg.Y_MIN, cfg.Y_MAX)
    ax.set_aspect('equal')
    ax.legend()
    
    plot_path = os.path.join(trajectory_plot_dir, f"walkers_step_{step:07d}.png")
    plt.savefig(plot_path, dpi=100)
    print(trajectory_plot_dir)
    plt.close(fig)

def calculate_committor_for_grid(potential, grid_points, trajectory_plot_dir):
    """
    Calculates the committor probability for a batch of starting points.

    Args:
        potential (MullerBrown): The potential energy surface.
        grid_points (torch.Tensor): A tensor of shape (N_points, 2) with starting coordinates.

    Returns:
        torch.Tensor: A tensor of shape (N_points,) with the committor value for each point.
    """
    n_points = grid_points.shape[0]
    print(f"  Calculating committor for {n_points} grid points with {N_TRAJECTORIES_PER_POINT} trajectories each...")

    # --- Setup for intermediate plotting ---
    os.makedirs(trajectory_plot_dir, exist_ok=True)
    plot_xs = np.linspace(cfg.X_MIN, cfg.X_MAX, 100)
    plot_ys = np.linspace(cfg.Y_MIN, cfg.Y_MAX, 100)
    plot_XX, plot_YY = np.meshgrid(plot_xs, plot_ys)
    plot_grid_t = torch.tensor(np.stack([plot_XX.ravel(), plot_YY.ravel()], axis=-1), dtype=torch.float32, device=cfg.DEVICE)
    V_grid = potential.potential(plot_grid_t).cpu().numpy().reshape(100, 100)

    # --- Initialize Simulation State ---
    # Repeat each grid point N_TRAJECTORIES_PER_POINT times to create the initial positions for all walkers.
    # Shape: (n_points * N_TRAJECTORIES_PER_POINT, 2)
    positions = grid_points.repeat_interleave(N_TRAJECTORIES_PER_POINT, dim=0)

    # Keep track of which trajectories are still active.
    # Shape: (n_points * N_TRAJECTORIES_PER_POINT,)
    active_mask = torch.ones(positions.shape[0], dtype=torch.bool, device=cfg.DEVICE)

    # Store the results: 0 for basin A, 1 for basin B.
    # Shape: (n_points * N_TRAJECTORIES_PER_POINT,)
    results = torch.full((positions.shape[0],), -1, dtype=torch.float32, device=cfg.DEVICE)

    # Pre-calculate constants for Langevin dynamics.
    noise_factor = math.sqrt(2 * DT / cfg.BETA)
    
    # Basin definitions
    a_center_t = torch.tensor(cfg.A_CENTER, device=cfg.DEVICE)
    b_center_t = torch.tensor(cfg.B_CENTER, device=cfg.DEVICE)

    # --- Main Simulation Loop ---
    for step in range(1, MAX_SIMULATION_STEPS + 1):
        if not torch.any(active_mask):
            print(f"    All trajectories finished by step {step}.")
            break

        # Get positions of only the active trajectories.
        active_positions = positions[active_mask]

        # Calculate forces and update positions using Langevin dynamics.
        force = -potential.gradient(active_positions)
        noise = torch.randn_like(active_positions)
        active_positions += force * DT + noise * noise_factor

        # Update the main positions tensor.
        positions[active_mask] = active_positions

        # --- Check for Basin Entry (only every CHECK_STRIDE steps for efficiency) ---
        if step % CHECK_STRIDE == 0:
            # --- Check for NaN values after position update ---
            # This can happen if forces are too large, causing numerical instability.
            nan_mask = torch.any(torch.isnan(active_positions), dim=1)
            if torch.any(nan_mask):
                num_nans = nan_mask.sum().item()
                # Get the original indices of the trajectories that became NaN
                active_indices = active_mask.nonzero(as_tuple=True)[0]
                nan_indices = active_indices[nan_mask]

                # Deactivate these trajectories
                active_mask[nan_indices] = False
                # Set their result to NaN so they are ignored in the final average
                results[nan_indices] = torch.nan
                print(f"  Warning: Removed {num_nans} trajectories at step {step} due to NaN positions.\n", end='')

                # If we removed any NaNs, we MUST update active_positions to reflect the change
                # before proceeding with basin checks. This is the fix.
                active_positions = positions[active_mask]

            # Check distances for all active trajectories.
            dist_A = torch.linalg.norm(active_positions - a_center_t, axis=-1)
            dist_B = torch.linalg.norm(active_positions - b_center_t, axis=-1)

            # Identify trajectories that have just entered a basin.
            hit_A = dist_A < cfg.RADIUS
            hit_B = dist_B < cfg.RADIUS

            # Get the original indices of the active trajectories.
            active_indices = active_mask.nonzero(as_tuple=True)[0]

            # Update results and deactivate trajectories that hit basin A.
            if torch.any(hit_A):
                finished_indices_A = active_indices[hit_A]
                results[finished_indices_A] = 0.0
                active_mask[finished_indices_A] = False

            # Update results and deactivate trajectories that hit basin B.
            if torch.any(hit_B):
                finished_indices_B = active_indices[hit_B]
                results[finished_indices_B] = 1.0
                active_mask[finished_indices_B] = False
            
            num_active = active_mask.sum().item()
            print(f"    Step {step}/{MAX_SIMULATION_STEPS}: {num_active} trajectories remaining.", end='\r')

        # --- Plot intermediate positions ---
        if step % PLOT_STRIDE == 0 and torch.any(active_mask):
            # We must re-select active_positions as the mask may have changed
            plot_active_walkers(positions[active_mask], step, V_grid, plot_XX, plot_YY, trajectory_plot_dir)


    if torch.any(active_mask):
        print(f"\n  Warning: {active_mask.sum().item()} trajectories timed out after {MAX_SIMULATION_STEPS} steps.")        
        # Set timed-out trajectories to NaN to exclude them from the average.
        results[active_mask] = torch.nan

    # Reshape results to (n_points, N_TRAJECTORIES_PER_POINT) and calculate the mean for each point.
    committor_values = torch.nanmean(results.view(n_points, N_TRAJECTORIES_PER_POINT), dim=1)
    return committor_values

def main():
    """Main script to compute and plot the ground truth committor."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Calculate ground truth committor probability.")
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

    start_time = time.time()
    print("Starting ground truth committor calculation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Clean up old snapshot plots if they exist
    experiment_dir = os.path.join(OUTPUT_DIR, f"beta_{cfg.BETA:.3f}")
    trajectory_plot_dir = os.path.join(experiment_dir, "trajectory_snapshots")
    if os.path.exists(trajectory_plot_dir) and OUTPUT_DIR == "ground_truth":
        for f in os.listdir(trajectory_plot_dir): os.remove(os.path.join(trajectory_plot_dir, f))

    # --- Setup ---
    potential = MullerBrown(device=cfg.DEVICE)

    # Create the grid of starting points.
    xs = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.GRID_NX)
    os.makedirs(trajectory_plot_dir, exist_ok=True)
    ys = np.linspace(cfg.Y_MIN, cfg.Y_MAX, cfg.GRID_NY)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts_np = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    grid_pts_t = torch.tensor(grid_pts_np, dtype=torch.float32, device=cfg.DEVICE)

    # --- Calculation ---
    # Process grid points in batches to manage memory, though for this grid size it's likely fine.
    committor_grid_flat = calculate_committor_for_grid(potential, grid_pts_t, trajectory_plot_dir)
    committor_grid = committor_grid_flat.cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)    

    # --- Save Results ---
    data_path = os.path.join(experiment_dir, "committor_ground_truth.npz")
    np.savez(data_path, committor=committor_grid, x_grid=XX, y_grid=YY)
    print(f"\nSaved committor data to {os.path.abspath(data_path)}")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    
    # Plot the potential energy contours in the background.
    V_grid = potential.potential(grid_pts_t).cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    ax.contour(XX, YY, V_grid, levels=np.logspace(0, 3, 15), cmap='viridis_r', norm=Normalize(0, 200), alpha=0.5)

    # Plot the calculated committor probability.
    im = ax.imshow(committor_grid, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                   origin='lower', cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label='Committor Probability q(x)')

    # Plot the committor isolines (isocommittor surfaces).
    ax.contour(XX, YY, committor_grid, levels=[0.1, 0.5, 0.9], colors='black', linewidths=2)

    # Add circles for basins A and B.
    circle_A = plt.Circle(cfg.A_CENTER, cfg.RADIUS, color='lime', fill=False, lw=2, label='Basin A')
    circle_B = plt.Circle(cfg.B_CENTER, cfg.RADIUS, color='yellow', fill=False, lw=2, label='Basin B')
    ax.add_patch(circle_A)
    ax.add_patch(circle_B)

    ax.set_title(f'Ground Truth Committor (N={N_TRAJECTORIES_PER_POINT} trajectories/point)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.legend()
    
    os.makedirs(experiment_dir, exist_ok=True)
    plot_path = os.path.join(experiment_dir, f"committor_ground_truth.beta.{cfg.BETA:.3f}.png")
    plt.savefig(plot_path)
    print(f"Saved committor plot to {os.path.abspath(plot_path)}")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()