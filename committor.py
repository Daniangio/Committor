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

# Import project modules
import config as cfg
from potential import MullerBrown

# --- Simulation Parameters ---
# Number of independent trajectories to start from each grid point.
N_TRAJECTORIES_PER_POINT = 5000
# Maximum number of simulation steps for any trajectory before it's timed out.
MAX_SIMULATION_STEPS = 200000
# Langevin dynamics timestep. Should be small enough for stability.
DT = 1e-6
# How often (in steps) to check if trajectories have reached a basin.
CHECK_STRIDE = 100
# Directory to save the results.
OUTPUT_DIR = "ground_truth"

def calculate_committor_for_grid(potential, grid_points):
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

    # --- Initialize Simulation State ---
    # Repeat each grid point N_TRAJECTORIES_PER_POINT times to create the initial positions for all walkers.
    # Shape: (n_points * N_TRAJECTORIES_PER_POINT, 2)
    positions = grid_points.repeat_interleave(N_TRAJECTORIES_PER_POINT, dim=0)

    # Keep track of which trajectories are still active.
    # Shape: (n_points * N_TRAJECTORIES_PER_POINT,)
    active_mask = torch.ones(positions.shape[0], dtype=torch.bool, device=cfg.DEVICE)

    # Store the results: 0 for basin A, 1 for basin B.
    # Shape: (n_points * N_TRAJECTORIES_PER_POINT,)
    results = torch.full(positions.shape, -1, dtype=torch.float32, device=cfg.DEVICE)

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

    if torch.any(active_mask):
        print(f"\n  Warning: {active_mask.sum().item()} trajectories timed out after {MAX_SIMULATION_STEPS} steps.")
        # Set timed-out trajectories to NaN to exclude them from the average.
        results[active_mask] = torch.nan

    # Reshape results to (n_points, N_TRAJECTORIES_PER_POINT) and calculate the mean for each point.
    committor_values = torch.nanmean(results.view(n_points, N_TRAJECTORIES_PER_POINT), dim=1)
    return committor_values

def main():
    """Main script to compute and plot the ground truth committor."""
    start_time = time.time()
    print("Starting ground truth committor calculation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Setup ---
    potential = MullerBrown(device=cfg.DEVICE)

    # Create the grid of starting points.
    xs = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.GRID_NX)
    ys = np.linspace(cfg.Y_MIN, cfg.Y_MAX, cfg.GRID_NY)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts_np = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    grid_pts_t = torch.tensor(grid_pts_np, dtype=torch.float32, device=cfg.DEVICE)

    # --- Calculation ---
    # Process grid points in batches to manage memory, though for this grid size it's likely fine.
    committor_grid_flat = calculate_committor_for_grid(potential, grid_pts_t)
    committor_grid = committor_grid_flat.cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)

    # --- Save Results ---
    data_path = os.path.join(OUTPUT_DIR, "committor_ground_truth.npz")
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
    
    plot_path = os.path.join(OUTPUT_DIR, "committor_ground_truth.png")
    plt.savefig(plot_path)
    print(f"Saved committor plot to {os.path.abspath(plot_path)}")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()