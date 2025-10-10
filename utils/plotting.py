# utils/plotting.py
# Functions for visualizing the results.

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from torch.autograd import grad
import os

import config as cfg

def plot_iteration_feedback(model, potential, new_samples, iteration):
    """
    Generates a comprehensive 2x2 plot showing the results of a single
    adaptive sampling iteration. This includes the sample distribution, the
    biasing potential, and the learned committor and reaction coordinate.
    """
    print(f"  Generating feedback plot for iteration {iteration}...")
    # Ensure the directory for plots exists
    output_dir = "iteration_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()

    # Create a grid for plotting
    xs = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.GRID_NX)
    ys = np.linspace(cfg.Y_MIN, cfg.Y_MAX, cfg.GRID_NY)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    grid_t = torch.tensor(grid_pts, dtype=torch.float32, requires_grad=True).to(cfg.DEVICE)

    # --- Create Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=100)
    fig.suptitle(f'Adaptive Sampling Feedback: Iteration {iteration}', fontsize=18)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # --- Evaluate model and calculate V_bias on the grid ---
    # We need gradients for V_K, so calculations happen within torch.no_grad()
    # but with torch.enable_grad() for the specific gradient computations.
    g_grid, z_grid, q_grid, _ = model(grid_t)
    
    if iteration == 1: # First iteration is unbiased
        V_bias_grid_np = np.zeros((cfg.GRID_NY, cfg.GRID_NX))
    else:
        dq_grid = grad(q_grid, grid_t, grad_outputs=torch.ones_like(q_grid))[0]
        
        dq_norm2 = torch.sum(dq_grid**2, dim=1)
        v_bias_grid = - (cfg.LAMBDA_BIAS / cfg.BETA) * torch.log(dq_norm2 + 1e-8)
        V_bias_grid_np = v_bias_grid.cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)

    # Move all grid data to CPU and reshape for plotting
    g_grid_np = g_grid.detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    q_grid_np = q_grid.detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    V_grid = potential.potential(grid_t.detach()).detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    samples_np = new_samples.detach().cpu().numpy()
    
    # Plot common elements (potential contours, boundary circles)
    for ax in axes.ravel():
        ax.contour(XX, YY, V_grid, levels=np.logspace(0, 3, 15), cmap='viridis_r', norm=Normalize(0, 200), alpha=0.5)
        circle_A = plt.Circle(cfg.A_CENTER, cfg.RADIUS, color='red', fill=False, lw=2)
        circle_B = plt.Circle(cfg.B_CENTER, cfg.RADIUS, color='blue', fill=False, lw=2)
        ax.add_patch(circle_A)
        ax.add_patch(circle_B)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # --- Plot 1: Newly Sampled Distribution ---
    ax1.hist2d(samples_np[:, 0], samples_np[:, 1], bins=60,
               range=[[cfg.X_MIN, cfg.X_MAX], [cfg.Y_MIN, cfg.Y_MAX]],
               cmap='magma', density=True)
    ax1.set_title(f'Sampled Distribution (N={len(samples_np)})')

    # --- Plot 2: Biasing Potential (Kolmogorov Bias) ---
    v_min, v_max = np.percentile(V_bias_grid_np[np.isfinite(V_bias_grid_np)], [5, 99])
    im2 = ax2.imshow(V_bias_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='plasma', aspect='auto', vmin=v_min, vmax=v_max)
    fig.colorbar(im2, ax=ax2, label=r'$V_K = -\frac{\lambda}{\beta} \log(|\nabla q|^2)$')
    ax2.set_title('Biasing Potential (from previous model)')

    # --- Plot 3: Learned Committor q(x) ---
    im3 = ax3.imshow(q_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    fig.colorbar(im3, ax=ax3, label='Committor q(x)')
    ax3.contour(XX, YY, q_grid_np, levels=[0.1, 0.5, 0.9], colors='black', linewidths=1.5)
    ax3.set_title('Learned Committor q(x)')

    # --- Plot 4: Learned Reaction Coordinate g(x) ---
    im4 = ax4.imshow(g_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='viridis', aspect='auto')
    fig.colorbar(im4, ax=ax4, label='Reaction Coordinate g(x)')
    ax4.set_title('Learned Reaction Coordinate g(x)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f"iteration_feedback_{iteration:02d}.png"))
    plt.close(fig) # Close figure to free memory


def plot_results(model, potential, final_samples):
    """
    Generates and saves a comprehensive plot of the final results.
    """
    print("Generating final plots...")
    model.eval()

    # Create a grid for plotting
    xs = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.GRID_NX)
    ys = np.linspace(cfg.Y_MIN, cfg.Y_MAX, cfg.GRID_NY)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    grid_t = torch.tensor(grid_pts, dtype=torch.float32, requires_grad=True).to(cfg.DEVICE)

    # Evaluate potential and model on the grid
    V_grid = potential.potential(grid_t).detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    g_grid, z_grid, q_grid, _ = model(grid_t)

    # Compute gradient of g for vector field
    dg_grid = grad(g_grid, grid_t, grad_outputs=torch.ones_like(g_grid))[0]
    dg_grid_np = dg_grid.detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX, 2)
    vg_x, vg_y = -dg_grid_np[..., 0], -dg_grid_np[..., 1]

    # Move results to CPU and reshape for plotting
    g_grid_np = g_grid.detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    q_grid_np = q_grid.detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    samples_np = final_samples.detach().cpu().numpy()

    # --- Create Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=120)
    fig.suptitle('Final Learned Committor and Reaction Coordinate', fontsize=16)

    # Plot common elements: potential contours and boundary circles
    for ax in axes:
        ax.contour(XX, YY, V_grid, levels=np.logspace(0, 3, 15), cmap='viridis_r', norm=Normalize(0, 200))
        circle_A = plt.Circle(cfg.A_CENTER, cfg.RADIUS, color='red', fill=False, lw=2, label='Basin A')
        circle_B = plt.Circle(cfg.B_CENTER, cfg.RADIUS, color='blue', fill=False, lw=2, label='Basin B')
        ax.add_patch(circle_A)
        ax.add_patch(circle_B)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # --- Plot 1: Learned Committor q(x) ---
    ax1 = axes[0]
    im1 = ax1.imshow(q_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    fig.colorbar(im1, ax=ax1, label='Committor q(x)')
    ax1.contour(XX, YY, q_grid_np, levels=[0.1, 0.5, 0.9], colors='black', linewidths=2)
    ax1.set_title('Committor q(x) and Isocommittor Surfaces')

    # --- Plot 2: Learned Reaction Coordinate g(x) ---
    ax2 = axes[1]
    im2 = ax2.imshow(g_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='plasma', aspect='auto')
    fig.colorbar(im2, ax=ax2, label='Reaction Coordinate g(x)')
    ax2.streamplot(XX, YY, vg_x, vg_y, color='white', linewidth=0.7, density=1.2)
    ax2.set_title('Reaction Coordinate g(x) and Field -∇g')
    
    # --- Plot 3: Final Sample Distribution ---
    ax3 = axes[2]
    ax3.hist2d(samples_np[:, 0], samples_np[:, 1], bins=60,
               range=[[cfg.X_MIN, cfg.X_MAX], [cfg.Y_MIN, cfg.Y_MAX]],
               cmap='magma', density=True)
    ax3.set_title(f'Final Sampling Density (N={len(samples_np)})')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("final_committor_results.png")
    print("Final plots saved to final_committor_results.png")
    plt.show()
