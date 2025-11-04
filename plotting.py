# utils/plotting.py
# Functions for visualizing the results.

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from torch.autograd import grad
import os
import glob
import imageio

import config as cfg


def plot_sampling_feedback(model, potential, new_samples, iteration, bias_manager, output_dir):
    """
    Generates a plot showing the results of a sampling step, including the
    sample distribution and the biasing potentials that were used.

    Args:
        model (nn.Module): The neural network model.
        potential (MullerBrown): The potential energy surface object.
        new_samples (torch.Tensor): The samples generated in this iteration.
        iteration (int): The current iteration number.
        bias_manager (BiasManager): The bias manager containing OPES and Kolmogorov biases.
        output_dir (str): The directory to save the plot in.
    """
    print(f"  Generating sampling feedback plot for iteration {iteration}...")
    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    # Create a grid for plotting
    xs = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.GRID_NX)
    ys = np.linspace(cfg.Y_MIN, cfg.Y_MAX, cfg.GRID_NY)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    grid_t = torch.tensor(grid_pts, dtype=torch.float32, requires_grad=True).to(cfg.DEVICE)

    # --- Create Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=100)
    fig.suptitle(f'Sampling Feedback: Iteration {iteration} (Using Previous Model State)', fontsize=16)
    ax1, ax2, ax3 = axes.ravel()

    # --- Evaluate biases on the grid ---
    v_biases_grid = bias_manager.calculate_bias_potential(grid_t)
    V_opes_grid_np = v_biases_grid.get('opes_flooding', torch.zeros_like(grid_t[:,0])).cpu().detach().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    V_opes_grid_np -= V_opes_grid_np.min()
    V_kolmogorov_grid_np = v_biases_grid.get('kolmogorov', torch.zeros_like(grid_t[:,0])).cpu().detach().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    V_kolmogorov_grid_np -= V_kolmogorov_grid_np.min()

    samples_np = new_samples.detach().cpu().numpy()
    V_grid = potential.potential(grid_t.detach()).detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)

    # Plot common elements
    for i, ax in enumerate(axes.ravel()):
        # For the first plot, we will render the potential with imshow. For others, use contours.
        if i > 0:
            ax.contour(XX, YY, V_grid, levels=np.logspace(0, 3, 15), cmap='plasma', norm=Normalize(0, 200), alpha=0.5)
        circle_A = plt.Circle(cfg.A_CENTER, cfg.RADIUS, color='red', fill=False, lw=2)
        circle_B = plt.Circle(cfg.B_CENTER, cfg.RADIUS, color='blue', fill=False, lw=2)
        ax.add_patch(circle_A)
        ax.add_patch(circle_B)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # --- Plot 1: Newly Sampled Distribution ---
    im1 = ax1.imshow(V_grid, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='plasma', aspect='auto', norm=Normalize(0, 200))
    fig.colorbar(im1, ax=ax1, label='Potential V(x)')
    ax1.scatter(samples_np[:, 0], samples_np[:, 1], s=1, alpha=0.3, c='black')
    ax1.set_title(f'Sampled Distribution (N={len(samples_np)}) on V(x)')

    # --- Plot 2: OPES Biasing Potential ---
    v_min, v_max = np.percentile(V_opes_grid_np[np.isfinite(V_opes_grid_np)], [5, 99]) if np.any(np.isfinite(V_opes_grid_np)) else (0, 1)
    im2 = ax2.imshow(V_opes_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='plasma', aspect='auto', vmin=v_min, vmax=v_max)
    fig.colorbar(im2, ax=ax2, label=r'$V_{bias}^{OPES}(z(x))$')
    ax2.set_title('OPES Biasing Potential (Used for Sampling)')

    # --- Plot 3: Kolmogorov Biasing Potential ---
    v_min, v_max = np.percentile(V_kolmogorov_grid_np[np.isfinite(V_kolmogorov_grid_np)], [5, 99]) if np.any(np.isfinite(V_kolmogorov_grid_np)) else (0, 1)
    im3 = ax3.imshow(V_kolmogorov_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='cividis', aspect='auto', vmin=v_min, vmax=v_max)
    fig.colorbar(im3, ax=ax3, label=r'$V_{bias}^{K}(q(x))$')
    ax3.set_title('Kolmogorov Biasing Potential (Used for Sampling)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"sampling_feedback_{iteration:02d}.png"))
    plt.close(fig)


def plot_opes_convergence_step(cv_grid, v_bias_new, v_bias_old, max_diff, eq_step, iteration, output_dir):
    """
    Generates a plot to visualize a single step of the OPES bias convergence.

    Args:
        cv_grid (np.ndarray): The CV grid points for the x-axis.
        v_bias_new (np.ndarray): The new bias potential on the grid.
        v_bias_old (np.ndarray): The old bias potential on the grid.
        max_diff (float): The maximum difference between new and old potentials.
        eq_step (int): The current equilibration step.
        iteration (int): The main training iteration number.
        output_dir (str): Directory to save the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=90)

    # Plot the new and old bias potentials
    ax1.plot(cv_grid, v_bias_new, label=f'V_bias (step {eq_step})', color='black')
    ax1.plot(cv_grid, v_bias_old, label=f'V_bias (previous)', color='gray', linestyle='--')
    ax1.set_xlabel("Collective Variable (z_pred)")
    ax1.set_ylabel("Bias Potential V_bias", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Create a second y-axis for the difference
    ax2 = ax1.twinx()
    delta_v = np.abs(v_bias_new - v_bias_old)
    ax2.plot(cv_grid, delta_v, label='|ΔV_bias|', color='crimson', alpha=0.8)
    ax2.set_ylabel("|ΔV_bias|", color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    # Use a log scale for the difference to see small changes
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1e-4) # Set a reasonable lower limit for log scale

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    fig.suptitle(f'OPES Convergence | Iteration {iteration}, Step {eq_step} | Max Diff: {max_diff:.3e}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f"opes_conv_iter_{iteration:02d}_step_{eq_step:05d}.png"))
    plt.close(fig)

def create_gif_from_plots(plot_dir, iteration):
    """
    Creates a GIF from all OPES convergence plots for a given iteration.

    Args:
        plot_dir (str): The directory containing the PNG files.
        iteration (int): The iteration number, used for naming the output GIF.
    """
    print(f"  Creating OPES convergence GIF for iteration {iteration}...")
    # Find all PNG files for the given iteration, sorted by step number
    search_pattern = os.path.join(plot_dir, f"opes_conv_iter_{iteration:02d}_step_*.png")
    image_files = sorted(glob.glob(search_pattern))

    if not image_files:
        print(f"    Warning: No PNG files found in {plot_dir} to create a GIF.")
        return

    # Read images and create GIF
    images = [imageio.imread(filename) for filename in image_files]
    gif_path = os.path.join(plot_dir, f"opes_convergence_iter_{iteration:02d}.gif")
    
    imageio.mimsave(gif_path, images, duration=150, loop=0) # duration in ms
    print(f"  -> Saved convergence GIF to {gif_path}")

def plot_iteration_feedback(model, potential, iteration, output_dir):
    """
    Generates a plot showing the results of a training iteration, including
    the learned committor q(x) and reaction coordinate g(x).

    Args:
        model (nn.Module): The neural network model (after training).
        potential (MullerBrown): The potential energy surface object.
        iteration (int): The current iteration number.
        output_dir (str): The directory to save the plot in.
    """
    print(f"  Generating feedback plot for iteration {iteration}...")
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
    
    model.eval()

    # Create a grid for plotting
    xs = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.GRID_NX)
    ys = np.linspace(cfg.Y_MIN, cfg.Y_MAX, cfg.GRID_NY)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    grid_t = torch.tensor(grid_pts, dtype=torch.float32, requires_grad=True).to(cfg.DEVICE)

    # --- Create Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=100)
    fig.suptitle(f'Training Results: Iteration {iteration}', fontsize=16)
    ax1, ax2, ax3 = axes.ravel()

    # --- Evaluate model on the grid ---
    g_grid, z_grid, q_grid, _ = model(grid_t)

    # Move all grid data to CPU and reshape for plotting
    g_grid_np = g_grid.detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    z_grid_np = z_grid.detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    q_grid_np = q_grid.detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    V_grid = potential.potential(grid_t.detach()).detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    
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

    # --- Plot 1: Learned Committor q(x) ---
    im1 = ax1.imshow(q_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    fig.colorbar(im1, ax=ax1, label='Committor q(x)')
    ax1.contour(XX, YY, q_grid_np, levels=[0.1, 0.5, 0.9], colors='black', linewidths=1.5)
    ax1.set_title('Learned Committor q(x)')

    # --- Plot 2: Learned Reaction Coordinate g(x) ---
    im2 = ax2.imshow(g_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='viridis', aspect='auto')
    fig.colorbar(im2, ax=ax2, label='Reaction Coordinate g(x)')
    ax2.set_title('Learned Reaction Coordinate g(x)')

    # --- Plot 3: Learned z(x) (pre-sigmoid for committor) ---
    im3 = ax3.imshow(z_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='bwr', aspect='auto')
    fig.colorbar(im3, ax=ax3, label='Logit of Committor z(x)')
    ax3.set_title('Learned z(x) (pre-sigmoid)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f"iteration_feedback_{iteration:02d}.png"))
    plt.close(fig) # Close figure to free memory

def plot_results(model, potential, final_samples, bias_manager, output_dir):
    """
    Generates and saves a comprehensive plot of the final results.

    Args:
        model (nn.Module): The final trained neural network model.
        potential (MullerBrown): The potential energy surface object.
        final_samples (torch.Tensor): The samples from the last iteration.
        bias_manager (BiasManager): The final bias manager object.
        output_dir (str): The directory to save the plot in.
    """
    print("Generating final plots...")
    model.eval()
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
    output_path = os.path.join(output_dir, "final_results_summary.png")

    # Create a grid for plotting
    xs = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.GRID_NX)
    ys = np.linspace(cfg.Y_MIN, cfg.Y_MAX, cfg.GRID_NY)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    grid_t = torch.tensor(grid_pts, dtype=torch.float32, requires_grad=True).to(cfg.DEVICE)

    # --- Evaluate model, potential, and bias on the grid ---
    V_grid = potential.potential(grid_t.detach()).detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    
    g_grid, z_grid, q_grid, _ = model(grid_t)
    
    # Calculate final bias potentials
    v_biases_grid = bias_manager.calculate_bias_potential(grid_t)
    V_opes_grid_np = v_biases_grid.get('opes_flooding', torch.zeros_like(grid_t[:,0])).cpu().detach().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    V_opes_grid_np -= V_opes_grid_np.min()
    V_kolmogorov_grid_np = v_biases_grid.get('kolmogorov', torch.zeros_like(grid_t[:,0])).cpu().detach().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    V_kolmogorov_grid_np -= V_kolmogorov_grid_np.min()

    # Move results to CPU and reshape for plotting
    g_grid_np = g_grid.detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    q_grid_np = q_grid.detach().cpu().numpy().reshape(cfg.GRID_NY, cfg.GRID_NX)
    samples_np = final_samples.detach().cpu().numpy()
    
    # --- Create Figure (2x3 Layout) ---
    fig, axes = plt.subplots(2, 3, figsize=(22, 14), dpi=120)
    fig.suptitle('Final Results Summary', fontsize=18)

    # Plot common elements: potential contours and boundary circles
    for ax in axes.ravel():
        ax.contour(XX, YY, V_grid, levels=np.logspace(0, 3, 10), cmap='viridis_r', norm=Normalize(0, 150), alpha=0.6)
        circle_A = plt.Circle(cfg.A_CENTER, cfg.RADIUS, color='red', fill=False, lw=2, label='Basin A')
        circle_B = plt.Circle(cfg.B_CENTER, cfg.RADIUS, color='blue', fill=False, lw=2, label='Basin B')
        # Ensure patches are added only once if iterating over axes.ravel()
        if not ax.patches: # Check if patches already exist to avoid duplicates
            ax.add_patch(circle_A)
            ax.add_patch(circle_B)
        ax.add_patch(circle_A)
        ax.add_patch(circle_B)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # --- Row 1: Committor and associated bias ---
    im1 = axes[0, 0].imshow(q_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    fig.colorbar(im1, ax=axes[0, 0], label='Committor q(x)')
    axes[0, 0].contour(XX, YY, q_grid_np, levels=[0.1, 0.5, 0.9], colors='black', linewidths=2)
    axes[0, 0].set_title('Final Committor q(x)')

    v_min, v_max = np.percentile(V_kolmogorov_grid_np[np.isfinite(V_kolmogorov_grid_np)], [5, 99]) if np.any(np.isfinite(V_kolmogorov_grid_np)) else (0, 1)
    im2 = axes[0, 1].imshow(V_kolmogorov_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='cividis', aspect='auto', vmin=v_min, vmax=v_max)
    fig.colorbar(im2, ax=axes[0, 1], label=r'$V_{bias}^{K}(q(x))$')
    axes[0, 1].set_title('Final Kolmogorov Bias')

    # --- Row 2: RC and associated bias ---
    im3 = axes[1, 0].imshow(g_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='plasma', aspect='auto')
    fig.colorbar(im3, ax=axes[1, 0], label='Reaction Coordinate g(x)')
    axes[1, 0].set_title('Final Reaction Coordinate g(x)')

    v_min, v_max = np.percentile(V_opes_grid_np[np.isfinite(V_opes_grid_np)], [5, 99]) if np.any(np.isfinite(V_opes_grid_np)) else (0, 1)
    im4 = axes[1, 1].imshow(V_opes_grid_np, extent=[cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX],
                     origin='lower', cmap='plasma', aspect='auto', vmin=v_min, vmax=v_max)
    fig.colorbar(im4, ax=axes[1, 1], label=r'$V_{bias}^{OPES}(z(x))$')
    axes[1, 1].set_title('Final OPES Bias')

    # --- Final Sample Distribution ---
    axes[0, 2].hist2d(samples_np[:, 0], samples_np[:, 1], bins=60,
               range=[[cfg.X_MIN, cfg.X_MAX], [cfg.Y_MIN, cfg.Y_MAX]],
               cmap='magma', density=True)
    axes[0, 2].set_title(f'Final Sample Distribution (N={len(samples_np)})')

    # --- Hide the last subplot ---
    axes[1, 2].axis('off')

    # Adjust layout to prevent titles/labels overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    print(f"Final plots saved to {os.path.abspath(output_path)}")
    plt.show()
