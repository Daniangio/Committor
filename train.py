# train.py
# This script handles the training of the committor model on a given dataset.
# It can be called from the adaptive protocol or run standalone on an npz file.

import torch
import os
import numpy as np
import argparse

# Import project modules
import config as cfg
from model import SmallNet
from losses import calculate_all_losses
from plotting import plot_training_data_feedback

def train_on_dataset(model, optimizer, data_loader, n_epochs):
    """
    Runs the training loop for a specified number of epochs.

    Args:
        model (nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (DataLoader): DataLoader providing batches of (samples, weights, boundary_type).
        n_epochs (int): Number of epochs to train for.
    """
    for epoch in range(1, n_epochs + 1):
        model.train()

        # Initialize accumulators for all loss components for the epoch
        epoch_losses = {
            'total': 0.0, 'boundary': 0.0, 'eikonal': 0.0,
            'committor': 0.0, 'link': 0.0, 'nonneg': 0.0
        }
        num_batches = 0

        for batch_samples, batch_weights, batch_boundary_type, v_batch in data_loader:
            batch_samples.requires_grad_(True)

            # Calculate all loss components in one go
            all_losses = calculate_all_losses(model, batch_samples, v_batch, batch_weights, batch_boundary_type)
            total_loss = all_losses['total']

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate all loss components
            for key in epoch_losses:
                epoch_losses[key] += all_losses[key].item()
            num_batches += 1

        if epoch % cfg.LOG_LOSS_EVER_N_EPOCHS == 0:
            if num_batches > 0:
                avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
                log_str = (f"  Epoch {epoch}/{n_epochs} | "
                           f"Total: {avg_losses['total']:.3e} | "
                           f"Bound: {avg_losses['boundary']:.2e} | "
                           f"Eik: {avg_losses['eikonal']:.2e} | "
                           f"Comm: {avg_losses['committor']:.2e} | "
                           f"Link: {avg_losses['link']:.2e} | "
                           f"NonNeg: {avg_losses['nonneg']:.2e}")
                print(log_str)


def main():
    """Main entry point to train the committor model on a dataset."""
    parser = argparse.ArgumentParser(description="Train the committor model on a specified dataset.")
    parser.add_argument('-d', '--dataset', type=str, nargs='+', required=True, help="Path(s) to .npz file(s) with 'samples', 'weights', and 'boundary_type' fields.")
    parser.add_argument('-im', '--input-model-path', type=str, default=None, help="Path to a pre-trained model state_dict to start from.")
    parser.add_argument('-om', '--output-model-path', type=str, required=True, help="Path to save the trained model state_dict.")
    parser.add_argument('-op', '--output-plot-path', type=str, default="training_feedback.png", help="Path to save the output feedback plot.")
    parser.add_argument('--iteration', type=int, default=None, help="Current iteration number for logging and plotting (optional).")
    parser.add_argument('--n-samples', type=int, default=None, help="Subsample the dataset to this many training samples.")
    parser.add_argument('--device', type=str, default=str(cfg.DEVICE), help=f"Computation device ('cpu', 'cuda'). Default: {cfg.DEVICE}")
    parser.add_argument('--ranges', type=float, nargs='+', help="Domain ranges for each dimension, e.g., --ranges -2.5 1.5 -1.0 3.0 for x_min x_max y_min y_max.")
    parser.add_argument('--periodic', type=str, action='append', help="Specify a periodic dimension and its period, e.g., --periodic=0:6.28 means on dimension 0 the periodicity is 6.28 (2pi). Can be specified multiple times.")
    args = parser.parse_args()

    # --- Update Config from Args ---
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: CUDA requested via '--device {args.device}' but not available. Falling back to 'cpu'.")
        cfg.DEVICE = torch.device('cpu')
    else:
        cfg.DEVICE = torch.device(args.device)
    print(f"Running on device: {cfg.DEVICE}")
    print(f"Loss weights: Eik={cfg.W_EIK}, Comm={cfg.W_COMM}, Link={cfg.W_LINK}, NonNeg={cfg.W_NONNEG}, Bound={cfg.W_BOUND}")

    # --- Load Data ---
    all_samples, all_weights, all_boundary_types, all_potentials = [], [], [], []
    print(f"Loading data from {len(args.dataset)} file(s)...")
    for file_path in args.dataset:
        data = np.load(file_path)
        samples = data['samples']
        num_samples = len(samples)

        # If 'boundary_type' is not in the file, fill with NaNs.
        boundary_type = data.get('boundary_type', np.full(num_samples, np.nan, dtype=np.float32))
        # If 'potential' is not in the file, fill with zeros (Eikonal loss will be skipped if weight is 0 anyway)
        potential = data.get('potential', np.zeros(num_samples, dtype=np.float32))

        all_samples.append(samples)
        all_weights.append(data.get('weights', np.ones(num_samples)))
        all_boundary_types.append(boundary_type)
        all_potentials.append(potential)

    # Concatenate data from all files
    samples_np = np.concatenate(all_samples, axis=0)
    weights_np = np.concatenate(all_weights, axis=0)
    boundary_type_np = np.concatenate(all_boundary_types, axis=0)
    potential_np = np.concatenate(all_potentials, axis=0)

    # --- Subsampling ---
    if args.n_samples and args.n_samples < len(samples_np):
        print(f"Subsampling dataset from {len(samples_np)} to {args.n_samples} points.")
        indices = np.random.choice(len(samples_np), args.n_samples, replace=False)
        samples_np = samples_np[indices]
        weights_np = weights_np[indices]
        boundary_type_np = boundary_type_np[indices]
        potential_np = potential_np[indices]

    # Convert to tensors
    samples_t = torch.tensor(samples_np, dtype=torch.float32, device=cfg.DEVICE)
    weights_t = torch.tensor(weights_np, dtype=torch.float32, device=cfg.DEVICE)
    boundary_type_t = torch.tensor(boundary_type_np, dtype=torch.float32, device=cfg.DEVICE)
    potential_t = torch.tensor(potential_np, dtype=torch.float32, device=cfg.DEVICE)
    print(f"Loaded a total of {len(samples_t)} samples.")
    print(f"  ({torch.sum(torch.isnan(boundary_type_t))} bulk, {torch.sum(~torch.isnan(boundary_type_t))} boundary)")

    # --- Create a single, unified DataLoader ---
    unified_dataset = torch.utils.data.TensorDataset(samples_t, weights_t, boundary_type_t, potential_t)
    data_loader = torch.utils.data.DataLoader(unified_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)


    # --- Setup Domain and Periodicity ---
    num_dims = samples_np.shape[1]
    if args.ranges:
        if len(args.ranges) != num_dims * 2:
            raise ValueError(f"Expected {num_dims*2} values for --ranges (min/max for each of {num_dims} dims), but got {len(args.ranges)}.")
        domain_min = [args.ranges[i] for i in range(0, num_dims*2, 2)]
        domain_max = [args.ranges[i] for i in range(1, num_dims*2, 2)]
    else:
        # If no ranges are given, compute from data with a small margin
        print("Warning: --ranges not provided. Computing from data min/max.")
        domain_min = samples_np.min(axis=0) - 0.1
        domain_max = samples_np.max(axis=0) + 0.1

    periodicity_info = [{'periodic': False, 'period': None} for _ in range(num_dims)]
    if args.periodic:
        for p_arg in args.periodic:
            dim_str, period_str = p_arg.split(':')
            dim_idx, period_val = int(dim_str), float(period_str)
            if 0 <= dim_idx < num_dims:
                periodicity_info[dim_idx]['periodic'] = True
                periodicity_info[dim_idx]['period'] = period_val

    # --- Setup Model and Optimizer ---
    model = SmallNet(domain_min, domain_max, periodicity_info).to(cfg.DEVICE)
    if args.input_model_path:
        print(f"Loading initial model weights from {args.input_model_path}")
        model.load_state_dict(torch.load(args.input_model_path, map_location=cfg.DEVICE))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # --- Run Training ---
    train_on_dataset(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        n_epochs=cfg.N_EPOCHS_PER_ITER
    )

    # --- Save Final Model ---
    output_model_dir = os.path.dirname(args.output_model_path)
    if output_model_dir:
        os.makedirs(output_model_dir, exist_ok=True)
    print(f"Saving trained model to {args.output_model_path}")
    torch.save(model.state_dict(), args.output_model_path)

    # --- Generate Training Data Feedback Plot ---
    # This plot shows the model's output on the data it was just trained on.
    plot_training_data_feedback(
        model, samples_np, boundary_type_np,
        output_path=args.output_plot_path,
        iteration=args.iteration
    )

if __name__ == '__main__':
    main()