# train.py
# Main script to orchestrate the iterative training and sampling process.

import torch
from torch.autograd import grad
import numpy as np
import time

# Import project modules
import config as cfg
from potential import MullerBrown
from model import SmallNet
from sampler import LangevinSampler
from losses import calculate_losses
from utils.plotting import plot_results, plot_iteration_feedback

def main():
    """Main training loop."""
    # --- Initialization ---
    potential = MullerBrown(device=cfg.DEVICE)
    model = SmallNet().to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    sampler = LangevinSampler(
        potential=potential,
        beta=cfg.BETA,
        dt=cfg.LANGEVIN_DT,
        n_steps=cfg.LANGEVIN_N_STEPS,
        n_walkers=cfg.N_WALKERS,
        record_stride=cfg.LANGEVIN_RECORD_STRIDE,
        device=cfg.DEVICE
    )

    # --- Data Storage ---
    all_samples = torch.empty((0, 2), device=cfg.DEVICE)
    all_weights = torch.empty(0, device=cfg.DEVICE)

    start_time = time.time()

    # --- Iterative Training and Sampling Loop ---
    for iteration in range(cfg.N_ITERATIONS):
        iteration_num = iteration + 1
        print(f"\n--- Iteration {iteration_num}/{cfg.N_ITERATIONS} ---")

        # 1. Define the biasing potential gradient based on the current model
        if iteration == 0:
            bias_grad_fn = None
            print("Running initial unbiased sampling...")
        else:
            # The bias potential V_K uses the committor q directly.
            # V_K = - (lambda/beta) * log(|grad(q)|^2)
            # The biasing force is -grad(V_K), which requires second derivatives of q.
            model.eval()
            def bias_grad_fn(x_in):
                x = x_in.clone().requires_grad_(True)
                _, _, q_pred, _ = model(x)
                
                # First gradient of q
                dq = grad(q_pred, x, grad_outputs=torch.ones_like(q_pred), create_graph=True)[0]
                dq_norm2 = torch.sum(dq**2, dim=1)
                
                # Log of the squared norm
                log_dq_norm2 = torch.log(dq_norm2 + 1e-8) # Epsilon for stability
                
                # Gradient of the log-term, which is proportional to grad(V_K)
                grad_log_term = grad(log_dq_norm2, x, grad_outputs=torch.ones_like(log_dq_norm2), create_graph=False)[0]
                
                # grad(V_K) = - (lambda/beta) * grad(log(|grad(q)|^2))
                bias_gradient = - (cfg.LAMBDA_BIAS / cfg.BETA) * grad_log_term
                return bias_gradient

            print("Running adaptively biased sampling with direct Kolmogorov bias on q...")

        # 2. Generate new samples
        new_samples = sampler.sample(cfg.N_SAMPLES_PER_ITER, initial_pos=[cfg.A_CENTER, cfg.B_CENTER], bias_grad_fn=bias_grad_fn)

        # 3. *** PLOT ITERATION FEEDBACK ***
        plot_iteration_feedback(model, potential, new_samples, iteration_num)

        # 4. Calculate importance weights for the new samples
        if iteration == 0:
            new_weights = torch.ones(new_samples.size(0), device=cfg.DEVICE)
        else:
            new_samples.requires_grad_(True)
            
            _, _, q_values, _ = model(new_samples)
            dq = grad(q_values, new_samples, grad_outputs=torch.ones_like(q_values), create_graph=False)[0]    
            dq_norm2 = torch.sum(dq**2, dim=1)
            
            v_bias = - (cfg.LAMBDA_BIAS / cfg.BETA) * torch.log(dq_norm2 + 1e-8)
            new_weights = torch.exp(cfg.BETA * v_bias).detach()
            new_samples = new_samples.detach()
        
        # 5. Aggregate data
        all_samples = new_samples
        all_weights = new_weights
        
        # Clip weights to prevent the remaining numerical instabilities from dominating the loss
        max_weight = 100.0
        all_weights.clamp_(max=max_weight)
        all_weights /= all_weights.mean()
        
        print(f"Total samples: {all_samples.size(0)}, Mean weight: {all_weights.mean():.4f}")
        
        # --- Inner Training Loop ---
        dataset = torch.utils.data.TensorDataset(all_samples, all_weights)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
        
        for epoch in range(1, cfg.N_EPOCHS_PER_ITER + 1):
            model.train()
            for batch_samples, batch_weights in loader:
                batch_samples.requires_grad_(True)
                
                v_batch = potential.potential(batch_samples)
                
                dist_A = torch.linalg.norm(batch_samples - torch.tensor(cfg.A_CENTER, device=cfg.DEVICE), axis=-1)
                dist_B = torch.linalg.norm(batch_samples - torch.tensor(cfg.B_CENTER, device=cfg.DEVICE), axis=-1)
                masks = {'A': dist_A < cfg.RADIUS, 'B': dist_B < cfg.RADIUS}

                optimizer.zero_grad()
                losses = calculate_losses(model, batch_samples, v_batch, masks, batch_weights)
                
                losses['total'].backward()
                optimizer.step()

            if epoch % 200 == 0:
                l_tot = losses['total'].item()
                l_eik = losses['eikonal'].item()
                l_com = losses['committor'].item()
                l_bnd = losses['boundary'].item()
                l_lnk = losses['link'].item()
                l_non = losses['nonneg'].item()
                print(f"  Epoch {epoch}/{cfg.N_EPOCHS_PER_ITER} | Loss: {l_tot:.3e} "
                      f"[Eik: {l_eik:.2e}, Comm: {l_com:.2e}, Bound: {l_bnd:.2e}, Link: {l_lnk:.2e}, Nonneg: {l_non:.2e}]")

    end_time = time.time()
    print(f"\nTraining complete in {end_time - start_time:.2f} seconds.")

    # --- Final Evaluation and Visualization ---
    plot_results(model, potential, all_samples)

if __name__ == '__main__':
    main()
