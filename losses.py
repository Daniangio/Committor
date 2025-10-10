# losses.py
# Defines the components of the total loss function.

import torch
from torch.autograd import grad
import config as cfg

def get_gradients(model_output, x):
    """Helper to compute gradients of a scalar model output w.r.t. input x."""
    return grad(model_output, x, grad_outputs=torch.ones_like(model_output), create_graph=True)[0]

def calculate_losses(model, x_t, v_t, masks, weights):
    """
    Calculates all components of the loss function.

    Args:
        model (nn.Module): The neural network model.
        x_t (torch.Tensor): Training points.
        v_t (torch.Tensor): Potential values at training points.
        masks (dict): Dictionary containing boolean masks for regions 'A' and 'B'.
        weights (torch.Tensor): Importance weights for each sample.

    Returns:
        dict: A dictionary containing all calculated loss components.
    """
    g_pred, z_pred, q_pred, alpha = model(x_t)

    # Pre-calculate factors used in multiple losses
    exp_mbetaV = torch.exp(-cfg.BETA * v_t)

    # --- Eikonal Loss ---
    # L_eik = E_rho[ w * (exp(-beta*V) * |grad(g)| - 1)^2 ]
    dg = get_gradients(g_pred, x_t)
    dg_norm = torch.linalg.norm(dg, dim=1) # $|\nabla g|$
    
    # --- Committor Loss (Weighted Dirichlet Energy) ---
    # L_comm = E_rho[ w * exp(-beta*V) * |grad(q)|^2 ]
    dq = get_gradients(q_pred, x_t)
    dq_norm2 = torch.sum(dq**2, dim=1) # $|\nabla q|^2$
    
    # Eikonal Loss (Your original, non-standard formulation)
    # L_eik = $\mathbb{E}_{\rho} [ (e^{-\beta V} |\nabla g| - 1)^2 ]$
    res_eik = exp_mbetaV * dg_norm - 1.0
    L_eik = (res_eik**2).mean()

    # Committor Loss (Dirichlet Energy weighted by Boltzmann factor)
    # L_comm = $\mathbb{E}_{\rho} [ e^{-\beta V} |\nabla q|^2 ]$
    L_comm = (weights * dq_norm2).mean()

    # --- Boundary Loss ---
    # q should be 0 in A and 1 in B. g should be 0 in A.
    idx_A = masks['A'].nonzero(as_tuple=True)[0]
    idx_B = masks['B'].nonzero(as_tuple=True)[0]
    L_bound = torch.tensor(0.0, device=cfg.DEVICE)
    if len(idx_A) > 0:
        qA = q_pred[idx_A]
        gA = g_pred[idx_A]
        wA = weights[idx_A] # Use importance weights for the boundary points as well
        
        loss_qA = (wA * qA**2).mean()
        loss_gA = (wA * gA**2).mean()
        L_bound = loss_qA + loss_gA
        
        if len(idx_B) > 0:
            qB = q_pred[idx_B]
            wB = weights[idx_B] # Use importance weights for the boundary points as well
        
            loss_qB = (wB * (qB - 1.0)**2).mean()
            L_bound += loss_qB

    # --- Link Loss ---
    # Forces q to approximate a sigmoid of -alpha*g
    res_link = q_pred - torch.nn.functional.sigmoid(-alpha * g_pred)
    L_link = (res_link**2).mean()
    
    # --- Non-negativity Loss for g ---
    # Penalizes g < 0, ensuring g=0 is the minimum
    L_nonneg = (torch.relu(-g_pred)).mean()

    losses = {
        'eikonal': L_eik,
        'committor': L_comm,
        'boundary': L_bound,
        'link': L_link,
        'nonneg': L_nonneg,
    }
    
    # --- Total Loss ---
    total_loss = (cfg.W_EIK * L_eik +
                  cfg.W_COMM * L_comm +
                  cfg.W_BOUND * L_bound +
                  cfg.W_LINK * L_link +
                  cfg.W_NONNEG * L_nonneg)
    
    losses['total'] = total_loss
    return losses
