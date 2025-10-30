# losses.py
# Defines the components of the total loss function.
# REFACTORED: Loss calculation is now split into two functions.

import torch
from torch.autograd import grad
import config as cfg

def get_gradients(model_output, x):
    """Helper to compute gradients of a scalar model output w.r.t. input x."""
    return grad(model_output, x, grad_outputs=torch.ones_like(model_output), create_graph=True)[0]

def calculate_variational_losses(model, x_t, v_t, weights):
    """
    Calculates all VARIATIONAL components of the loss function on BIASED data.
    All losses here are correctly weighted by the importance weights.

    Args:
        model (nn.Module): The neural network model.
        x_t (torch.Tensor): Training points from biased sampling.
        v_t (torch.Tensor): Potential values at training points.
        weights (torch.Tensor): Importance weights for each sample.

    Returns:
        dict: A dictionary containing all calculated variational loss components.
    """
    g_pred, z_pred, q_pred, alpha = model(x_t)

    # Pre-calculate factors used in multiple losses
    exp_mbetaV = torch.exp(-cfg.BETA * v_t)

    # --- Eikonal Loss ---
    dg = get_gradients(g_pred, x_t)
    dg_norm = torch.linalg.norm(dg, dim=1)
    res_eik = exp_mbetaV * dg_norm - 1.0
    # REFACTORED: Eikonal loss is now correctly weighted.
    L_eik = torch.sum(weights * res_eik**2) / torch.sum(weights)

    # --- Committor Loss (Weighted Dirichlet Energy) ---
    dq = get_gradients(q_pred, x_t)
    dq_norm2 = torch.sum(dq**2, dim=1)
    L_comm = torch.sum(weights * exp_mbetaV * dq_norm2) / torch.sum(weights)

    # --- Link Loss ---
    res_link = q_pred - torch.sigmoid(-alpha * g_pred)
    # REFACTORED: Link loss is now correctly weighted.
    L_link = torch.sum(weights * res_link**2) / torch.sum(weights)
    
    # --- Non-negativity Loss for g ---
    # REFACTORED: Non-negativity loss is now correctly weighted.
    L_nonneg = torch.sum(weights * torch.relu(-g_pred)**2) / torch.sum(weights)

    losses = {
        'eikonal': L_eik,
        'committor': L_comm,
        'link': L_link,
        'nonneg': L_nonneg,
    }
    
    # --- Total Variational Loss ---
    total_loss = (cfg.W_EIK * L_eik +
                  cfg.W_COMM * L_comm +
                  cfg.W_LINK * L_link +
                  cfg.W_NONNEG * L_nonneg)
    
    losses['total_variational'] = total_loss
    return losses


def calculate_boundary_loss(model, x_unbiased, masks):
    """
    Calculates the boundary loss on a separate, UNBIASED, unweighted dataset.
    This loss penalizes:
    - q(x) != 0 for x in A
    - q(x) != 1 for x in B
    - g(x) != 0 for x in A (ADDED)

    Args:
        model (nn.Module): The neural network model.
        x_unbiased (torch.Tensor): Training points from unbiased basin sampling.
        masks (dict): Dictionary containing boolean masks for regions 'A' and 'B'.

    Returns:
        torch.Tensor: The calculated boundary loss.
    """
    # Get both g and q predictions from the model
    g_pred, _, q_pred, _ = model(x_unbiased)
    
    idx_A = masks['A'].nonzero(as_tuple=True)[0]
    idx_B = masks['B'].nonzero(as_tuple=True)[0]
    
    loss_qA = torch.tensor(0.0, device=cfg.DEVICE)
    loss_qB = torch.tensor(0.0, device=cfg.DEVICE)
    loss_gA = torch.tensor(0.0, device=cfg.DEVICE) # New loss term for g in A

    if len(idx_A) > 0:
        qA = q_pred[idx_A]
        gA = g_pred[idx_A] # Get g predictions for basin A
        loss_qA = (qA**2).mean() # Unweighted mean
        loss_gA = (gA**2).mean() # Penalize g if it's not zero in A
        
    if len(idx_B) > 0:
        qB = q_pred[idx_B]
        loss_qB = ((qB - 1.0)**2).mean() # Unweighted mean
            
    # Combine all boundary loss components
    return loss_qA + loss_qB + loss_gA
