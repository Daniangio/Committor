# losses.py
# Defines the components of the total loss function.
# Loss calculation is split into two functions.

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
    gA_pred, gB_pred, _, q_pred, alpha = model(x_t)

    # Squeeze outputs just in case model defines them as (batch, 1)
    gA_pred = gA_pred.squeeze(-1) if gA_pred.ndim > 1 else gA_pred
    gB_pred = gB_pred.squeeze(-1) if gB_pred.ndim > 1 else gB_pred
    q_pred = q_pred.squeeze(-1) if q_pred.ndim > 1 else q_pred

    # --- 1. Eikonal Losses (for gA and gB) ---
    # We want to solve |grad g| = exp(beta * V)
    # The residual is R(x) = exp(-beta * V) * |grad g| - 1
    exp_mbetaV = torch.exp(-cfg.BETA * v_t)

    # Eikonal for gA(x) (cost from A)
    dgA = get_gradients(gA_pred, x_t)
    dgA_norm = torch.linalg.norm(dgA, dim=1)
    res_eik_gA = (exp_mbetaV * dgA_norm) - 1.0
    # Average over the p_bias (flat) distribution, NOT p_B (weighted)
    L_eik_gA = torch.mean(res_eik_gA**2)

    # Eikonal for gB(x) (cost from B)
    dgB = get_gradients(gB_pred, x_t)
    dgB_norm = torch.linalg.norm(dgB, dim=1)
    res_eik_gB = (exp_mbetaV * dgB_norm) - 1.0
    L_eik_gB = torch.mean(res_eik_gB**2) # Average over p_bias (flat)

    L_eik = L_eik_gA + L_eik_gB

    # --- 2. Committor Loss (for q) ---
    # We minimize the Dirichlet energy K[q] = <|grad q|^2>_p_B
    # The 'weights' correctly re-weight from p_bias to p_B.
    dq = get_gradients(q_pred, x_t)
    dq_norm2 = torch.sum(dq**2, dim=1)
    
    # This is the observable O(x) = |grad q|^2
    # We average it over p_B using the importance weights.
    L_comm = torch.sum(weights * dq_norm2) / torch.sum(weights)

    # --- 3. Link Loss (for self-consistency) ---
    # Normalize with a safe epsilon
    dg_norm_vec = dgA / (torch.linalg.norm(dgA, dim=1, keepdim=True) + 1e-8)
    dq_norm_vec = dq / (torch.linalg.norm(dq, dim=1, keepdim=True) + 1e-8)

    # Calculate the squared L2 distance between the unit vectors
    res_link = torch.sum((dg_norm_vec - dq_norm_vec)**2, dim=1)

    # Average over the biased, flat distribution (p_bias)
    L_link = torch.mean(res_link)
    
    # # # # We enforce q_pred ≈ sigmoid(alpha * (gA - gB))
    # # # # We use (gB_pred - gA_pred) so it's negative at A (q=0) and positive at B (q=1)
    # # # q_derived = torch.sigmoid(alpha.squeeze() * (gB_pred - gA_pred))

    # # # # We use a simple MSE loss, averaged over the flat p_bias
    # # # res_link = (q_pred - q_derived)**2
    # # # L_link = torch.mean(res_link)

    # --- 4. Non-negativity Loss (for gA and gB) ---
    # These are costs, they cannot be negative.
    L_nonneg_gA = torch.mean(torch.relu(-gA_pred)**2) # Averaged over p_bias
    L_nonneg_gB = torch.mean(torch.relu(-gB_pred)**2) # Averaged over p_bias
    L_nonneg = L_nonneg_gA + L_nonneg_gB

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
    gA_pred, gB_pred, _, q_pred, _ = model(x_unbiased)
    
    idx_A = masks['A'].nonzero(as_tuple=True)[0]
    idx_B = masks['B'].nonzero(as_tuple=True)[0]
    
    loss_qA = torch.tensor(0.0, device=cfg.DEVICE)
    loss_qB = torch.tensor(0.0, device=cfg.DEVICE)
    loss_gA = torch.tensor(0.0, device=cfg.DEVICE)
    loss_gB = torch.tensor(0.0, device=cfg.DEVICE)

    if len(idx_A) > 0:
        qA = q_pred[idx_A]
        gA = gA_pred[idx_A]
        loss_qA = (qA**2).mean() # q(A) = 0
        loss_gA = (gA**2).mean() # gA(A) = 0
        
    if len(idx_B) > 0:
        qB = q_pred[idx_B]
        gB = gB_pred[idx_B]
        loss_qB = ((qB - 1.0)**2).mean() # q(B) = 1
        loss_gB = (gB**2).mean() # gB(B) = 0
            
    return loss_qA + loss_qB + loss_gA + loss_gB
