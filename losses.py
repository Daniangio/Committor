# losses.py
# Defines the components of the total loss function.

import torch
from torch.autograd import grad
import config as cfg

def _get_gradients(model_output, x, create_graph=True):
    """Helper to compute gradients of a scalar model output w.r.t. input x."""
    if not x.requires_grad:
        return torch.zeros_like(x)
    gradients = grad(model_output, x, grad_outputs=torch.ones_like(model_output), create_graph=create_graph, allow_unused=True)[0]
    return gradients if gradients is not None else torch.zeros_like(x)

def calculate_eikonal_loss(gA_pred, gB_pred, x, v, weights):
    """Calculates the Eikonal loss for gA and gB."""
    exp_mbetaV = torch.exp(-cfg.BETA * v)

    # Eikonal for gA(x)
    dgA = _get_gradients(gA_pred, x)
    dgA_norm = torch.linalg.norm(dgA, dim=1)
    res_eik_gA = (exp_mbetaV * dgA_norm) - 1.0
    L_eik_gA = torch.sum(weights * res_eik_gA**2) / torch.sum(weights)

    # Eikonal for gB(x)
    dgB = _get_gradients(gB_pred, x)
    dgB_norm = torch.linalg.norm(dgB, dim=1)
    res_eik_gB = (exp_mbetaV * dgB_norm) - 1.0
    L_eik_gB = torch.sum(weights * res_eik_gB**2) / torch.sum(weights)

    return L_eik_gA + L_eik_gB

def calculate_committor_loss(q_pred, x, weights):
    """Calculates the Committor loss (Dirichlet energy)."""
    dq = _get_gradients(q_pred, x)
    dq_norm2 = torch.sum(dq**2, dim=1)
    return torch.sum(weights * dq_norm2) / torch.sum(weights)

def calculate_link_loss(gA_pred, q_pred, x, weights):
    """Calculates the Link loss to enforce self-consistency."""
    dgA = _get_gradients(gA_pred, x)
    dq = _get_gradients(q_pred, x)

    dg_norm_vec = dgA / (torch.linalg.norm(dgA, dim=1, keepdim=True) + 1e-8)
    dq_norm_vec = dq / (torch.linalg.norm(dq, dim=1, keepdim=True) + 1e-8)

    res_link = torch.sum((dg_norm_vec - dq_norm_vec)**2, dim=1)
    return torch.sum(weights * res_link) / torch.sum(weights)

def calculate_nonneg_loss(gA_pred, gB_pred, weights):
    """Calculates the non-negativity loss for gA and gB."""
    L_nonneg_gA = torch.sum(weights * torch.relu(-gA_pred)**2) / torch.sum(weights)
    L_nonneg_gB = torch.sum(weights * torch.relu(-gB_pred)**2) / torch.sum(weights)
    return L_nonneg_gA + L_nonneg_gB

def calculate_boundary_loss(gA_pred, gB_pred, q_pred):
    """Calculates the boundary loss for q, gA, and gB."""
    # This loss is unweighted as it applies only to boundary points
    # which are sampled from an unbiased distribution.
    loss_qA = (q_pred**2).mean() # q(A) = 0
    loss_gA = (gA_pred**2).mean() # gA(A) = 0
    loss_qB = ((q_pred - 1.0)**2).mean() # q(B) = 1
    loss_gB = (gB_pred**2).mean() # gB(B) = 0
    return loss_qA + loss_qB + loss_gA + loss_gB

def calculate_all_losses(model, x, v, weights, boundary_type):
    """
    Orchestrates the calculation of all loss components on a single data batch.

    Args:
        model (nn.Module): The neural network model.
        x (torch.Tensor): Training points.
        v (torch.Tensor): Potential values at training points.
        weights (torch.Tensor): Importance weights for each sample.
        boundary_type (torch.Tensor): Tensor indicating boundary type (0 for A, 1 for B, NaN for bulk).

    Returns:
        dict: A dictionary containing all loss components and the total loss.
    """
    losses = {
        'eikonal': torch.tensor(0.0, device=cfg.DEVICE),
        'committor': torch.tensor(0.0, device=cfg.DEVICE),
        'link': torch.tensor(0.0, device=cfg.DEVICE),
        'nonneg': torch.tensor(0.0, device=cfg.DEVICE),
        'boundary': torch.tensor(0.0, device=cfg.DEVICE),
    }

    # --- Partition data into bulk and boundary ---
    bulk_mask = torch.isnan(boundary_type)
    boundary_A_mask = (boundary_type == 0.0)
    boundary_B_mask = (boundary_type == 1.0)

    # --- Calculate Bulk Losses (weighted) ---
    if torch.any(bulk_mask):
        x_bulk = x[bulk_mask]
        v_bulk = v[bulk_mask]
        w_bulk = weights[bulk_mask]

        gA_bulk, gB_bulk, _, q_bulk, _ = model(x_bulk)

        if cfg.W_EIK > 0:
            losses['eikonal'] = calculate_eikonal_loss(gA_bulk, gB_bulk, x_bulk, v_bulk, w_bulk)
        if cfg.W_COMM > 0:
            losses['committor'] = calculate_committor_loss(q_bulk, x_bulk, w_bulk)
        if cfg.W_LINK > 0:
            losses['link'] = calculate_link_loss(gA_bulk, q_bulk, x_bulk, w_bulk)
        if cfg.W_NONNEG > 0:
            losses['nonneg'] = calculate_nonneg_loss(gA_bulk, gB_bulk, w_bulk)

    # --- Calculate Boundary Losses (unweighted) ---
    loss_qA, loss_gA, loss_qB, loss_gB = (torch.tensor(0.0, device=cfg.DEVICE) for _ in range(4))

    if torch.any(boundary_A_mask):
        x_A = x[boundary_A_mask]
        gA_A, _, _, q_A, _ = model(x_A)
        loss_qA = (q_A**2).mean()
        loss_gA = (gA_A**2).mean()

    if torch.any(boundary_B_mask):
        x_B = x[boundary_B_mask]
        _, gB_B, _, q_B, _ = model(x_B)
        loss_qB = ((q_B - 1.0)**2).mean()
        loss_gB = (gB_B**2).mean()

    if cfg.W_BOUND > 0:
        losses['boundary'] = loss_qA + loss_gA + loss_qB + loss_gB

    # --- Combine all losses with their respective weights ---
    total_loss = (cfg.W_EIK * losses['eikonal'] +
                  cfg.W_COMM * losses['committor'] +
                  cfg.W_LINK * losses['link'] +
                  cfg.W_NONNEG * losses['nonneg'] +
                  cfg.W_BOUND * losses['boundary'])

    losses['total'] = total_loss
    return losses
