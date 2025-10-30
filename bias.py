# bias.py
# A modular framework for combining multiple sources of biasing potential.
# REFACTORED: OPES bias is now static per iteration.

import math
import torch
from torch.autograd import grad
import config as cfg

# --- Base Class (for structure and type hinting) ---
class BiasSource:
    """Abstract base class for a source of biasing potential."""
    def __init__(self, device='cpu'):
        self.device = device
        self.name = "base_bias"
        self.is_built = False

    def calculate_potential(self, x, **cvs):
        """Calculates the bias potential."""
        raise NotImplementedError

    def update_state(self, *args, **kwargs):
        """Updates the internal state of the bias (e.g., for adaptive methods)."""
        pass

# --- OPES Bias Implementation (Static per Iteration) ---
class OPESBias(BiasSource):
    """
    REFACTORED: Implements an on-the-fly OPES bias potential.
    The bias is built up during a simulation by periodically depositing
    Gaussian kernels. The simulation has a warm-up phase until the bias
    converges, after which a production run with the fixed bias begins.
    """
    def __init__(self, cv_func, bias_factor, kernel_sigma, device):
        super().__init__(device=device)
        self.name = "opes"
        self.cv_func = cv_func
        if bias_factor < 1.0:
            raise ValueError("Bias factor gamma must be >= 1.0")
        self.prefactor = (1.0 - 1.0 / bias_factor) / cfg.BETA
        self.kernel_sigma = kernel_sigma
        self.kernel_centers = torch.empty((0,), device=self.device)
        self.is_converged = False
        
        # For convergence check
        self.conv_grid = torch.linspace(0, 1, cfg.OPES_CONV_GRID_PTS, device=self.device)
        self.v_bias_prev = torch.zeros_like(self.conv_grid)

    def _get_prob_estimate(self, s):
        """Calculates the kernel density estimate of the probability."""
        if self.kernel_centers.shape[0] == 0:
             return torch.ones_like(s) # Return uniform probability if no centers exist
        s_u = s.unsqueeze(1)
        centers_u = self.kernel_centers.unsqueeze(0)
        dist2 = (s_u - centers_u)**2
        kernels = torch.exp(-0.5 * dist2 / self.kernel_sigma**2)
        p_s = kernels.mean(dim=1) # Use mean instead of sum for stability
        # Normalize probability distribution
        return p_s / torch.trapezoid(p_s, s) if s.shape[0] > 1 else p_s

    def calculate_potential(self, x, z_pred=None, **kwargs):
        """Calculates the static OPES bias potential."""
        if not self.is_built or z_pred is None:
            return torch.zeros_like(z_pred)
        
        p_s = self._get_prob_estimate(z_pred)
        # Add epsilon to prevent log(0)
        v_opes_raw = self.prefactor * torch.log(p_s + 1e-12)
        # Return a potential that is zero at its maximum to prevent large energy shifts
        return v_opes_raw - v_opes_raw.max().detach() if v_opes_raw.numel() > 0 else v_opes_raw

    def add_kernels(self, new_samples):
        """Adds new kernel centers based on new samples."""
        new_cvs = self.cv_func(new_samples).detach()
        self.kernel_centers = torch.cat([self.kernel_centers, new_cvs])
        self.is_built = True

    def check_convergence(self):
        """Checks if the bias potential has converged."""
        if self.kernel_centers.shape[0] < 2:
            return False

        with torch.no_grad():
            v_bias_curr = self.calculate_potential(x=self.conv_grid.unsqueeze(1), z_pred=self.conv_grid)
            max_diff = torch.max(torch.abs(v_bias_curr - self.v_bias_prev))
            self.v_bias_prev = v_bias_curr

        if max_diff < cfg.OPES_CONV_TOL:
            print(f"    OPES converged! Max bias change: {max_diff:.2e}")
            self.is_converged = True
            return True
        return False

    def update_state(self, previous_samples):
        """Resets the OPES bias state for a new on-the-fly run."""
        print("  Resetting OPES state for new on-the-fly run.")
        self.kernel_centers = torch.empty((0,), device=self.device)
        self.is_built = False
        self.is_converged = False
        self.v_bias_prev = torch.zeros_like(self.conv_grid)


# --- Kolmogorov Bias Implementation ---
class KolmogorovBias(BiasSource):
    """
    Implements the Kolmogorov bias potential based on the committor gradient.
    V_K(q(x)) = - (lambda / beta) * log(|grad q(x)|^2)
    """
    def __init__(self, lambda_k, device):
        super().__init__(device=device)
        self.name = "kolmogorov"
        self.prefactor = -lambda_k / cfg.BETA
        self.is_built = (self.prefactor != 0.0)

    def calculate_potential(self, x, q_pred=None, **kwargs):
        """Calculates the Kolmogorov bias potential."""
        if self.prefactor == 0.0:
            return torch.zeros(x.shape[0], device=self.device)
            
        dq = grad(q_pred, x, grad_outputs=torch.ones_like(q_pred), create_graph=True, allow_unused=True)[0]
        dq_norm2 = torch.sum(dq**2, dim=1) if dq is not None else torch.zeros_like(q_pred)
        # Add epsilon to prevent log(0)
        return self.prefactor * torch.log(dq_norm2 + 1e-12)


# --- Bias Manager ---
class BiasManager:
    """Manages a collection of bias sources."""
    def __init__(self, model, device):
        self.biases = []
        self.device = device
        self.model = model

    def add_bias(self, bias_source):
        """Adds a new bias source to the manager."""
        self.biases.append(bias_source)
        print(f"Bias source '{bias_source.name}' added to manager.")

    def calculate_bias_potential(self, x):
        """Calculates the total bias and its components."""
        potentials = {'total': torch.zeros(x.shape[0], device=self.device)}
        
        g_pred, z_pred, q_pred, alpha = self.model(x)
        cvs = {'g_pred': g_pred, 'z_pred': z_pred, 'q_pred': q_pred, 'alpha': alpha}

        for bias in self.biases:
            if bias.is_built:
                v_component = bias.calculate_potential(x, **cvs)
                potentials[bias.name] = v_component
                potentials['total'] += v_component
        return potentials

    def calculate_bias_force(self, x):
        """
        Calculates the total force from all managed bias potentials.
        The force is the negative gradient of the total bias potential.
        """
        if not any(b.is_built for b in self.biases):
            return torch.zeros_like(x)

        x_for_grad = x.clone().requires_grad_(True)
        v_total = self.calculate_bias_potential(x_for_grad)['total']
        grad_v = torch.autograd.grad(outputs=v_total.sum(), inputs=x_for_grad)[0]
        return -grad_v

    def update_all_states(self, previous_samples):
        """Delegates state updates to all managed biases."""
        for bias in self.biases:
            bias.update_state(previous_samples)
