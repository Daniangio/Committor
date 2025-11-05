# bias_refactored.py
# A modular framework for combining multiple sources of biasing potential.
# REFACTORED for On-the-Fly OPES:
# 1. OPES bias is now dynamic within an iteration, converging before production.
# 2. Implements reset(), add_kernels(), and check_convergence() methods.
# 3. Continues to use an efficient grid-based KDE.

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
    
    def reset(self):
        """Resets the internal state of the bias (e.g., for adaptive methods)."""
        pass

    def update_state(self, *args, **kwargs):
        """Updates the state of the bias (e.g., with samples from a previous run)."""
        pass

# --- OPES Bias Implementation (On-the-Fly) ---
class OPESBias(BiasSource):
    """
    REFACTORED: Implements an ON-THE-FLY OPES Flooding bias potential.
    The bias starts at zero, kernels are added every OPES_STRIDE steps, and the
    potential converges dynamically within a single sampling run.
    """
    def __init__(self, model, bias_factor, kernel_sigma, delta_E, device,
                 grid_min=-5.0, grid_max=5.0, grid_bins=2000):
        super().__init__(device=device)
        self.name = "opes_flooding"
        self.model = model
        self.cv_func = lambda x: self.model(x)[2] # Use z_pred as the CV

        if bias_factor < 1.0:
            raise ValueError("Bias factor gamma must be >= 1.0")
        if delta_E <= 0.0:
            raise ValueError("Barrier Delta_E must be positive.")

        self.bias_factor = bias_factor
        self.kernel_sigma = kernel_sigma
        self.delta_E = delta_E

        # Grid properties for efficient KDE
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_bins = grid_bins
        self.cv_grid = torch.linspace(self.grid_min, self.grid_max, self.grid_bins, device=self.device)
        self.delta_s = (self.grid_max - self.grid_min) / (self.grid_bins - 1)
        
        # Grid for testing convergence
        self.conv_test_grid = torch.linspace(self.grid_min, self.grid_max, cfg.OPES_CONV_GRID_PTS, device=self.device)
        self.v_bias_grid_old = torch.zeros_like(self.conv_test_grid)

        # State variables
        self.p_s_grid = torch.zeros_like(self.cv_grid)
        self.total_kernels = 0.0
        self.is_converged = False
        self.is_built = True # It's always "built" in on-the-fly mode, just might be zero

        # The prefactor for the bias calculation, (1 - 1/gamma)/beta
        self.prefactor = (1.0 - 1.0 / self.bias_factor) / cfg.BETA
        
        # The regularization term, epsilon, as defined in the paper
        self.epsilon = math.exp(-cfg.BETA * self.delta_E / (1.0 - 1.0 / self.bias_factor))

    def reset(self):
        """Resets the OPES state for a new sampling iteration."""
        print("  Resetting on-the-fly OPES bias state.")
        self.p_s_grid.zero_()
        self.v_bias_grid_old.zero_()
        self.total_kernels = 0.0
        self.is_converged = False

    def _get_prob_from_grid(self, s):
        """Calculates probability by interpolating from the pre-computed grid."""
        # Ensure s is within grid bounds
        s_clamped = torch.clamp(s, self.grid_min, self.grid_max)

        # Calculate fractional grid coordinates
        grid_coords = (s_clamped - self.grid_min) / self.delta_s

        # Get lower and upper integer indices
        idx_lower = torch.floor(grid_coords).long()
        idx_upper = idx_lower + 1

        # Clamp indices to valid range [0, self.grid_bins - 1]
        idx_lower = torch.clamp(idx_lower, 0, self.grid_bins - 1)
        idx_upper = torch.clamp(idx_upper, 0, self.grid_bins - 1)

        # Calculate interpolation weights
        frac = grid_coords - idx_lower.float() # Fractional part between lower and upper index

        # Get values from the probability grid at lower and upper indices
        p_lower = self.p_s_grid[idx_lower]
        p_upper = self.p_s_grid[idx_upper]

        # Perform linear interpolation
        interpolated_p = p_lower * (1 - frac) + p_upper * frac
        
        return interpolated_p

    def _calculate_potential_on_grid(self, grid_points):
        """Helper to calculate V_bias on a specific grid."""
        if self.total_kernels == 0:
            return torch.zeros_like(grid_points)
        
        # Estimate probability on the test grid
        p_s_test = self._get_prob_from_grid(grid_points)
        
        # Normalization Z is integral of P(s), approximated as sum(P_grid)*delta_s
        z_norm = self.p_s_grid.sum() * self.delta_s
        if z_norm < 1e-12: z_norm = 1.0

        argument = (p_s_test / z_norm) + self.epsilon
        v_opes_raw = self.prefactor * torch.log(argument)
        return v_opes_raw - v_opes_raw.max().detach()

    def get_potential_on_grid(self):
        """
        Public method to get the bias potential on the internal CV grid.
        Returns the grid and the potential on that grid.
        """
        return self.cv_grid, self._calculate_potential_on_grid(self.cv_grid)

    def calculate_potential(self, x, z_pred=None, **kwargs):
        """Calculates the current on-the-fly OPES bias potential."""
        if self.total_kernels == 0:
            return 0 * z_pred # Do not use torch.zeros_like(z_pred): breaks gradient computation
        
        p_s = self._get_prob_from_grid(z_pred)
        
        z_norm = self.p_s_grid.sum() * self.delta_s
        if z_norm < 1e-12: z_norm = 1.0
        
        argument = (p_s / z_norm) + self.epsilon
        v_opes_raw = self.prefactor * torch.log(argument)
        
        return v_opes_raw - v_opes_raw.max().detach()

    def add_kernels(self, positions):
        """Adds new kernels from the given walker positions to the probability grid."""
        cv_values = self.cv_func(positions).detach()
        
        grid_u = self.cv_grid.unsqueeze(1)
        centers_u = cv_values.unsqueeze(0)
        
        dist2 = (grid_u - centers_u)**2
        kernels = torch.exp(-0.5 * dist2 / self.kernel_sigma**2)
        
        # Add the sum of new kernels to the grid
        self.p_s_grid += kernels.sum(dim=1)
        self.total_kernels += positions.shape[0]

    def check_convergence(self):
        """Checks if the bias potential has converged."""
        v_bias_grid_new = self._calculate_potential_on_grid(self.conv_test_grid)
        max_diff = torch.max(torch.abs(v_bias_grid_new - self.v_bias_grid_old))
        
        # Store data for plotting before updating the old grid
        plot_data = {
            "v_bias_new": v_bias_grid_new.cpu().numpy(),
            "v_bias_old": self.v_bias_grid_old.cpu().numpy(),
            "cv_grid": self.conv_test_grid.cpu().numpy(),
            "max_diff": max_diff.item()
        }

        self.v_bias_grid_old = v_bias_grid_new.clone() # Update for next check
        if not self.is_converged and max_diff < cfg.OPES_CONV_TOL:
            self.is_converged = True # Set flag only once
        return plot_data

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
            return 0 * q_pred # Do not use torch.zeros(x.shape[0], device=self.device): breaks gradient computation
            
        dq = grad(q_pred, x, grad_outputs=torch.ones_like(q_pred), create_graph=True, allow_unused=True)[0]
        dq_norm2 = torch.sum(dq**2, dim=1) if dq is not None else torch.zeros_like(q_pred)
        return self.prefactor * torch.log(dq_norm2 + 1e-12)


# --- Bias Manager ---
class BiasManager:
    """Manages a collection of bias sources."""
    def __init__(self, model, device):
        self.biases = []
        self.device = device
        self.model = model

    def add_bias(self, bias_source):
        self.biases.append(bias_source)
        print(f"Bias source '{bias_source.name}' added to manager.")

    def calculate_bias_potential(self, x):
        potentials = {'total': torch.zeros(x.shape[0], device=self.device)}
        gA_pred, gB_pred, z_pred, q_pred, alpha = self.model(x)
        cvs = {'g_pred': gA_pred, 'z_pred': z_pred, 'q_pred': q_pred, 'alpha': alpha}

        for bias in self.biases:
            if bias.is_built:
                v_component = bias.calculate_potential(x, **cvs)
                potentials[bias.name] = v_component
                potentials['total'] += v_component
        return potentials

    def get_bias_grad_fn(self):
        def bias_grad_fn(x_in):
            if not any(b.is_built for b in self.biases):
                return torch.zeros_like(x_in)
            
            # Create a clone of x_in that requires gradients for this specific calculation.
            # This ensures that the computation graph for the bias potential is built
            # starting from x_for_grad.
            x_for_grad = x_in.clone().requires_grad_(True)

            # Evaluate the model with x_for_grad to ensure all CVs are part of the graph
            # that leads back to x_for_grad.
            g_for_grad, _, z_for_grad, q_for_grad, alpha_for_grad = self.model(x_for_grad)

            cvs_for_grad = {'g_pred': g_for_grad, 'z_pred': z_for_grad, 'q_pred': q_for_grad, 'alpha': alpha_for_grad}

            v_total = torch.zeros(x_in.shape[0], device=self.device)

            for bias in self.biases:
                # Kolmogorov bias needs gradients through the model, so we re-calculate it
                if bias.is_built:
                    # Pass the CVs derived from x_for_grad to ensure proper gradient tracking
                    v_total += bias.calculate_potential(x_for_grad, **cvs_for_grad)

            # The force is the negative gradient of the potential
            grad_v = torch.autograd.grad(outputs=v_total.sum(), inputs=x_for_grad)[0]
            return -grad_v
        return bias_grad_fn # This now returns a function that computes the force

    def reset_all_states(self):
        """Resets the state of all managed biases."""
        for bias in self.biases:
            bias.reset()
