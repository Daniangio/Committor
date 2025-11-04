# config.py
# Central configuration file for hyperparameters and constants.

import numpy as np
import torch

# --- Simulation & Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Potential Parameters (Müller-Brown) ---
# See potential.py for the full definition
A = np.array([-200, -100, -170, 15], dtype=float)
a = np.array([-1, -1, -6.5, 0.7], dtype=float)
b = np.array([0, 0, 11, 0.6], dtype=float)
c = np.array([-10, -10, -6.5, 0.7], dtype=float)
x0 = np.array([1, 0, -0.5, -1], dtype=float)
y0 = np.array([0, 0.5, 1.5, 1], dtype=float)
POTENTIAL_OFFSET = 150.0 # Shift minimum to be near zero

# --- Domain and Boundary Definitions ---
X_MIN, X_MAX = -1.5, 1.5
Y_MIN, Y_MAX = -1.0, 2.0

# Approximate centers for reactant (A) and product (B) basins
A_CENTER = np.array([-0.558, 1.441])
B_CENTER = np.array([0.623, 0.028])
RADIUS = 0.12 # Radius for defining boundary regions A and B

# --- Sampling Parameters ---
BETA = 0.05  # Inverse temperature (1/kT)
N_SAMPLES_UNBIASED_INITIAL = 1024 # Number of samples to generate from unbiased dataset for boundary conditions
N_SAMPLES_PER_ITER = 4096   # Number of samples to generate in each adaptive iteration
LANGEVIN_DT = 1e-5          # Timestep for Langevin dynamics
LANGEVIN_N_STEPS = 5000     # Number of steps per trajectory
LANGEVIN_RECORD_STRIDE = 10 # Record a point every N steps to reduce correlation
N_WALKERS = 50              # Number of parallel Langevin walkers

# --- Training Parameters ---
# Iterative refinement settings
N_ITERATIONS = 5           # Number of adaptive sampling/re-weighting iterations
N_EPOCHS_PER_ITER = 100    # Number of training epochs for each iteration
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
LOG_LOSS_EVER_N_EPOCHS = 20

# Model architecture
HIDDEN_UNITS = 256

# --- OPES Biasing Parameters ---
# Bias factor (gamma in the paper). Larger gamma -> flatter sampling.
OPES_BIAS_FACTOR = 50.0
# Width of the Gaussian kernels for the KDE of the CV distribution.
OPES_KERNEL_SIGMA = 0.5
# The flooding barrier height (Delta E in the paper). This is crucial.
# It sets the maximum energy up to which the bias will fill.
# This should be chosen to be less than the true free energy barrier.
OPES_DELTA_E = 150.0 # Units of kT/BETA
OPES_CV_MIN = -8.0
OPES_CV_MAX = 8.0
OPES_CV_BINS = 400

# On-the-fly OPES parameters
MAX_OPES_EQUILIBRATION_STEPS = 10000 # Max steps to run for OPES convergence before production
OPES_STRIDE = 200          # Deposit a new kernel every N steps
OPES_CONV_TOL = 1e-1       # Convergence tolerance for the max change in bias potential
OPES_CONV_GRID_PTS = 1000  # Number of points to test convergence on

# Strength of the Kolmogorov bias V_K = - (lambda/beta) * log(|grad q|^2)
W_KOLMOGOROV = 0 # 0.1


# --- Loss function weights ---
# Variational losses (applied to the biased, reweighted dataset)
W_EIK = 0.1     # Eikonal loss
W_COMM = 1.0    # Committor loss (weighted Dirichlet energy)
W_LINK = 0.     # Link between g and q
W_NONNEG = 1.0  # Penalty for g < 0

# Boundary loss (applied to the separate, UNBIASED dataset)
W_UNBIASED_BOUND = 10.0

# --- Visualization ---
GRID_NX, GRID_NY = 60, 60 # Resolution for plotting grids