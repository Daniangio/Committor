# config.py
# Central configuration file for hyperparameters and constants.

import numpy as np
import torch

# --- Simulation & Device ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
BETA = 0.1  # Inverse temperature (1/kT)
N_SAMPLES_PER_ITER = 2000 # Number of samples to generate in each adaptive iteration
LANGEVIN_DT = 1e-5         # Timestep for Langevin dynamics
LANGEVIN_N_STEPS = 5000    # Number of steps per trajectory
LANGEVIN_RECORD_STRIDE = 20 # Record a point every N steps to reduce correlation
N_WALKERS = 50             # Number of parallel Langevin walkers

# --- Training Parameters ---
# Iterative refinement settings
N_ITERATIONS = 5           # Number of adaptive sampling/re-weighting iterations
N_EPOCHS_PER_ITER = 1000   # Number of training epochs for each iteration
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024

# Model architecture
HIDDEN_UNITS = 160

# Loss function weights
W_EIK = 1.0     # Eikonal loss
W_COMM = 1.0    # Committor loss (weighted Dirichlet energy)
W_BOUND = 1.0   # Boundary conditions
W_LINK = 0.1    # Link between g and q
W_NONNEG = 1.0  # Penalty for g < 0

# Adaptive sampling bias strength
# This lambda controls how strongly we bias towards the transition state (q=0.5)
LAMBDA_BIAS = 1.0

# --- Visualization ---
GRID_NX, GRID_NY = 120, 120 # Resolution for plotting grids
