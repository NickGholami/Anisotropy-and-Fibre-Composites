"""Exercise 04 - Ply failure criteria and first-ply failure"""

#%% SETUP
import numpy as np
from composites import ply, laminate as lam, failure

try:
    from IPython import get_ipython
except ImportError:
    pass  # running as a normal script
else:
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")

#%% INPUT DATA
# ---------- INPUT DATA ----------
# Same laminate as Exercise 03
E1 = 46000     # [MPa]
E2 = 13000     # [MPa]
G12 = 4400     # [MPa]
nu12 = 0.30

# Top to bottom
angles = np.array([0, 90, 90, 0])
thicknesses = np.array([0.25, 0.25, 0.25, 0.25])  # [mm]

# Strengths [sigma_1t, sigma_1c, sigma_2t, sigma_2c, tau_12] [MPa]
strengths = np.array([1000, 1000, 35, 150, 50])
F12_star = -0.5  # [-] Tsai-Wu interaction coefficient

# Load case 4a: loads at x = 1 (the loop scales them by x)
N_4a = np.array([300.0, -50.0, 25.0])  # [N/mm]
M_4a = np.array([50.0, 0.0, 0.0])      # [N]

# Load case 4b: fixed loads
N_4b = np.array([300.0, -50.0, 25.0])  # [N/mm]
M_4b = np.array([0.0, 0.0, 5.0])       # [N]

# Load stepping for 4a
dx = 0.001     # [-] load step
x_max = 3.0    # [-] max x

positions = ["top", "middle", "bottom"]
n_plies = len(angles)
n_crit = len(failure.CRITERIA)

#%% SOLUTION
# ---------- SOLUTION ----------

# Laminate stiffness
S_local = ply.constitutive_matrix_2D(E1, E2, G12, nu12)
Q_local = np.linalg.inv(S_local)

# Positive z points downwards
z = lam.z_coordinates(thicknesses)

A, B, D = lam.abd_matrices(Q_local, angles, z)
ABD = np.block([[A, B], [B, D]])


# 4a: Increasing x until each ply fails
# Ply failure does not change the stiffness, so ABD stays the same.
x_fail = np.full((n_plies, 3, n_crit), np.inf)  # [ply, position, criterion]

for x in np.arange(dx, x_max, dx):
    loads = x * np.concatenate((N_4a, M_4a))

    # Laminate strains and curvatures
    eps_kappa_x = np.linalg.solve(ABD, loads)
    epsilon_0_x = eps_kappa_x[:3]
    kappa_x = eps_kappa_x[3:]

    # Local strains and stresses (top, middle, bottom of each ply)
    strains_x, stresses_x = lam.local_ply_response(Q_local, angles, z,
                                                   epsilon_0_x, kappa_x)

    # Failure criteria: save x when a criterion fails
    for i in range(n_plies):
        for k in range(3):
            f = failure.failure_criteria(stresses_x[i, k], strains_x[i, k],
                                         strengths, E1, E2, G12, F12_star)
            for j in range(n_crit):
                if f[j] >= 1 and x_fail[i, k, j] == np.inf:
                    x_fail[i, k, j] = x

# First-ply failure: smallest x of all plies and positions, for each criterion
x_fpf = np.zeros(n_crit)
for j in range(n_crit):
    x_fpf[j] = np.min(x_fail[:, :, j])


# 4b: Failure at a fixed load
# Laminate strains and curvatures
loads_4b = np.concatenate((N_4b, M_4b))
eps_kappa = np.linalg.solve(ABD, loads_4b)
epsilon_0 = eps_kappa[:3]
kappa = eps_kappa[3:]

# Local strains and stresses (top, middle, bottom of each ply)
strains, stresses = lam.local_ply_response(Q_local, angles, z,
                                           epsilon_0, kappa)

# Failure indices [ply, position, criterion]
indices = np.zeros((n_plies, 3, n_crit))
for i in range(n_plies):
    for k in range(3):
        indices[i, k] = failure.failure_criteria(stresses[i, k], strains[i, k],
                                                 strengths, E1, E2, G12, F12_star)

# 0/1 failure index: 1 = failure (index >= 1)
failure_0_1 = (indices >= 1).astype(int)
ply_failure = np.max(failure_0_1, axis=1)  # 1 if it fails at either top, middle or bottom position

#%% OUTPUT
# ---------- OUTPUT ----------

# 4a
print("4a: increasing x until ply failure")
print(f"Loads: N = x * {N_4a} N/mm, M = x * {M_4a} N, dx = {dx}")

print("\nx at failure:")
print(f"{'Ply':>3} {'Angle':>6} {'Position':>9}"
      f"{'Max stress':>12}{'Max strain':>12}{'Tsai-Hill':>12}{'Tsai-Wu':>12}{'Hoffman':>12}")
for i in range(n_plies):
    for k in range(3):
        row = ""
        for j in range(n_crit):
            row += f"{x_fail[i, k, j]:12.3f}"
        print(f"{i + 1:3d} {angles[i]:6d} {positions[k]:>9}{row}")

row = ""
for j in range(n_crit):
    row += f"{x_fpf[j]:12.3f}"
print(f"\n{'First-ply failure':>19}{row}")

# 4b
print("\n4b: fixed load")
print(f"Loads: N = {N_4b} N/mm, M = {M_4b} N")
print(f"Mid-plane strains [eps_x, eps_y, gamma_xy]: {np.round(epsilon_0, 6)}")
print(f"Curvatures [kappa_x, kappa_y, kappa_xy] [1/mm]: {np.round(kappa, 6)}")

print("\nLocal stresses [MPa] and strains:")
print(f"{'Ply':>3} {'Angle':>6} {'Position':>9}"
      f"{'sigma_1':>10}{'sigma_2':>10}{'tau_12':>10}"
      f"{'eps_1':>11}{'eps_2':>11}{'gamma_12':>11}")
for i in range(n_plies):
    for k in range(3):
        s1, s2, t12 = stresses[i, k]
        e1, e2, g12 = strains[i, k]
        print(f"{i + 1:3d} {angles[i]:6d} {positions[k]:>9}"
              f"{s1:10.2f}{s2:10.2f}{t12:10.2f}"
              f"{e1:11.6f}{e2:11.6f}{g12:11.6f}")

print("\n0/1 failure index (1 = failure):")
print(f"{'Ply':>3} {'Angle':>6} {'Position':>9}"
      f"{'Max stress':>12}{'Max strain':>12}{'Tsai-Hill':>12}{'Tsai-Wu':>12}{'Hoffman':>12}")
for i in range(n_plies):
    for k in range(3):
        row = ""
        for j in range(n_crit):
            row += f"{failure_0_1[i, k, j]:12d}"
        print(f"{i + 1:3d} {angles[i]:6d} {positions[k]:>9}{row}")

print("\n0/1 failure index per ply (1 = fails at any position):")
print(f"{'Ply':>3} {'Angle':>6} {'':>9}"
      f"{'Max stress':>12}{'Max strain':>12}{'Tsai-Hill':>12}{'Tsai-Wu':>12}{'Hoffman':>12}")
for i in range(n_plies):
    row = ""
    for j in range(n_crit):
        row += f"{ply_failure[i, j]:12d}"
    print(f"{i + 1:3d} {angles[i]:6d} {'':>9}{row}")