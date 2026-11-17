"""Exercise 05A - Progressive failure with the ply discount method"""

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
# Same laminate as Exercise 03 and 04
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

# Load case 4a: loads at x = 1 (the load is applied in steps of dx)
N_4a = np.array([300.0, -50.0, 25.0])  # [N/mm]
M_4a = np.array([50.0, 0.0, 0.0])      # [N]

# Load stepping
dx = 0.001     # [-] load step
x_max = 3.0    # [-] max x

positions = ["top", "middle", "bottom"]
n_plies = len(angles)
n_crit = len(failure.CRITERIA)

#%% SOLUTION
# ---------- SOLUTION ----------
S_local = ply.constitutive_matrix_2D(E1, E2, G12, nu12)
Q_local = np.linalg.inv(S_local)

# Positive z points downwards
z = lam.z_coordinates(thicknesses)

# Load increment at x = 1 (multiplied by dx in each step)
loads = np.concatenate((N_4a, M_4a))

# x at failure of each ply [ply, criterion] (inf = no failure up to x_max)
x_fail = np.full((n_plies, n_crit), np.inf)

# Each criterion gives its own failure sequence, so start from zero each time
for j in range(n_crit):
    ply_failed = np.zeros(n_plies, dtype=bool)  # True when the ply has failed
    strains = np.zeros((n_plies, 3, 3))         # [ply, position, component]
    stresses = np.zeros((n_plies, 3, 3))        # [ply, position, component]

    for x in np.arange(dx, x_max, dx):
        # Stop at last-ply failure
        if np.all(ply_failed):
            break

        # ABD from the intact plies only (ply discount: failed ply has Q = 0)
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        for i in range(n_plies):
            if not ply_failed[i]:
                A_i, B_i, D_i = lam.abd_matrices(Q_local, [angles[i]], [z[i], z[i + 1]])
                A += A_i
                B += B_i
                D += D_i
        ABD = np.block([[A, B], [B, D]])

        # Load increment - increments of mid-plane strains and curvatures
        d_eps_kappa = np.linalg.solve(ABD, dx * loads)
        d_epsilon_0 = d_eps_kappa[:3]
        d_kappa = d_eps_kappa[3:]

        # Increments of local strains and stresses (top, middle, bottom)
        d_strains, d_stresses = lam.local_ply_response(Q_local, angles, z,
                                                       d_epsilon_0, d_kappa)

        # Intact plies: new value = old value + increment, then check failure
        for i in range(n_plies):
            if ply_failed[i]:
                continue

            strains[i] += d_strains[i]
            stresses[i] += d_stresses[i]

            for k in range(3):
                f = failure.failure_criteria(stresses[i, k], strains[i, k],
                                             strengths, E1, E2, G12, F12_star)
                if f[j] >= 1:
                    ply_failed[i] = True

            if ply_failed[i]:
                x_fail[i, j] = x


# First-ply failure and last-ply failure for each criterion
x_fpf = np.min(x_fail, axis=0)   # first-ply failure: smallest x over the plies
x_lpf = np.max(x_fail, axis=0)   # last-ply failure: largest x over the plies

#%% OUTPUT
# ---------- OUTPUT ----------
header = (f"{'Max stress':>12}{'Max strain':>12}{'Tsai-Hill':>12}"
          f"{'Tsai-Wu':>12}{'Hoffman':>12}")

# 5A
print("5A: progressive failure, ply discount")
print(f"Loads: N = x * {N_4a} N/mm, M = x * {M_4a} N, dx = {dx}")

print("\nx at failure of each ply:")
print(f"{'Ply':>3} {'Angle':>6}{header}")
for i in range(n_plies):
    row = ""
    for j in range(n_crit):
        row += f"{x_fail[i, j]:12.3f}"
    print(f"{i + 1:3d} {angles[i]:6d}{row}")

row_fpf = ""
row_lpf = ""
for j in range(n_crit):
    row_fpf += f"{x_fpf[j]:12.3f}"
    row_lpf += f"{x_lpf[j]:12.3f}"
print(f"\n{'FPF':>10}{row_fpf}")
print(f"{'LPF':>10}{row_lpf}")