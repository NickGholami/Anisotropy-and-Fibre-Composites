"""Exercise 06 - Bending of simply supported orthotropic plates"""

#%% SETUP
import numpy as np
from composites import ply, laminate as lam, plate

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
# Carbon-epoxy
E1 = 181000    # [MPa]
E2 = 10300     # [MPa]
G12 = 7170     # [MPa]
nu12 = 0.28

t_ply = 0.5    # [mm] thickness of each ply

# Lay-ups, top to bottom (8 plies each)
layup_names = ["[0 0 0 0]s", "[90 90 90 90]s", "[0 90 0 90]s", "[+45 -45 +45 -45]s"]
layups = [np.array([0, 0, 0, 0, 0, 0, 0, 0]),
          np.array([90, 90, 90, 90, 90, 90, 90, 90]),
          np.array([0, 90, 0, 90, 90, 0, 90, 0]),
          np.array([45, -45, 45, -45, -45, 45, -45, 45])]

# Panels [mm]
panel_names = ["Square, a = b = 1000 mm", "Rectangular, a = 500 mm, b = 1000 mm"]
a_values = [1000.0, 500.0]
b_values = [1000.0, 1000.0]

# Uniformly distributed load q on a patch of the plate
q = 0.001      # [MPa] = 1 kPa

# Load cases: patch alpha[0]*a < x < alpha[1]*a and beta[0]*b < y < beta[1]*b
load_names = ["Whole plate loaded", "Quarter of the plate loaded (centre)"]
alphas = [[0.0, 1.0], [0.25, 0.75]]
betas = [[0.0, 1.0], [0.25, 0.75]]

# Number of terms in each sum
n_terms_list = [1,3,5,7,11,19,25]

n_layups = len(layups)
n_panels = len(a_values)
n_loads = len(load_names)
n_runs = len(n_terms_list)
last = n_runs - 1   # index of the result with the most terms

#%% SOLUTION
# ---------- SOLUTION ----------
S_local = ply.constitutive_matrix_2D(E1, E2, G12, nu12)
Q_local = np.linalg.inv(S_local)

# Bending stiffness D for each lay-up
D_all = []
for j in range(n_layups):
    thicknesses = np.full(len(layups[j]), t_ply)
    z = lam.z_coordinates(thicknesses)
    A, B, D = lam.abd_matrices(Q_local, layups[j], z)
    D_all.append(D)

# Deflection and moments for each panel, load case and lay-up
# [panel, load case, lay-up, number of terms]
w_c = np.zeros((n_panels, n_loads, n_layups, n_runs))     # w at centre [mm]
Mx_c = np.zeros((n_panels, n_loads, n_layups, n_runs))    # Mx at centre [N]
My_c = np.zeros((n_panels, n_loads, n_layups, n_runs))    # My at centre [N]
Mxy_c = np.zeros((n_panels, n_loads, n_layups, n_runs))   # Mxy at corner [N]

for p in range(n_panels):
    a = a_values[p]
    b = b_values[p]
    for c in range(n_loads):
        alpha = alphas[c]
        beta = betas[c]
        for j in range(n_layups):
            for r in range(n_runs):
                # w, Mx and My at the centre
                w, Mx, My, Mxy = plate.simply_supported_plate(D_all[j], a, b, q, alpha, beta,
                                                              a / 2, b / 2, n_terms_list[r])
                w_c[p, c, j, r] = w
                Mx_c[p, c, j, r] = Mx
                My_c[p, c, j, r] = My

                # Mxy at the corner
                w, Mx, My, Mxy = plate.simply_supported_plate(D_all[j], a, b, q, alpha, beta,
                                                              0.0, 0.0, n_terms_list[r])
                Mxy_c[p, c, j, r] = Mxy

#%% OUTPUT
# ---------- OUTPUT ----------
header = ""
for name in layup_names:
    header += f"{name:>20}"

# D matrices
print("Bending stiffness D [N mm]")
print(f"{'':>10}{header}")
labels = ["D11", "D12", "D22", "D66", "D16", "D26"]
row_index = [0, 0, 1, 2, 0, 1]   # D11 is D[0, 0], D12 is D[0, 1]
col_index = [0, 1, 1, 2, 2, 2]
for k in range(len(labels)):
    row = ""
    for j in range(n_layups):
        row += f"{D_all[j][row_index[k], col_index[k]]:20.0f}"
    print(f"{labels[k]:>10}{row}")
print("D16 and D26 are not zero for [+45 -45 +45 -45]s, "
      "so the orthotropic solution is only an approximation there.")

# for each panel and load case
for p in range(n_panels):
    for c in range(n_loads):
        print(f"\n{panel_names[p]} | {load_names[c]}, q = {q} MPa")
        print(f"m, n = 1, 2, 3, ... {n_terms_list[last]}")
        print(f"{'':>16}{header}")

        row_w = ""
        row_mx = ""
        row_my = ""
        row_mxy = ""
        for j in range(n_layups):
            row_w += f"{w_c[p, c, j, last]:20.4f}"
            row_mx += f"{Mx_c[p, c, j, last]:20.4f}"
            row_my += f"{My_c[p, c, j, last]:20.4f}"
            row_mxy += f"{abs(Mxy_c[p, c, j, last]):20.4f}"
        print(f"{'w centre [mm]':>16}{row_w}")
        print(f"{'Mx centre [N]':>16}{row_mx}")
        print(f"{'My centre [N]':>16}{row_my}")
        print(f"{'|Mxy| corner [N]':>16}{row_mxy}")

        print(f"\n Best lay-up is the one with the smallest w")

        print("\nConvergence of w centre [mm]")
        print(f"{'m, n up to':>16}{header}")
        for r in range(n_runs):
            row = ""
            for j in range(n_layups):
                row += f"{w_c[p, c, j, r]:20.4f}"
            print(f"{n_terms_list[r]:>16}{row}")