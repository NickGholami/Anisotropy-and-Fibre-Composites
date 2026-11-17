"""Exercise 07 - Buckling and free vibration of simply supported orthotropic plates"""

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

# Panels [mm] and density [kg/m^3]
panel_names = ["Square, a = b = 1000 mm, rho = 1500 kg/m^3",
               "I)  a = 500 mm, b = 1000 mm, rho = 1397.5 kg/m^3",
               "II) a = 1000 mm, b = 500 mm, rho = 1397.5 kg/m^3"]
a_values = [1000.0, 500.0, 1000.0]
b_values = [1000.0, 1000.0, 500.0]
rho_values = [1500.0, 1397.5, 1397.5]   # [kg/m^3]

# Buckling: number of half-waves in x (n = 1 always gives the lowest load)
m_values = [1, 2, 3, 4, 5]

# Vibration: number of lowest frequencies, and largest m and n to search
n_modes = 4
mn_max = 6

n_layups = len(layups)
n_panels = len(a_values)
n_m = len(m_values)

#%% SOLUTION
# ---------- SOLUTION ----------
S_local = ply.constitutive_matrix_2D(E1, E2, G12, nu12)
Q_local = np.linalg.inv(S_local)

# Bending stiffness D and total thickness h for each lay-up
D_all = []
h_all = []
for j in range(n_layups):
    thicknesses = np.full(len(layups[j]), t_ply)
    z = lam.z_coordinates(thicknesses)
    A, B, D = lam.abd_matrices(Q_local, layups[j], z)
    D_all.append(D)
    h_all.append(np.sum(thicknesses))

# Buckling loads [panel, lay-up, m]
P = np.zeros((n_panels, n_layups, n_m))   # [N/mm]
for p in range(n_panels):
    for j in range(n_layups):
        for k in range(n_m):
            P[p, j, k] = plate.buckling_load(D_all[j], a_values[p], b_values[p], m_values[k])

# Lowest buckling load and its m [panel, lay-up]
P_cr = np.zeros((n_panels, n_layups))
m_cr = np.zeros((n_panels, n_layups), dtype=int)
for p in range(n_panels):
    for j in range(n_layups):
        k_min = np.argmin(P[p, j, :])   # index of the smallest P over all m
        P_cr[p, j] = P[p, j, k_min]
        m_cr[p, j] = m_values[k_min]

# Lowest natural frequencies [panel, lay-up, mode]
omega = np.zeros((n_panels, n_layups, n_modes))            # [rad/s]
m_mode = np.zeros((n_panels, n_layups, n_modes), dtype=int)
n_mode = np.zeros((n_panels, n_layups, n_modes), dtype=int)
rho_star = np.zeros((n_panels, n_layups))                  # [N s^2/mm^3] mass per area
for p in range(n_panels):
    for j in range(n_layups):
        # rho* = rho * h. 1 kg/m^3 = 1e-12 N s^2/mm^4 (tonne/mm^3)
        rho_star[p, j] = rho_values[p] * 1e-12 * h_all[j]
        omega_list, m_list, n_list = plate.lowest_frequencies(D_all[j], a_values[p], b_values[p],
                                                              rho_star[p, j], n_modes, mn_max)
        omega[p, j, :] = omega_list
        m_mode[p, j, :] = m_list
        n_mode[p, j, :] = n_list

#%% OUTPUT
# ---------- OUTPUT ----------
header = ""
for name in layup_names:
    header += f"{name:>20}"

# D matrices
print("Bending stiffness D [N mm]")
print(f"{'':>14}{header}")
labels = ["D11", "D12", "D22", "D66", "D16", "D26"]
row_index = [0, 0, 1, 2, 0, 1]   # D11 is D[0, 0], D12 is D[0, 1]
col_index = [0, 1, 1, 2, 2, 2]
for k in range(len(labels)):
    row = ""
    for j in range(n_layups):
        row += f"{D_all[j][row_index[k], col_index[k]]:20.0f}"
    print(f"{labels[k]:>14}{row}")
print("D16 and D26 are not zero for [+45 -45 +45 -45]s, "
      "so the orthotropic solution is only an approximation there.")

for p in range(n_panels):
    print(f"\n========== {panel_names[p]} ==========")

    # Buckling
    print("\nBuckling load P [N/mm], n = 1")
    print(f"{'m':>14}{header}")
    for k in range(n_m):
        row = ""
        for j in range(n_layups):
            row += f"{P[p, j, k]:20.4f}"
        print(f"{m_values[k]:>14}{row}")
    row_P = ""
    row_m = ""
    for j in range(n_layups):
        row_P += f"{P_cr[p, j]:20.4f}"
        row_m += f"{m_cr[p, j]:20d}"
    print(f"{'P_cr':>14}{row_P}")
    print(f"{'m at P_cr':>14}{row_m}")

    # Vibration
    print("\nLowest natural frequencies omega_mn [rad/s]")
    row = ""
    for j in range(n_layups):
        row += f"{rho_star[p, j] * 1e9:20.3f}"
    print(f"{'rho* [kg/m^2]':>14}{row}")
    for mode in range(n_modes):
        row = ""
        for j in range(n_layups):
            label = f"omega{m_mode[p, j, mode]}{n_mode[p, j, mode]} ="
            row += f"{label:>12}{omega[p, j, mode]:8.2f}"
        print(f"{'mode ' + str(mode + 1):>14}{row}")
