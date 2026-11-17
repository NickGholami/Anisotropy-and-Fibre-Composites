"""Exercise 03 - Laminate stiffness (ABD) and ply stresses"""

#%% SETUP
import numpy as np
from composites import ply, laminate as lam

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
E1 = 46000     # [MPa]
E2 = 13000     # [MPa]
G12 = 4400     # [MPa]
nu12 = 0.30

# Top to bottom
angles = np.array([0, 90, 90, 0])
thicknesses = np.array([0.25, 0.25, 0.25, 0.25])  # [mm]

N = np.array([300.0, -50.0, 25.0])   # [N/mm]
M = np.array([50.0, 0.0, 0.0])       # [N]

#%% SOLUTION
# ---------- SOLUTION ----------
S_local = ply.constitutive_matrix_2D(E1, E2, G12, nu12)
Q_local = np.linalg.inv(S_local)

# Positive z points downwards
z = lam.z_coordinates(thicknesses)

A, B, D = lam.abd_matrices(Q_local, angles, z)
ABD = np.block([[A, B], [B, D]])

loads = np.concatenate((N, M))
eps_kappa = np.linalg.solve(ABD, loads)
epsilon_0 = eps_kappa[:3]
kappa = eps_kappa[3:]

local_stresses = lam.local_ply_stresses(Q_local, angles, z, epsilon_0, kappa)

# Check: does doubling the loads double the stresses?
eps_kappa_double = np.linalg.solve(ABD, 2 * loads)
epsilon_0_double = eps_kappa_double[:3]
kappa_double = eps_kappa_double[3:]
local_stresses_double = lam.local_ply_stresses(Q_local, angles, z,
                                               epsilon_0_double, kappa_double)

tol = 1e-8 
stress_double = np.all(np.abs(local_stresses_double - 2 * local_stresses) < tol)

#%% OUTPUT
# ---------- OUTPUT ----------
print(f"Ply boundary positions z [mm]: \n {z}")
print(f"\nA [N/mm]: \n {A}")
print(f"\nB [N]: \n {B}")
print(f"\nD [N·mm]: \n {D}")

print(f"\nLoad vector [Nx, Ny, Nxy, Mx, My, Mxy]: \n {loads}")
print(f"\nMid-plane strains: \n {epsilon_0}")
print(f"\nCurvatures [1/mm]: \n {kappa}")

print("\nLocal stresses [MPa]:")
print(f"{'Ply':>3} {'Angle':>6} {'Face':>6} {'z [mm]':>9} "
      f"{'sigma_1':>12} {'sigma_2':>12} {'tau_12':>12}")
for i in range(len(angles)):
    angle = angles[i]
    # Top face
    s1, s2, t12 = local_stresses[i, 0]
    print(f"{i + 1:3d} {angle:6d} {'top':>6} {z[i]:9.2f} "
          f"{s1:12.2f} {s2:12.2f} {t12:12.2f}")
    # Bottom face
    s1, s2, t12 = local_stresses[i, 1]
    print(f"{i + 1:3d} {angle:6d} {'bottom':>6} {z[i + 1]:9.2f} "
          f"{s1:12.2f} {s2:12.2f} {t12:12.2f}")

print(f"\nDo all local stresses double? {stress_double}")
