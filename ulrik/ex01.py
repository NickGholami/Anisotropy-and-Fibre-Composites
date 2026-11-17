"""Exercise 01 - Micro-mechanics"""

#%% SETUP
import numpy as np
from composites import ply

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
E_m = 4     # [GPa] Young's modulus of matrix
nu_m = 0.30 # [-]   Poisson's ratio of matrix
E_f = 76    # [GPa] Young's modulus of fibre
nu_f = 0.20 # [-]   Poisson's ratio of fibre
v_f = 0.55  # [-]   Fibre volume fraction (matrix: v_m = 1 - v_f)
xi_1 = 1    # [-]   Xi used for Halpin-Tsai E_2
xi_2 = 1    # [-]   Xi used for Halpin-Tsai G_12

#%% SOLUTION
# ---------- SOLUTION ----------

# Exercise 1.1 - Calculate the 5 independent parameters (E_1,E_2,G_12,nu_12, & nu_23)

(E1, E2_reuss, E2_ht, G12_reuss, G12_ht,
 nu12, nu23_reuss, nu23_ht) = ply.stiffness_parameters(
     v_f, E_f, E_m, nu_m, nu_f, xi_1, xi_2)

# Exercise 1.2 - Calculate the constitutive matrix, S, for both full 3D and plane stress (2D)

# Compliance matrices — Reuss
S_3D_reuss = ply.constitutive_matrix_3D(
    E1, E2_reuss, G12_reuss, nu12, nu23_reuss
)
S_2D_reuss = ply.constitutive_matrix_2D(E1, E2_reuss, G12_reuss, nu12)

# Compliance matrices — Halpin-Tsai E2
S_3D_ht = ply.constitutive_matrix_3D(
    E1, E2_ht, G12_ht, nu12, nu23_ht
)
S_2D_ht = ply.constitutive_matrix_2D(E1, E2_ht, G12_ht, nu12)

# Exercise 1.3 - Calculate the reduced stiffness matrix, Q, for plane stress (2D)

# 2D Stifness Matrix - Reuss
Q_2D_reuss = np.linalg.inv(S_2D_reuss)

# 2D Stifness Matrix - Reuss
Q_2D_ht = np.linalg.inv(S_2D_ht)



#%% OUTPUT
# ---------- OUTPUT ----------
# Exercise 1.1 - Printing the 5 independent parameters (E_1,E_2,G_12,nu_12, & nu_23)
print("Exercise 1.1 - Printing the 5 independent parameters \n")
print("E1 =", E1, "GPa")
print("E2 (Reuss) =", E2_reuss, "GPa")
print("E2 (Halpin-Tsai) =", E2_ht, "GPa")
print("G12 (Reuss) =", G12_reuss, "GPa")
print("G12 (Halpin-Tsai) =", G12_ht, "GPa")
print("nu12 =", nu12)
print("nu23 (Reuss) =", nu23_reuss)
print("nu23 (Halpin-Tsai) =", nu23_ht)


# Exercise 1.2 - Calculate the compliance matrix, S, for both full 3D and plane stress (2D)
print('\nExercise 1.2 - The compliance matrix, S\n')
print("S 3D — Reuss (1/GPa):")
print(S_3D_reuss)
print("\nS 2D — Reuss (1/GPa):")
print(S_2D_reuss)
print("\nS 3D — Halpin-Tsai (1/GPa):")
print(S_3D_ht)
print("\nS 2D — Halpin-Tsai (1/GPa):")
print(S_2D_ht)

# Exercise 1.3 - Calculate the reduced stiffness matrix, Q, for plane stress (2D)
print('\nExercise 1.3 The reduced stiffness matrix, Q\n')
print("Q 2D — Reuss E2 (GPa):")
print(Q_2D_reuss)
print("\nQ 2D — Halpin-Tsai E2 (GPa):")
print(Q_2D_ht)