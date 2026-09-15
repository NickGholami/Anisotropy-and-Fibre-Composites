# %%

import numpy as np


def stiffness_parameters(v_f, E_f, E_m, nu_m, nu_f, xi_1=1, xi_2=1):
    # Volume fraction of matrix
    v_m = 1 - v_f

    # E1 - Rule of mixtures
    E1 = v_f * E_f + v_m * E_m

    # E2 - Equal stress / Reuss
    E2 = 1 / (v_f / E_f + v_m / E_m)

    # Shear modulus for fiber and matrix
    G_f = E_f / (2 * (1 + nu_f))
    G_m = E_m / (2 * (1 + nu_m))

    # E2_ht - Halpin-Tsai
    eta_1 = (E_f / E_m - 1) / (E_f / E_m + xi_1)
    E2_ht = E_m * (1 + xi_1 * eta_1 * v_f) / (1 - eta_1 * v_f)

    # G12 - Halpin-Tsai
    eta_2 = (G_f / G_m - 1) / (G_f / G_m + xi_2)
    G12 = G_m * (1 + xi_2 * eta_2 * v_f) / (1 - eta_2 * v_f)

    # Poisson's ratio nu12
    nu12 = v_f * nu_f + v_m * nu_m

    # Bulk modulus
    K_f = E_f / (3 * (1 - 2 * nu_f))
    K_m = E_m / (3 * (1 - 2 * nu_m))
    K = 1 / (v_f / K_f + v_m / K_m)

    # Poisson's ratios using your original nu23 formula
    nu21_reuss = nu12 * E2 / E1
    nu21_ht = nu12 * E2_ht / E1

    nu23_reuss = 1 - nu21_reuss - E2 / (3 * K)
    nu23_ht = 1 - nu21_ht - E2_ht / (3 * K)

    return E1, E2, E2_ht, G12, nu12, nu23_reuss, nu23_ht


def constitutive_matrix_3D(E1, E2, G12, nu12, nu23):
    E3 = E2

    nu13 = nu12
    nu21 = nu12 * E2 / E1
    nu31 = nu21
    nu32 = nu23

    G23 = E2 / (2 * (1 + nu23))
    G31 = G12

    # Order: 11, 22, 33, 23, 31, 12
    # Engineering shear strains
    S = np.array([
        [1/E1,     -nu21/E2, -nu31/E3, 0,     0,     0],
        [-nu12/E1,  1/E2,    -nu32/E3, 0,     0,     0],
        [-nu13/E1, -nu23/E2,  1/E3,    0,     0,     0],
        [0,         0,        0,       1/G23, 0,     0],
        [0,         0,        0,       0,     1/G31, 0],
        [0,         0,        0,       0,     0,     1/G12]
    ])

    return S


def constitutive_matrix_2D(E1, E2, G12, nu12):
    nu21 = nu12 * E2 / E1

    # Plane stress; order: 11, 22, 12
    S = np.array([
        [1/E1,    -nu21/E2, 0],
        [-nu12/E1, 1/E2,    0],
        [0,        0,       1/G12]
    ])

    return S


def inverted_constitutive_matrix(S):
    return np.linalg.inv(S)


# Input — moduli in GPa

Em = 4
num = 0.30

Ef = 76
nuf = 0.20

vf = 0.55

xi1 = 1
xi2 = 1


# Stiffness parameters

E1, E2, E2_ht, G12, nu12, nu23_reuss, nu23_ht = stiffness_parameters(
    vf, Ef, Em, num, nuf, xi1, xi2
)


# Constitutive matrices — Reuss E2

S_3D_reuss = constitutive_matrix_3D(
    E1, E2, G12, nu12, nu23_reuss
)

S_2D_reuss = constitutive_matrix_2D(E1, E2, G12, nu12)

Q_2D_reuss = inverted_constitutive_matrix(S_2D_reuss)


# Constitutive matrices — Halpin-Tsai E2

S_3D_ht = constitutive_matrix_3D(
    E1, E2_ht, G12, nu12, nu23_ht
)

S_2D_ht = constitutive_matrix_2D(E1, E2_ht, G12, nu12)

Q_2D_ht = inverted_constitutive_matrix(S_2D_ht)


# Display results

np.set_printoptions(precision=6, suppress=True)

print("E1 =", E1, "GPa")
print("E2 (Reuss) =", E2, "GPa")
print("E2 (Halpin-Tsai) =", E2_ht, "GPa")
print("G12 =", G12, "GPa")
print("nu12 =", nu12)
print("nu23 (Reuss) =", nu23_reuss)
print("nu23 (Halpin-Tsai) =", nu23_ht)

print("\nS 3D — Reuss E2 (1/GPa):")
print(S_3D_reuss)

print("\nS 3D — Halpin-Tsai E2 (1/GPa):")
print(S_3D_ht)

print("\nS 2D — Reuss E2 (1/GPa):")
print(S_2D_reuss)

print("\nS 2D — Halpin-Tsai E2 (1/GPa):")
print(S_2D_ht)

print("\nQ 2D — Reuss E2 (GPa):")
print(Q_2D_reuss)

print("\nQ 2D — Halpin-Tsai E2 (GPa):")
print(Q_2D_ht)