# %%

import numpy as np

def stiffness_parameters(v_f, E_f, E_m, nu_m, nu_f, xi_1 = 1, xi_2 = 1):
    # Volume fraction of matrix
    v_m = 1 - v_f

    # E1
    E1 = v_f * E_f + (1-v_f) * E_m

    # E2 - Equal stress / Reuss
    E2 = ( v_f / E_f + (1 - v_f) / E_m )**(-1)


    # E2_ht - Halpin-Tsai




    # Shear modulus for fiber and matrix
    Gf = Ef / (2 * (1 + nuf))
    Gm = Em / (2 * (1 + num))

    # E2 - Halpin-Tsai
    eta1 = (Ef / Em - 1) / (Ef / Em + xi1)
    E2 = Em * (1 + xi1 * eta1 * vf) / (1 - eta1 * vf)

    # G12 - Halpin-Tsai
    eta2 = (Gf / Gm - 1) / (Gf / Gm + xi2)
    G12 = Gm * (1 + xi2 * eta2 * vf) / (1 - eta2 * vf)

    # Poisson's ratio nu12
    nu12 = vf * nuf + vm * num

    # nu21
    nu21 = nu12 * E2 / E1

    # Bulk modulus
    Kf = Ef / (3 * (1 - 2 * nuf))
    Km = Em / (3 * (1 - 2 * num))

    K = 1 / (vf / Kf + vm / Km)

    # Poisson's ratio nu23
    nu23 = 1 - nu21 - E2 / (3 * K)

    return E1, E2, G12, nu12, nu23


def constitutive_matrix_3D(E1, E2, G12, nu12, nu23):

    E3 = E2

    nu13 = nu12
    nu21 = nu12 * E2 / E1
    nu31 = nu21
    nu32 = nu23

    G23 = E2 / (2 * (1 + nu23))
    G31 = G12

    S = np.array([
        [1/E1,     -nu21/E2, -nu31/E3, 0,     0,     0],
        [-nu12/E1, 1/E2,     -nu32/E3, 0,     0,     0],
        [-nu13/E1, -nu23/E2, 1/E3,     0,     0,     0],
        [0,         0,         0,        1/G23, 0,     0],
        [0,         0,         0,        0,     1/G31, 0],
        [0,         0,         0,        0,     0,     1/G12]
    ])

    return S


def constitutive_matrix_2D(E1, E2, G12, nu12):

    nu21 = nu12 * E2 / E1

    S = np.array([
        [1/E1,     -nu21/E2, 0],
        [-nu12/E1, 1/E2,     0],
        [0,         0,         1/G12]
    ])

    return S


def inverted_constitutive_matrix(S):

    Q = np.linalg.inv(S)

    return Q


# Input
Em = 4
num = 0.30

Ef = 76
nuf = 0.20

vf = 0.55

xi1 = 1
xi2 = 1


# Stiffness parameters
E1, E2, G12, nu12, nu23 = stiffness_parameters(
    vf, Ef, Em, num, nuf, xi1, xi2
)

# Constitutive matrices
S_3D = constitutive_matrix_3D(E1, E2, G12, nu12, nu23)

S_2D = constitutive_matrix_2D(E1, E2, G12, nu12)

Q_2D = inverted_constitutive_matrix(S_2D)


print("E1 =", E1, "GPa")
print("E2 =", E2, "GPa")
print("G12 =", G12, "GPa")
print("nu12 =", nu12)
print("nu23 =", nu23)

print("\nS 3D:")
print(S_3D)

print("\nS 2D:")
print(S_2D)

print("\nQ 2D:")
print(Q_2D)