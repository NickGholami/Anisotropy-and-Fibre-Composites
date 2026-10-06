# %%
import numpy as np
import matplotlib.pyplot as plt

def stiffness_parameters(v_f, E_f, E_m, nu_m, nu_f, xi_1=1, xi_2=1):
    # Volume fraction of matrix
    v_m = 1 - v_f

    # E1 - Axial stiffness (equal strain - Voigt model)
    E1 = v_f * E_f + v_m * E_m

    # E2 - Transverse stiffness (equal stress - Reuss model)
    E2_reuss = 1 / (v_f / E_f + v_m / E_m)

    # E2_ht - Transverse stiffness (Halpin-Tsai model)
    eta_1 = (E_f / E_m - 1) / (E_f / E_m + xi_1)
    E2_ht = E_m * (1 + xi_1 * eta_1 * v_f) / (1 - eta_1 * v_f)

    # Shear modulus for fiber and matrix
    G_f = E_f / (2 * (1 + nu_f))
    G_m = E_m / (2 * (1 + nu_m))

    # G12 - Shear modulus (equal stress - Reuss model)
    G12_reuss = (v_f / G_f + v_m / G_m)**-1

    # G12_ht - Shear modulus (Halpin-Tsai model)
    eta_2 = (G_f / G_m - 1) / (G_f / G_m + xi_2)
    G12_ht = G_m * (1 + xi_2 * eta_2 * v_f) / (1 - eta_2 * v_f)

    # Poisson's ratio nu12
    nu12 = v_f * nu_f + v_m * nu_m

    # Bulk modulus
    K_f = E_f / (3 * (1 - 2 * nu_f))
    K_m = E_m / (3 * (1 - 2 * nu_m))
    K = (v_f / K_f + v_m / K_m)**-1

    # Poisson's ratios
    nu21_reuss = nu12 * E2_reuss / E1
    nu21_ht = nu12 * E2_ht / E1

    nu23_reuss = 1 - nu21_reuss - E2_reuss / (3 * K)
    nu23_ht = 1 - nu21_ht - E2_ht / (3 * K)

    return E1, E2_reuss, E2_ht, G12_reuss, G12_ht, nu12, nu23_reuss, nu23_ht

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
        [-nu12/E1,  1/E2,    -nu32/E3, 0,     0,     0],
        [-nu13/E1, -nu23/E2,  1/E3,    0,     0,     0],
        [0,         0,        0,       1/G23, 0,     0],
        [0,         0,        0,       0,     1/G31, 0],
        [0,         0,        0,       0,     0,     1/G12]
    ])

    return S

def constitutive_matrix_2D(E1, E2, G12, nu12):
    nu21 = nu12 * E2 / E1

    S = np.array([
        [1/E1,    -nu21/E2, 0],
        [-nu12/E1, 1/E2,    0],
        [0,        0,       1/G12]
    ])

    return S

def inverted_constitutive_matrix(S):
    return np.linalg.inv(S)

def transformation_matrix(theta):
    theta = np.deg2rad(theta)
    c = np.cos(theta)
    s = np.sin(theta)

    T = np.array([
        [c**2, s**2, -2 * s * c],
        [s**2, c**2, 2 * s * c],
        [s * c, -s * c, c**2 - s**2]
    ])
    return T

def global_stiffness_matrix(Q_local, theta):
    T = transformation_matrix(theta)
    Q_global = T @ Q_local @ T.T
    return Q_global

# %% Exercise 3 
E1 = 46000     # MPa
E2 = 13000     # MPa
G12 = 4400     # MPa
nu12 = 0.30

# Top to bottom
angles = np.array([0, 90, 90, 0])
thicknesses = np.array([0.25, 0.25, 0.25, 0.25])  # mm

# Local matrices
S_local = constitutive_matrix_2D(E1, E2, G12, nu12)
Q_local = inverted_constitutive_matrix(S_local)

total_thickness = np.sum(thicknesses)
bottom_positions = np.cumsum(thicknesses)
boundaries = np.concatenate(([0.0], bottom_positions))

# Zero to the mid-plane
# Positive z points downwards
z = boundaries - total_thickness / 2

print("Ply boundary positions [mm]:", z)

# Initialize A, B, D matrices
A = np.zeros((3, 3))
B = np.zeros((3, 3))
D = np.zeros((3, 3))

# For loop over each ply
for i in range(len(angles)):
    Q_global = global_stiffness_matrix(Q_local, angles[i])

    # Top and bottom positions
    z_top = z[i]
    z_bottom = z[i + 1]

    # Add this ply's contribution
    A += Q_global * (z_bottom - z_top)
    B += Q_global * (z_bottom**2 - z_top**2) / 2
    D += Q_global * (z_bottom**3 - z_top**3) / 3

print("A [N/mm]:")
print(A)

print("\nB [N]:")
print(B)

print("\nD [N·mm]:")
print(D)