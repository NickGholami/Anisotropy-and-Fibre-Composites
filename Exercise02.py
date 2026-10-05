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

# %% Exercise 1
# Exercise 1


# Input — moduli in GPa

Em = 4
num = 0.30

Ef = 76
nuf = 0.20

vf = 0.55

xi1 = 1
xi2 = 1


# Stiffness parameters

E1, E2, E2_ht, G12_reuss, G12_ht, nu12, nu23_reuss, nu23_ht = stiffness_parameters(
    vf, Ef, Em, num, nuf, xi1, xi2
)

# Constitutive matrices — Reuss E2 and G12

S_3D_reuss = constitutive_matrix_3D(
    E1, E2, G12_reuss, nu12, nu23_reuss
)

S_2D_reuss = constitutive_matrix_2D(E1, E2, G12_reuss, nu12)

Q_2D_reuss = inverted_constitutive_matrix(S_2D_reuss)


# Constitutive matrices — Halpin-Tsai E2 and G12

S_3D_ht = constitutive_matrix_3D(
    E1, E2_ht, G12_ht, nu12, nu23_ht
)

S_2D_ht = constitutive_matrix_2D(E1, E2_ht, G12_ht, nu12)

Q_2D_ht = inverted_constitutive_matrix(S_2D_ht)


# Display results

np.set_printoptions(precision=6, suppress=True)

print("E1 =", E1, "GPa")
print("E2 (Reuss) =", E2, "GPa")
print("E2 (Halpin-Tsai) =", E2_ht, "GPa")
print("G12 (Reuss) =", G12_reuss, "GPa")
print("G12 (Halpin-Tsai) =", G12_ht, "GPa")
print("nu12 =", nu12)
print("nu23 (Reuss) =", nu23_reuss)
print("nu23 (Halpin-Tsai) =", nu23_ht)

print("\nS 3D — Reuss E2 and G12 (1/GPa):")
print(S_3D_reuss)

print("\nS 3D — Halpin-Tsai E2 and G12 (1/GPa):")
print(S_3D_ht)

print("\nS 2D — Reuss E2 and G12 (1/GPa):")
print(S_2D_reuss)

print("\nS 2D — Halpin-Tsai E2 and G12 (1/GPa):")
print(S_2D_ht)

print("\nQ 2D — Reuss E2 and G12 (GPa):")
print(Q_2D_reuss)

print("\nQ 2D — Halpin-Tsai E2 and G12 (GPa):")
print(Q_2D_ht)

# %% Example 3.1 and 3.2

# Example 3.1 
E1 = 181 # GPa
E2 = 10.3  # GPa
nu12 = 0.28
G12 = 7.17  # GPa


S = constitutive_matrix_2D(E1, E2, G12, nu12)
Q = inverted_constitutive_matrix(S)

print("\nCompliance matrix S (GPa⁻¹):")
print(S)

print("\nStiffness matrix Q (GPa):")
print(Q)

# Example 3.2 with 30 degrees
theta = 30

Q_global = global_stiffness_matrix(Q, theta)
S_global = inverted_constitutive_matrix(Q_global)

print(f"\nGlobal stiffness matrix Q — θ = {theta}° (GPa):")
print(Q_global)

print(f"\nGlobal compliance matrix S — θ = {theta}° (GPa⁻¹):")
print(S_global)

# %% Exercise 2 - Q as function of angle

E1 = 54 # GPa
E2 = 18  # GPa
nu12 = 0.25
G12 = 9  # GPa

S = constitutive_matrix_2D(E1, E2, G12, nu12)
Q = inverted_constitutive_matrix(S)

angles = np.linspace(-90, 90, 181)

Q11 = []
Q22 = []
Q12 = []
Q16 = []
Q26 = []
Q66 = []

for theta in angles:
    Q_global = global_stiffness_matrix(Q, theta)

    Q11.append(Q_global[0, 0])
    Q22.append(Q_global[1, 1])
    Q12.append(Q_global[0, 1])
    Q16.append(Q_global[0, 2])
    Q26.append(Q_global[1, 2])
    Q66.append(Q_global[2, 2])

# Plot Q11, Q22 and Q12
plt.figure()

plt.plot(angles, Q11, label="Q11")
plt.plot(angles, Q22, label="Q22")
plt.plot(angles, Q12, label="Q12")

plt.xlabel("Angle θ [degrees]")
plt.ylabel("Stiffness [GPa]")
plt.legend()
plt.grid()

plt.show()

# Plot Q16, Q26 and Q66
plt.figure()

plt.plot(angles, Q16, label="Q16")
plt.plot(angles, Q26, label="Q26")
plt.plot(angles, Q66, label="Q66")

plt.xlabel("Angle θ [degrees]")
plt.ylabel("Stiffness [GPa]")
plt.legend()
plt.grid()

plt.show()

# %% Exercise 2 - Stress cases 1:
E1 = 54 # GPa
E2 = 18  # GPa
nu12 = 0.25
G12 = 9  # GPa

S = constitutive_matrix_2D(E1, E2, G12, nu12)
Q = inverted_constitutive_matrix(S)

# sigma = np.array([0.1, 0.0, 0.0])  # GPa - Stress case 1
# sigma = np.array([0.1, 0.05, 0.0])  # GPa - Stress case 2
sigma = np.array([0.002, 0.001, 0.0005])  # GPa - Stress case 3

epsilon_x = []
epsilon_y = []
gamma_xy = []

for theta in angles:

    # Global stiffness matrix at this angle
    Q_global = global_stiffness_matrix(Q, theta)

    # Global compliance matrix
    S_global = np.linalg.inv(Q_global)

    # Global strains
    epsilon = S_global @ sigma

    epsilon_x.append(epsilon[0])
    epsilon_y.append(epsilon[1])
    gamma_xy.append(epsilon[2])

plt.figure()

plt.plot(angles, epsilon_x, label="epsilon_x")
plt.plot(angles, epsilon_y, label="epsilon_y")
plt.plot(angles, gamma_xy, label="gamma_xy")

plt.xlabel("Angle θ [degrees]")
plt.ylabel("Strain")
plt.legend()
plt.grid()

plt.show()

# %% Local stresses and strains

theta = 30  # degrees

sigma_global = np.array([0.002, 0.001, 0.0005])  # GPa

T = transformation_matrix(theta)

sigma_local = np.linalg.inv(T) @ sigma_global

print("Global stress:")
print(sigma_global)

print("\nLocal stress:")
print(sigma_local)

epsilon_local = S @ sigma_local

print("\nLocal strain:")
print(epsilon_local)

print("epsilon_1 =", epsilon_local[0])
print("epsilon_2 =", epsilon_local[1])
print("gamma_12  =", epsilon_local[2])
