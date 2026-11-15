# %%
import numpy as np

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

def failure_criteria(sigma, epsilon, strengths, E1, E2, G12, F12_star=-0.5):
    sigma_1, sigma_2, tau_12 = sigma
    epsilon_1, epsilon_2, gamma_12 = epsilon
    sigma_1t, sigma_1c, sigma_2t, sigma_2c, tau_strength = strengths

    # Maximum stress - choose strength according to stress sign
    strength_1 = sigma_1t if sigma_1 >= 0 else sigma_1c
    strength_2 = sigma_2t if sigma_2 >= 0 else sigma_2c
    max_stress = max(abs(sigma_1) / strength_1, abs(sigma_2) / strength_2,
                     abs(tau_12) / tau_strength)

    # Maximum strain - choose limit according to strain sign
    strain_1 = sigma_1t / E1 if epsilon_1 >= 0 else sigma_1c / E1
    strain_2 = sigma_2t / E2 if epsilon_2 >= 0 else sigma_2c / E2
    max_strain = max(abs(epsilon_1) / strain_1, abs(epsilon_2) / strain_2,
                     abs(gamma_12) / (tau_strength / G12))

    # Tsai-Hill with sign-dependent strengths
    tsai_hill = ((sigma_1**2 - sigma_1 * sigma_2) / strength_1**2
                 + (sigma_2 / strength_2)**2 + (tau_12 / tau_strength)**2)

    # Tsai-Wu and Hoffman coefficients
    F1 = 1 / sigma_1t - 1 / sigma_1c
    F2 = 1 / sigma_2t - 1 / sigma_2c
    F11 = 1 / (sigma_1t * sigma_1c)
    F22 = 1 / (sigma_2t * sigma_2c)
    F66 = 1 / tau_strength**2
    F12 = F12_star * np.sqrt(F11 * F22)

    linear = F1 * sigma_1 + F2 * sigma_2
    quadratic_wu = F11 * sigma_1**2 + F22 * sigma_2**2 + 2 * F12 * sigma_1 * sigma_2 + F66 * tau_12**2
    quadratic_hoffman = F11 * (sigma_1**2 - sigma_1 * sigma_2) + F22 * sigma_2**2 + F66 * tau_12**2

    # Under proportional loading, each index is a*x**2 + b*x.
    a = np.array([0, 0, tsai_hill, quadratic_wu, quadratic_hoffman])
    b = np.array([max_stress, max_strain, 0, linear, linear])
    return a, b


# %% Input - same laminate as Exercise 3
E1 = 46000  # MPa
E2 = 13000  # MPa
G12 = 4400  # MPa
nu12 = 0.30

angles = np.array([0, 90, 90, 0])  # Top to bottom
thicknesses = np.array([0.25, 0.25, 0.25, 0.25])  # mm
strengths = np.array([1000, 1000, 35, 150, 50])  # sigma_1t, sigma_1c, sigma_2t, sigma_2c, tau_12 [MPa]
F12_star = -0.5
criteria = ["Max stress", "Max strain", "Tsai-Hill", "Tsai-Wu", "Hoffman"]
positions = ["top", "middle", "bottom"]

# %% A, B and D
S_local = constitutive_matrix_2D(E1, E2, G12, nu12)
Q_local = inverted_constitutive_matrix(S_local)
z = np.concatenate(([0.0], np.cumsum(thicknesses))) - np.sum(thicknesses) / 2

A = np.zeros((3, 3))
B = np.zeros((3, 3))
D = np.zeros((3, 3))

for i, angle in enumerate(angles):
    Q_global = global_stiffness_matrix(Q_local, angle)
    z_top = z[i]
    z_bottom = z[i + 1]
    A += Q_global * (z_bottom - z_top)
    B += Q_global * (z_bottom**2 - z_top**2) / 2
    D += Q_global * (z_bottom**3 - z_top**3) / 3

# %% Loads - choose case 4a or 4b
N = np.array([300.0, -50.0, 25.0])  # N/mm
M = np.array([50.0, 0.0, 0.0])      # N - Case 4a
# M = np.array([0.0, 0.0, 5.0])     # N - Case 4b
x = 1.0  # Positive load multiplier; use 1 for the specified case 4b

ABD = np.block([[A, B], [B, D]])
strains_and_curvatures = np.linalg.solve(ABD, np.concatenate((N, M)))
epsilon_0 = strains_and_curvatures[:3]
kappa = strains_and_curvatures[3:]

# %% Local stresses, strains and failure at top, middle and bottom
local_stresses = np.zeros((len(angles), 3, 3))
local_strains = np.zeros((len(angles), 3, 3))
indices = np.zeros((len(angles), 3, 5))  # Ply, position, criterion
load_factors = np.full((len(angles), 3, 5), np.inf)

for i, angle in enumerate(angles):
    T = transformation_matrix(angle)
    z_positions = [z[i], (z[i] + z[i + 1]) / 2, z[i + 1]]

    for face, z_face in enumerate(z_positions):
        # Reference response at x = 1; multiply by x for the applied load
        epsilon_local = T.T @ (epsilon_0 + z_face * kappa)
        sigma_local = Q_local @ epsilon_local
        local_strains[i, face] = x * epsilon_local
        local_stresses[i, face] = x * sigma_local

        a, b = failure_criteria(sigma_local, epsilon_local, strengths, E1, E2, G12, F12_star)
        indices[i, face] = a * x**2 + b * x

        # Failure load: solve a*x**2 + b*x - 1 = 0 for each criterion
        for j in range(5):
            roots = np.roots([a[j], b[j], -1])
            positive_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
            if positive_roots:
                load_factors[i, face, j] = min(positive_roots)

# %% Results
failure_01 = (indices >= 1 - 1e-10).astype(int)  # 1 = failure; tolerance for round-off
ply_failure = np.max(failure_01, axis=1)  # Failure at any of the three positions

print(f"Applied loads: N = {x * N} N/mm, M = {x * M} N")
print("\nLocal stresses [MPa] and strains: rows = top, middle, bottom")
for i in range(len(angles)):
    print(f"\nPly {i + 1} stresses:\n{np.round(local_stresses[i], 4)}")
    print(f"Ply {i + 1} strains:\n{np.round(local_strains[i], 6)}")

for j, criterion in enumerate(criteria):
    print(f"\n{criterion} - 0/1 failure vectors:")
    for face, name in enumerate(positions):
        print(f"{name:>6}: {failure_01[:, face, j]}")
    print(f"Per ply: {ply_failure[:, j]}")
    print(f"Safe according to this criterion: {not np.any(ply_failure[:, j])}")
    print(f"Failure x at ply midpoints: {np.round(load_factors[:, 1, j], 4)}")
    print(f"First failure x per ply: {np.round(np.min(load_factors[:, :, j], axis=1), 4)}")
    print(f"First-ply failure x: {np.min(load_factors[:, :, j]):.4f}")
