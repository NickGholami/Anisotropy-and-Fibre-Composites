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


# %% Input - Exercise 5A (same laminate and loads as 4a)
E1 = 46000  # MPa
E2 = 13000  # MPa
G12 = 4400  # MPa
nu12 = 0.30
angles = np.array([0, 90, 90, 0])  # Top to bottom
thicknesses = np.array([0.25, 0.25, 0.25, 0.25])  # mm
strengths = np.array([1000, 1000, 35, 150, 50])  # 1t, 1c, 2t, 2c, shear [MPa]
N = np.array([300.0, -50.0, 25.0])  # N/mm, multiplied by x
M = np.array([50.0, 0.0, 0.0])     # N, multiplied by x
F12_star = -0.5
step = 0.001  # Load multiplier increment
x_max = 2.0   # Stop here if not all plies have failed
criteria = ["Max stress", "Max strain", "Tsai-Hill", "Tsai-Wu", "Hoffman"]

# %% Local stiffness and ply positions
S_local = constitutive_matrix_2D(E1, E2, G12, nu12)
Q_local = inverted_constitutive_matrix(S_local)
z = np.concatenate(([0.0], np.cumsum(thicknesses))) - np.sum(thicknesses) / 2
load_increment = step * np.concatenate((N, M))
failure_loads = np.full((len(angles), 5), np.nan)  # Ply, criterion

# %% Progressive failure - separate analysis for each criterion
for j, criterion in enumerate(criteria):
    failed = np.zeros(len(angles), dtype=bool)
    local_strains = np.zeros((len(angles), 2, 3))  # Ply, top/bottom, component
    local_stresses = np.zeros_like(local_strains)
    print(f"\n{criterion}:")

    for increment in range(1, int(round(x_max / step)) + 1):
        x = increment * step
        A, B, D = np.zeros((3, 3, 3))

        for i, angle in enumerate(angles):
            if failed[i]:
                continue  # Ply discount: no stiffness from failed plies
            Q_global = global_stiffness_matrix(Q_local, angle)
            A += Q_global * (z[i + 1] - z[i])
            B += Q_global * (z[i + 1]**2 - z[i]**2) / 2
            D += Q_global * (z[i + 1]**3 - z[i]**3) / 3

        ABD = np.block([[A, B], [B, D]])
        delta = np.linalg.solve(ABD, load_increment)
        delta_epsilon_0 = delta[:3]
        delta_kappa = delta[3:]

        for i, angle in enumerate(angles):
            if failed[i]:
                continue
            T = transformation_matrix(angle)
            for face, z_face in enumerate((z[i], z[i + 1])):
                delta_local = T.T @ (delta_epsilon_0 + z_face * delta_kappa)
                local_strains[i, face] += delta_local
                local_stresses[i, face] += Q_local @ delta_local
                a, b = failure_criteria(local_stresses[i, face], local_strains[i, face],
                                       strengths, E1, E2, G12, F12_star)
                if a[j] + b[j] >= 1:
                    failed[i] = True
            if failed[i]:
                failure_loads[i, j] = x
                print(f"Ply {i + 1} fails at x = {x:.3f}")

        # Lecture's incremental model: discount stiffness for the next increment.
        if np.all(failed):
            break  # Do not solve the zero-stiffness system after LPF

    values = failure_loads[:, j]
    if np.any(failed):
        print(f"FPF: x = {np.nanmin(values):.3f}")
    if np.all(failed):
        print(f"LPF: x = {np.max(values):.3f}")
    else:
        print("LPF not reached; increase x_max.")
