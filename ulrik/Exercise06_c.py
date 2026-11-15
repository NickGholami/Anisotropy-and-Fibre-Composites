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

def plate_response(D, a, b, q0, x, y, max_term, load_area):
    # Orthotropic Navier approximation: D16 and D26 are neglected.
    D_plate = D.copy()
    D_plate[0, 2] = D_plate[2, 0] = 0
    D_plate[1, 2] = D_plate[2, 1] = 0
    alpha1, alpha2, beta1, beta2 = load_area  # Loaded bounds as fractions of a, b
    w = np.zeros_like(x, dtype=float)
    kappa = np.zeros((3, len(x)))

    # Full and central-quarter loading are symmetric: only odd terms contribute.
    for m in range(1, max_term + 1, 2):
        for n in range(1, max_term + 1, 2):
            kx = m * np.pi / a
            ky = n * np.pi / b
            q_mn = (4 * q0 / (m * n * np.pi**2)
                    * (np.cos(m * np.pi * alpha1) - np.cos(m * np.pi * alpha2))
                    * (np.cos(n * np.pi * beta1) - np.cos(n * np.pi * beta2)))
            W_mn = q_mn / (D_plate[0, 0] * kx**4
                           + 2 * (D_plate[0, 1] + 2 * D_plate[2, 2]) * kx**2 * ky**2
                           + D_plate[1, 1] * ky**4)
            sine = np.sin(kx * x) * np.sin(ky * y)
            w += W_mn * sine
            kappa[0] += kx**2 * W_mn * sine
            kappa[1] += ky**2 * W_mn * sine
            kappa[2] -= 2 * kx * ky * W_mn * np.cos(kx * x) * np.cos(ky * y)

    moments = D_plate @ kappa
    return w, moments


# %% Input - choose one panel, lay-up and load area
E1 = 181000  # MPa
E2 = 10300  # MPa
G12 = 7170  # MPa
nu12 = 0.28
half_angles = np.array([0, 0, 0, 0])
# half_angles = np.array([90, 90, 90, 90])
# half_angles = np.array([0, 90, 0, 90])
# half_angles = np.array([45, -45, 45, -45])
angles = np.concatenate((half_angles, half_angles[::-1]))  # Symmetric lay-up
thicknesses = np.full(len(angles), 0.5)  # mm

a = 1000.0  # mm; change to 500 for the rectangular panel
b = 1000.0  # mm
q0 = 0.001  # MPa = 1 kPa
load_area = (0, 1, 0, 1)  # Full plate
# load_area = (0.25, 0.75, 0.25, 0.75)  # Central quarter of the area, same pressure
max_term = 31  # Highest odd Fourier index; increase to check convergence

# %% Bending stiffness
S_local = constitutive_matrix_2D(E1, E2, G12, nu12)
Q_local = inverted_constitutive_matrix(S_local)
z = np.concatenate(([0.0], np.cumsum(thicknesses))) - np.sum(thicknesses) / 2
D = np.zeros((3, 3))

for i, angle in enumerate(angles):
    Q_global = global_stiffness_matrix(Q_local, angle)
    D += Q_global * (z[i + 1]**3 - z[i]**3) / 3

# %% Centre deflection and moments; twisting moments at the four corners
x = np.array([a / 2, 0, a, 0, a])
y = np.array([b / 2, 0, 0, b, b])
w, moments = plate_response(D, a, b, q0, x, y, max_term, load_area)

print(f"Lay-up: {angles}")
print(f"D [N mm]:\n{np.round(D, 2)}")
if not np.allclose(D[:2, 2], 0, atol=1e-6):
    print("Approximation: D16 and D26 are neglected in deflection and moments.")
print(f"Centre deflection [mm]: {w[0]:.4f}")
print(f"Centre Mx, My [N]: {np.round(moments[:2, 0], 4)}")
print(f"Corner Mxy [N], order (0,0), (a,0), (0,b), (a,b):\n{np.round(moments[2, 1:], 4)}")
