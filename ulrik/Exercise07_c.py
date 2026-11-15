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


# %% Input - choose one panel, lay-up and load area
E1 = 181000  # MPa
E2 = 10300  # MPa
G12 = 7170  # MPa
nu12 = 0.28
angles = np.array([0, 0, 0, 0, 0, 0, 0, 0])
# angles = np.array([90, 90, 90, 90, 90, 90, 90, 90])
# angles = np.array([0, 90, 0, 90, 90, 0, 90, 0])
# angles = np.array([45, -45, 45, -45, -45, 45, -45, 45])
thicknesses = np.full(len(angles), 0.5)  # mm

a = 1000.0  # mm; change to 500 for the rectangular panel
b = 1000.0  # mm
max_term = 31  # Highest odd Fourier index; increase to check convergence

# %% Bending stiffness
S_local = constitutive_matrix_2D(E1, E2, G12, nu12)
Q_local = inverted_constitutive_matrix(S_local)
z = np.concatenate(([0.0], np.cumsum(thicknesses))) - np.sum(thicknesses) / 2
D = np.zeros((3, 3))

for i, angle in enumerate(angles):
    Q_global = global_stiffness_matrix(Q_local, angle)
    D += Q_global * (z[i + 1]**3 - z[i]**3) / 3


# %% Buckling
m_values = np.arange(1, 6)  # m = 1, 2, 3, 4, 5
buckling_loads = np.zeros(len(m_values))
n = 1

for i, m in enumerate(m_values):
    kx = m * np.pi / a
    ky = n * np.pi / b

    # Calculate and store the buckling load here
    buckling_loads[i] = 


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
