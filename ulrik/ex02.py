"""Exercise 02 - Transformation and Rotation"""

#%% SETUP
import numpy as np
import matplotlib.pyplot as plt
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
# Example 3.2
E1_ex3 = 181    # [GPa] Longitudinal Young's modulus
E2_ex3 = 10.3   # [GPa] Transverse Young's modulus
nu12_ex3 = 0.28 # [-]   Poisson's ratio
G12_ex3 = 7.17  # [GPa] In-plane shear modulus
theta_ex3 = 30  # [deg] Rotation angle

# Input data for Q as function of angle
E1 = 54          # [GPa] Longitudinal Young's modulus
E2 = 18          # [GPa] Transverse Young's modulus
nu12 = 0.25      # [-]   Poisson's ratio
G12 = 9          # [GPa] In-plane shear modulus
angles = np.linspace(-90, 90, 181)  # [deg] Ply angles theta from -90° to 90° in 1° steps

# Global stiffness components [GPa]
Q11 = []
Q22 = []
Q12 = []
Q16 = []
Q26 = []
Q66 = []

# Global strains [-]
epsilon_x = []
epsilon_y = []
gamma_xy = []

# Uncomment the stress case to run
# sigma = np.array([0.1, 0.0, 0.0])         # [GPa] Stress case 1: uniaxial 100 MPa
# sigma = np.array([0.1, 0.05, 0.0])        # [GPa] Stress case 2: biaxial 100 MPa and 50 MPa
sigma = np.array([0.002, 0.001, 0.0005])    # [GPa] Stress case 3: [2, 1, 0.5] MPa

# Local stresses and strains
theta_loc = 90   # [deg] Ply angle used to find local stresses and strains

#%% SOLUTION
# ---------- SOLUTION ----------

# Example 3.2 calculating 2D compliance & stiffness matrix
S_ex3 = ply.constitutive_matrix_2D(E1_ex3, E2_ex3, G12_ex3, nu12_ex3)
Q_ex3 = np.linalg.inv(S_ex3)
# With theta = 30 degrees
T_ex3 = ply.transformation_matrix(theta_ex3)
Q_gl_ex3 = T_ex3 @ Q_ex3 @ T_ex3.transpose()
S_gl_ex3 = np.linalg.inv(Q_gl_ex3)


#Exercise - Plot Q as function of angle
S = ply.constitutive_matrix_2D(E1, E2, G12, nu12)
Q = np.linalg.inv(S)

for theta in angles:
    T = ply.transformation_matrix(theta)
    Q_global = T @ Q @ T.transpose()

    Q11.append(Q_global[0, 0])
    Q22.append(Q_global[1, 1])
    Q12.append(Q_global[0, 1])
    Q16.append(Q_global[0, 2])
    Q26.append(Q_global[1, 2])
    Q66.append(Q_global[2, 2])

    # Global compliance matrix
    S_global = np.linalg.inv(Q_global)

    # Global strains
    epsilon = S_global @ sigma
    epsilon_x.append(epsilon[0])
    epsilon_y.append(epsilon[1])
    gamma_xy.append(epsilon[2])


# Local stresses and strains at theta = theta_loc
T_loc = ply.transformation_matrix(theta_loc)
# T maps local to global stress, so invert it to go back
sigma_local = np.linalg.inv(T_loc) @ sigma
# Local strains from the local (material) compliance matrix S
epsilon_local = S @ sigma_local

#%% OUTPUT
# ---------- OUTPUT ----------

# Example 3.2
print("Example 3.2")
print(f"\nCompliance matrix S (GPa⁻¹): \n {S_ex3}")
print(f"\nStiffness matrix Q (GPa): \n {Q_ex3} ")

# Transformation of constitutive matrices to global - Example 3.2
print("\nTransformation of constitutive matrices to global - Example 3.2")
print(f"\nGlobal stiffness matrix Q — theta = {theta_ex3}° (GPa):")
print(Q_gl_ex3)
print(f"\nGlobal compliance matrix S — theta = {theta_ex3}° (GPa⁻¹):")
print(S_gl_ex3)


# Exercise -  Global stiffness components vs. angle theta
# Figure 1: normal stiffnesses (Q11, Q22) and Poisson coupling (Q12).
plt.figure()
plt.plot(angles, Q11, label="Q11")
plt.plot(angles, Q22, label="Q22")
plt.plot(angles, Q12, label="Q12")
plt.title("Normal stiffnesses and Poisson coupling vs. angle theta")
plt.xlabel("Angle theta [degrees]")
plt.ylabel("Stiffness [GPa]")
plt.legend()
plt.grid()
plt.show()

# Figure 2: shear-extension coupling (Q16, Q26) and shear stiffness (Q66).
plt.figure()
plt.plot(angles, Q16, label="Q16")
plt.plot(angles, Q26, label="Q26")
plt.plot(angles, Q66, label="Q66")
plt.title("Shear-extension coupling and shear stiffness vs. angle theta")
plt.xlabel("Angle theta [degrees]")
plt.ylabel("Stiffness [GPa]")
plt.legend()
plt.grid()
plt.show()


# Figure 3: global strains from the applied stress vs. angle theta.
plt.figure()
plt.plot(angles, epsilon_x, label="epsilon_x")
plt.plot(angles, epsilon_y, label="epsilon_y")
plt.plot(angles, gamma_xy, label="gamma_xy")
plt.title("Global strains vs. angle theta")
plt.xlabel("Angle theta [degrees]")
plt.ylabel("Strain")
plt.legend()
plt.grid()
plt.show()


# Local stresses and strains
print(f"\nLocal stresses and strains — theta = {theta_loc}°")
print(f"\nGlobal stress [sigma_x, sigma_y, tau_xy] (GPa): \n {sigma}")
print(f"\nLocal stress [sigma_1, sigma_2, tau_12] (GPa): \n {sigma_local}")
print("\nLocal strain:")
print("epsilon_1 =", epsilon_local[0])
print("epsilon_2 =", epsilon_local[1])
print("gamma_12  =", epsilon_local[2])