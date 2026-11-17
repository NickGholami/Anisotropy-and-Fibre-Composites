"""Laminated plate theory - load cases, buckling, vibration."""

import numpy as np

from . import laminate

# Functions for this topic go below.


def simply_supported_plate(D, a, b, q, alpha, beta, x, y, n_terms):
    # Simply supported, specially orthotropic plate (D16 = D26 are ignored)
    # Load q on the patch alpha[0]*a < x < alpha[1]*a and beta[0]*b < y < beta[1]*b
    # Whole plate: alpha = beta = [0, 1]
    D11 = D[0, 0]
    D12 = D[0, 1]
    D22 = D[1, 1]
    D66 = D[2, 2]

    w = 0.0
    Mx = 0.0
    My = 0.0
    Mxy = 0.0

    for n in range(1, n_terms + 1):
        for m in range(1, n_terms + 1):
            # Fourier coefficient of the load (Lecture 7, Fourier coefficients for other loads)
            q_mn = (4 * q / (np.pi**2 * m * n)
                    * (np.cos(m * np.pi * alpha[1]) - np.cos(m * np.pi * alpha[0]))
                    * (np.cos(n * np.pi * beta[1]) - np.cos(n * np.pi * beta[0])))

            # General load (Lecture 7, Bending of simply supported specially orthotropic plate - 2)
            denominator = (D11 * (m * np.pi / a)**4
                           + 2 * (D12 + 2 * D66) * (m * np.pi / a)**2 * (n * np.pi / b)**2
                           + D22 * (n * np.pi / b)**4)
            sin_sin = np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / b)
            cos_cos = np.cos(m * np.pi * x / a) * np.cos(n * np.pi * y / b)

            # Deflection
            w += q_mn * sin_sin / denominator

            # Moments
            Mx += q_mn * (D11 * (m * np.pi / a)**2 + D12 * (n * np.pi / b)**2) * sin_sin / denominator
            My += q_mn * (D12 * (m * np.pi / a)**2 + D22 * (n * np.pi / b)**2) * sin_sin / denominator
            Mxy += -2 * D66 * q_mn * (m * np.pi / a) * (n * np.pi / b) * cos_cos / denominator

    return w, Mx, My, Mxy

# ---------- Week 8: buckling and free vibration ----------


def buckling_load(D, a, b, m):
    # Critical edge load P [N/mm] for compression in x (Nx = -P, Ny = Nxy = 0)
    # Simply supported, specially orthotropic plate (D16 and D26 are ignored)
    # m = number of half-waves in x. The lowest load always has n = 1.
    # Lecture 8, p. 10-11 (Buckling of simply supported, specially orthotropic plate - 1 and 2)
    D11 = D[0, 0]
    D12 = D[0, 1]
    D22 = D[1, 1]
    D66 = D[2, 2]

    P = np.pi**2 * (D11 * (m / a)**2
                    + 2 * (D12 + 2 * D66) * (1 / b)**2
                    + D22 * (a / m)**2 * (1 / b)**4)
    return P


def natural_frequency(D, a, b, rho_star, m, n):
    # Natural frequency omega_mn [rad/s] of a simply supported, specially orthotropic plate
    # rho_star = mass per area = rho * h  [N s^2 / mm^3 when D is in N mm and a, b in mm]
    # m, n = number of half-waves in x and y
    # Lecture 8, p. 13 (Free vibration of simply supported, specially orthotropic plate)
    D11 = D[0, 0]
    D12 = D[0, 1]
    D22 = D[1, 1]
    D66 = D[2, 2]

    rho_omega2 = (D11 * (m * np.pi / a)**4
                  + 2 * (D12 + 2 * D66) * (m * np.pi / a)**2 * (n * np.pi / b)**2
                  + D22 * (n * np.pi / b)**4)
    omega = np.sqrt(rho_omega2 / rho_star)
    return omega


def lowest_frequencies(D, a, b, rho_star, n_modes, mn_max):
    # The n_modes lowest natural frequencies and their m and n
    # Calculates omega_mn for all m, n = 1 ... mn_max and sorts them
    # (Lecture 8, p. 13: "the combination of m and n giving the lowest omega_mn")
    omega_list = []
    m_list = []
    n_list = []
    for m in range(1, mn_max + 1):
        for n in range(1, mn_max + 1):
            omega_list.append(natural_frequency(D, a, b, rho_star, m, n))
            m_list.append(m)
            n_list.append(n)

    # Index order from lowest to highest omega
    order = np.argsort(omega_list)

    omega = np.zeros(n_modes)
    m_mode = np.zeros(n_modes, dtype=int)
    n_mode = np.zeros(n_modes, dtype=int)
    for mode in range(n_modes):
        omega[mode] = omega_list[order[mode]]
        m_mode[mode] = m_list[order[mode]]
        n_mode[mode] = n_list[order[mode]]

    return omega, m_mode, n_mode