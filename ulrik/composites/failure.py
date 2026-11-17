"""Ply failure criteria, first-ply failure (FPF), progressive failure to
last-ply failure (LPF), delamination."""

import numpy as np

from . import ply, laminate

# Functions for this topic go below.

CRITERIA = ["Max stress", "Max strain", "Tsai-Hill", "Tsai-Wu", "Hoffman"]


def failure_criteria(sigma, epsilon, strengths, E1, E2, G12, F12_star=-0.5):
    # Returns the failure index of each criterion. Failure when index >= 1.
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

    # Tsai-Wu
    tsai_wu = (F11 * sigma_1**2 + F22 * sigma_2**2
               + 2 * F12 * sigma_1 * sigma_2 + F66 * tau_12**2
               + F1 * sigma_1 + F2 * sigma_2)

    # Hoffman (Tsai-Wu with 2*F12 = -F11)
    hoffman = (F11 * (sigma_1**2 - sigma_1 * sigma_2)
               + F22 * sigma_2**2 + F66 * tau_12**2
               + F1 * sigma_1 + F2 * sigma_2)

    return np.array([max_stress, max_strain, tsai_hill, tsai_wu, hoffman])
