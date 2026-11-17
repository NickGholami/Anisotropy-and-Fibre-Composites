"""Stacking, z-coordinates, A/B/D matrices, laminate strains and stresses."""

import numpy as np
from . import ply

# Functions for this topic go below.

def z_coordinates(thicknesses):
    # z = 0 at the mid-plane
    boundaries = np.concatenate(([0.0], np.cumsum(thicknesses)))
    z = boundaries - np.sum(thicknesses) / 2
    return z


def abd_matrices(Q_local, angles, z):
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))

    for i in range(len(angles)):
        T = ply.transformation_matrix(angles[i])
        Q_global = T @ Q_local @ T.T
        z_top = z[i]
        z_bottom = z[i + 1]

        A += Q_global * (z_bottom - z_top)
        B += Q_global * (z_bottom**2 - z_top**2) / 2
        D += Q_global * (z_bottom**3 - z_top**3) / 3

    return A, B, D


def local_ply_stresses(Q_local, angles, z, epsilon_0, kappa):

    # stress[ply, face, stress component]: 
    # ply: counting from the top
    # face: top = 0, bottom = 1
    # stress component: sigma_1 = 0, sigma_2 = 1, tau_12 = 2
    stress = np.zeros((len(angles), 2, 3))

    for i in range(len(angles)):
        angle = angles[i]
        T = ply.transformation_matrix(angle)
        Q_global = T @ Q_local @ T.T

        z_top = z[i]
        z_bottom = z[i + 1]

        # Top face
        epsilon_top = epsilon_0 + z_top * kappa
        sigma_top = Q_global @ epsilon_top
        stress[i, 0] = np.linalg.solve(T, sigma_top)

        # Bottom face
        epsilon_bottom = epsilon_0 + z_bottom * kappa
        sigma_bottom = Q_global @ epsilon_bottom
        stress[i, 1] = np.linalg.solve(T, sigma_bottom)

    return stress


def local_ply_response(Q_local, angles, z, epsilon_0, kappa):
    # Local strains and stresses at top, middle and bottom of each ply
    # strain/stress[ply, position, component]:
    # ply: counting from the top
    # position: top = 0, middle = 1, bottom = 2
    # component: 1 = 0, 2 = 1, 12 = 2
    strain = np.zeros((len(angles), 3, 3))
    stress = np.zeros((len(angles), 3, 3))

    for i in range(len(angles)):
        T = ply.transformation_matrix(angles[i])
        z_positions = [z[i], (z[i] + z[i + 1]) / 2, z[i + 1]]

        for k, z_k in enumerate(z_positions):
            epsilon_global = epsilon_0 + z_k * kappa
            # Global -> local engineering strains
            strain[i, k] = T.T @ epsilon_global
            stress[i, k] = Q_local @ strain[i, k]

    return strain, stress
