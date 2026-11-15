# %%
from networkx import display
import numpy as np
import sympy as sp



# %%




def solvesoe(Ai, Bi, Di, imax):
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for i in range(1, imax + 1):
        A += Ai
        B += Bi
        D += Di
    return A, B, D

imax = 3

Ai = np.array([
    [1/imax, 0,      0],
    [0,      1/imax, 0],
    [0,      0,      1/imax]
])

Bi = np.zeros((3, 3))

D1 = (13/12) * 1.27**3 * np.array([
    [147.87, 2.81, 0],
    [2.81,   9.05, 0],
    [0,      0,    3.3]
])

D2 = (1/12) * 1.27**3 * np.array([
    [9.05,   2.81, 0],
    [2.81, 147.87, 0],
    [0,      0,    3.3]
])

D3 = D1

Di = D1 + D2 + D3

display(Di)


solvesoe(Ai, Bi, Di, imax)

# %%
