import numpy as np
from typing import List, Optional

def Cmat(layup, MatProp):
# Calculate local stiffness/ constitutive Tensor 
#
# Input:
#   mat - Materiale properties for each type of material
#       mat = [E1 E2 nu12 G12;E1 E2 nu12 G12;...]
#   layup - Layer angle, thickness and material   
#       layup = [theta1 t1 mat1; theta2 t2 mat2; ...]
# Output:
#   Constitutive matrix in element coordinates (6x6)

    raise ValueError("Error Cmat has not been implemented")

    return C

def evaluate_stresses(epsilon0, kappa0, layup, mat_prop):
#   Function to evaluate element stresses and strains
#
#   Input:
#       epsilon0:   Element strain                   (3x1)
#       kappa0:     Element Curvature                (3x1)
#       layup:      Layup matrix                     (n layer x 3)
#       matProp:    Material properties              (n materials x 9)
#       
#   Output:
#       sigma:      Stress at each layer             ( - )
#       epsilon:    Strain at each layer             ( - )

    raise ValueError("Error evaluate_stresses has not been implemented")

    return sigma_local, epsilon_local

def EvaluateFailureCriteria(criteria, nLayerMax,matProp, layup, sigma,epsilon):
#   Function to evaluate failure criterias
#
#   Input:
#       criteria:   Failure Criteria to be evaluated (str)
#       nLayerMax:  Maximum number of layers         (int)
#       matProp:    Material properties              (n materials x 9)
#       layup:      Layup matrix                     (n layer x 3)
#       sigma:      Stress at each layer             (same type as in evaluateStresses)
#       epsilon:    Strain at each layer             (same type as in evaluateStresses)
#       
#   Output:
#       fval:       Matrix of failure criteria value
#                   evaluated at top and bottom      (2 * n layer)
    
    raise ValueError("Error EvaluateFailureCriteria has not been implemented")

    return fval