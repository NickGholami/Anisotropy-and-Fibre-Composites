import numpy             as np
import scipy             as sp
import scipy.sparse      as sps
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import math
import os

# Changing to the directory of this file
os.chdir(os.path.dirname(__file__))
print ("Current working dir : %s" % os.getcwd())

from MESH import generate_mesh
from PLOT import plot_mesh, plotBC, plot_deformed,plot_failure_criteria
from TOOLS import select_nodes
from FEM import Bmat
from MATERIAL import Cmat,evaluate_stresses,EvaluateFailureCriteria

# Input Parameters
Lx = 1000.0   # Side length x [mm]
Ly = 100.0    # Side length y [mm]
nelx = 100    # Number of elements in x [-]
nely = 20     # Number of elements in y [-]
loadcase = 1  # Load case

# If you are color blind change the following values
colors = np.array([[1,0,0],[0,1,0],[0,0,1]])

# Material, Mesh & Layup definitions
# Material properties are defined in the same order as table 4.2:
# E1, E2, G12, nu12, sigma_1t, sigma_1c, etc.
matProp = np.array([[181e3, 10.3e3, 7.17e3, 0.28, 1100, 600, 20, 140, 70, 1e-3, 1e-2, 1e-3, 1e-2, 1-2]])

# Failure criteria to be evaluated
failureCriteria = "Max Stress"

# Generate Mesh
X, Xrot, IX, edofMat, nelem, nnode, ndof = generate_mesh(Lx, Ly, nelx, nely)

# Layup is defined as a list of numpy arrays
# Each array has columns: [fiber angle, thickness, material number]
layupCells = [np.array([[0, 1.25, 0],[math.pi/4, 1.25, 0],[math.pi/4, 1.25, 0],[0, 1.25, 0]]) for _ in range(nelem)]
nLayerMax = 4

# Local coordinate system for each element
# Each array has two rows: [local x-axis; local y-axis]
local_coor_cells = [np.array([[1, 0, 0], [0, 1, 0]]) for _ in range(nelem)]

# Plot mesh
plot_mesh(X, IX, local_coor_cells, colors, Lx/10)

# Boundary conditions & Loads
# fixed boundary conditions:
alldofs = np.arange(0,ndof)
fixeddofs = np.array([],dtype=int)
F = np.zeros((ndof,1))

match loadcase:
    case 1:
         # Fixed edge at x=0
        node_selection = select_nodes([],X,'NEW','BOX',-0.01,0.01,-0.01,Ly,-1,1)        # Select Nodes in a box
        fixeddofs = np.union1d(fixeddofs,3*node_selection);                             # Lock x-displacement
        fixeddofs = np.union1d(fixeddofs,3*node_selection+1);                           # Lock y-displacement
        fixeddofs = np.union1d(fixeddofs,3*node_selection+2);                           # Lock z-displacement
        node_selection = select_nodes([],Xrot,'NEW','BOX',-0.01,0.01,-0.01,Ly,-1,1)     # Select Rotation nodes in a box
        fixeddofs = np.union1d(fixeddofs,node_selection+3*nnode).astype(np.int64)      # Lock rotation degree of freedom

        # Nodal load along edge x=Lx:
        node_selection = select_nodes([],X,'NEW','BOX',Lx-0.01,Lx+0.01,-0.01,Ly,-1,1)     # Select Nodes in a box
        F[3*node_selection+2] = 20e-3*Ly/(2*nely+1)
    case _:
        sys.exit('Load case not implemented')
freedofs = np.setdiff1d(alldofs,fixeddofs).astype(np.int64)

## Plot loads and boundary conditions
plotBC(X, Xrot, IX, fixeddofs, F, colors, Lx/10)

## FE-Solver
iK = np.repeat(edofMat, 12, axis=0).T.reshape(-1, order='F')
jK = np.repeat(edofMat.T, 12, axis=0).reshape(-1, order='F')
sK = np.empty(144*nelem, dtype=float)                     
U = np.zeros(ndof)           # Displacement vector

# Assembly of stiffness matrix
for e in range(nelem):
    nodenum = IX[e,0:3];                                # Node numbers
    Xe = X[nodenum,0:3];                                # Nodal coordiantes
    C = Cmat(layupCells[e],matProp);                    # Calculate constitutive matrix
    [B,T,Ae] = Bmat(nodenum,Xe, local_coor_cells[e]);   # Calculate strain interpolation matrix                   
    KE = Ae*T.T @ (B.T @ C @ B) @ T                     # Calculate Element stiffness matrix
    sK[144*e:144*(e+1)]  += KE.ravel()

Kmatr = sps.csr_matrix((sK, (iK, jK)), shape=(ndof, ndof))  # Assemble using triplets


# Solve system
U[freedofs] = spsolve(Kmatr[np.ix_(freedofs, freedofs)].tocsc(), F[freedofs])  # Solve system of equations  

##Post processing
sigma = [np.empty((0,)) for _ in range(nelem)]
epsilon =  [np.empty((0,)) for _ in range(nelem)]
fval = np.zeros((nelem,2*nLayerMax))
for e in range(nelem):
    edof = edofMat[e,:];                                   # Retrieve element dofs
    nodenum = IX[e,0:3];                                   # Retrieve nodenumbers
    Xe = X[nodenum,0:3];                                   # Nodal coordiante
    [B,T,Ae] = Bmat(nodenum,Xe, local_coor_cells[e]);      # Evaluate Strain interpolation matrix                   
    Res = B@U[edofMat[e,:]]                                # Calculate Strain and curvature
    epsilon0 = Res[0:3] 
    kappa0 = Res[3:6]
    [sigma[e], epsilon[e]] = evaluate_stresses(epsilon0, kappa0, layupCells[e],matProp)
    fval[e,:]  = EvaluateFailureCriteria(failureCriteria,nLayerMax,matProp,layupCells[e],sigma[e], epsilon[e])

## Plot Results
# Plot deformed shape
plot_deformed(X, IX, U, colors, 1)

# Plot failure criteria
plot_failure_criteria(X, IX, fval, 1)

# Compute norms per node and take max
disp_matrix = np.column_stack((U[0:3*nnode:3],U[1:3*nnode:3],U[2:3*nnode:3]))
max_norm = np.linalg.norm(disp_matrix, axis=1).max()

## Print results
print('Number of elements = ',nelem)
print('Maximum nodal deflection vector sum = ',max_norm)
print('maximum failure criteria value',fval.max())