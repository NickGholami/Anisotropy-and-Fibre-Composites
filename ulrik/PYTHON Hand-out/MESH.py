
import numpy as np
def generate_mesh(Lx: float, Ly: float, nelx: int, nely: int):
    """
          [X, Xrot, IX, edofMat, nelem, nnode, ndof] = GenerateMesh(Lx, Ly, nelx, nely)
    Inputs:
        nelx, nely : number of macro elements along x and y (each split into 8 triangles)
        Lx, Ly     : domain size in x and y
    Outputs:
        X  : Nodal coordinate
        Xrot : Edge coordinates
        IX : Element connectivity matrix
        edofMat : Element degree of freedom matrix
        nelem : Number of elements
        nnode : Number of nodes
        ndof  : Number of degrees of freedom
    """

    # --- Define amount of elements and nodes ---
    nelem = 8 * nelx * nely
    nx = 2 * nelx + 1
    ny = 2 * nely + 1
    nnode = nx * ny

    # --- Node coordinates along each axis ---
    x1 = np.linspace(0.0, Lx, nx)
    y1 = np.linspace(0.0, Ly, ny)

    # --- Define nodal coordinates ---
    xx, yy = np.meshgrid(x1, y1, indexing="xy")  # shape (ny, nx)
    X = np.zeros((nnode, 3), dtype=float)
    X[:, 0] = xx.ravel(order="C")
    X[:, 1] = yy.ravel(order="C")


    # --- Define Element connectivity (first 3 columns: nodes, 1-based) ---
    IX = np.zeros((nelem, 6), dtype=int)
    row = 0
    for iy in range(nely):
        for ix in range(nelx):
            # Center node (0-based) in the refined grid
            n0 = (2 * iy + 1) * nx + (2 * ix + 1)

            tris = np.array([
                [n0, n0 - 1,         n0 - nx - 1],
                [n0, n0 - nx - 1,    n0 - nx    ],
                [n0, n0 - nx,        n0 - nx + 1],
                [n0, n0 - nx + 1,    n0 + 1     ],
                [n0, n0 + 1,         n0 + nx + 1],
                [n0, n0 + nx + 1,    n0 + nx    ],
                [n0, n0 + nx,        n0 + nx - 1],
                [n0, n0 + nx - 1,    n0 - 1     ],
            ], dtype=int)

            IX[row:row + 8, 0:3] = tris 
            row += 8

    # --- Setup Edge Numbering ---
    IX, Xrot = _setup_edge_numbering(X, IX)

    # --- Setup Element DOF Numbering (1-based like MATLAB) ---
    edofMat = np.zeros((nelem, 12), dtype=int)
    # Nodal DOFs: for each node id n (1-based) -> DOFs (3n-2, 3n-1, 3n)
    edofMat[:, 0:3] = IX[:, 0:3] * 3
    edofMat[:, 3:6] = IX[:, 0:3] * 3 +1
    edofMat[:, 6:9] = IX[:, 0:3] * 3 +2
    # Edge DOFs: offset by 3*nnode
    edofMat[:, 9:12] = IX[:, 3:6] + 3 * nnode
    ndof = 3 * nnode + Xrot.shape[0]

    return X, Xrot, IX, edofMat, nelem, nnode, ndof

def _setup_edge_numbering(X: np.ndarray, IX: np.ndarray):
    """
    Python translation of the MATLAB 'setupEdgeNumbering' subfunction.

    Inputs:
        X  : (nnode, 3) node coordinates
        IX : (nelem, 6) (first 3 columns are node IDs, 1-based)

    Outputs:
        IX   : with columns 4..6 filled with 1-based unique edge IDs
        Xrot : (nedge, 6) rows = [midpoint_x, midpoint_y, midpoint_z, vx, vy, vz]
    """
    nelem = IX.shape[0]

    # Dictionary to map undirected edges -> edge_id (1-based)
    # Edge key = tuple(sorted((n1, n2))) with 1-based node IDs
    edge_map = {}
    xrot_rows = []  # accumulate rows, then convert to array
    next_edge_id = 0

    for e in range(nelem):
        v1 = int(IX[e, 0])
        v2 = int(IX[e, 1])
        v3 = int(IX[e, 2])

        verts = [v1, v2, v3, v1]  # cycle edges: (v1,v2), (v2,v3), (v3,v1)

        for ei in range(3):
            n1 = int(verts[ei])
            n2 = int(verts[ei + 1])
            a, b = (n1, n2) if n1 < n2 else (n2, n1)  # undirected edge key
            key = (a, b)

            if key not in edge_map:
                edge_map[key] = next_edge_id

                # Compute midpoint and edge direction vector
                # Convert to 0-based for X indexing
                p1 = X[a, :]
                p2 = X[b, :]
                pt = 0.5 * (p1 + p2)
                v = (p2 - p1)
                v = v / np.linalg.norm(v)
                xrot_rows.append([pt[0], pt[1], pt[2], v[0], v[1], v[2]])
                IX[e, 3 + ei] = next_edge_id  # store edge id (1-based)
                next_edge_id += 1
            else:
                IX[e, 3 + ei] = edge_map[key]

    Xrot = np.array(xrot_rows, dtype=float) if xrot_rows else np.zeros((0, 6), dtype=float)
    return IX, Xrot
