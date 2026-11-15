import numpy as np


def generate_mesh(Lx, Ly, nelx, nely):
    #   Inputs:
    #     nelx, nely : half number of elements in x and y
    #     Lx, Ly     : domain size in x and y
    #   Outputs:
    #     X  : Nodal coordinate
    #     Xrot : Edge coordinates
    #     IX : Element connectivity matrix
    #     edofMat : Element degree of freedom matrix
    #     nelem : Number of elements
    #     nnode : Number of nodes
    #     ndof  : Number of degrees of freedom

    # --- Define amount of elment as nodes ---
    nelem = 8*nelx*nely
    nx = 2*nelx + 1
    ny = 2*nely + 1
    nnode = nx*ny

    # --- Node coordinates along each axis ---
    x1 = np.linspace(0, Lx, nx)
    y1 = np.linspace(0, Ly, ny)

    # --- Define nodal coordinate ---
    X = np.zeros((nnode, 3))
    for iy in range(ny):
        for ix in range(nx):
            id_ = ix + iy*nx
            X[id_, 0] = x1[ix]
            X[id_, 1] = y1[iy]
            X[id_, 2] = 0.0

    # --- Define Element connectivity: ---
    IX = np.zeros((nelem, 6), dtype=int)
    row = 0
    for iy in range(nely):
        for ix in range(nelx):
            n0 = (2*iy + 1)*nx + (2*ix + 1)
            IX[row + 0, 0:3] = [n0,         n0-1,         n0-(nx)-1        ]
            IX[row + 1, 0:3] = [n0,         n0-(nx)-1,    n0-(nx)          ]
            IX[row + 2, 0:3] = [n0,         n0-(nx),      n0-(nx)+1        ]
            IX[row + 3, 0:3] = [n0,         n0-(nx)+1,    n0+1             ]
            IX[row + 4, 0:3] = [n0,         n0+1,         n0+(nx)+1        ]
            IX[row + 5, 0:3] = [n0,         n0+(nx)+1,    n0+(nx)          ]
            IX[row + 6, 0:3] = [n0,         n0+(nx),      n0+(nx)-1        ]
            IX[row + 7, 0:3] = [n0,         n0+(nx)-1,    n0-1             ]
            row = row + 8

    # --- Setup Edge Numbering ---
    IX, Xrot = setup_edge_numbering(X, IX)

    # --- Setup Element DOF Numbering ---
    edofMat = np.zeros((nelem, 12), dtype=int)
    edofMat[:, 0:3] = IX[:, 0:3]*3
    edofMat[:, 3:6] = IX[:, 0:3]*3+1
    edofMat[:, 6:9] = IX[:, 0:3]*3+2
    edofMat[:, 9:12] = IX[:, 3:6]+3*nnode

    ndof = 3*nnode+Xrot.shape[0]
    return X, Xrot, IX, edofMat, nelem, nnode, ndof


def generate_L_mesh(a, b, c, d, ex, ey):
    #   Inputs:
    #     ex, ey: approximate element size in x and y
    #     a,b,c,d: geometry parameters (see picture below)
    #   Outputs:
    #     X  : Nodal coordinate
    #     Xrot : Edge coordinates
    #     IX : Element connectivity matrix
    #     edofMat : Element degree of freedom matrix
    #     nelem : Number of elements
    #     nnode : Number of nodes
    #     ndof  : Number of degrees of freedom
    #
    # The function generate a mesh with the shape:
    #
    #                    a
    #     +-------------------------+
    #     |                         |
    # b   |                         |
    #     |                         |
    #     +-------------------+     |
    #                         |     |
    #                      d  |     |
    #                         +-----+
    #                            c

    # --- Define each mesh block ---
    nelx = int(np.floor((a-c)/ex))
    nely = int(np.floor((b)/ey))
    X1, IX1 = generate_block(0, (a-c), 0, b, nelx, nely)

    nelx = int(np.floor(c/ex))
    nely = int(np.floor(b/ey))
    X2, IX2 = generate_block((a-c), a, 0, b, nelx, nely)
    IX2 = IX2+X1.shape[0]

    nelx = int(np.floor(c/ex))
    nely = int(np.floor(d/ey))
    X3, IX3 = generate_block((a-c), a, 0, -d, nelx, nely)
    IX3 = IX3+(X1.shape[0]+X2.shape[0])

    X = np.vstack([X1, X2, X3])
    IX = np.vstack([IX1, IX2, IX3])

    # --- Merge nodes ---
    X, IX = merge_nodes(X, IX)

    # --- Define amount of element and nodes ---
    nnode = X.shape[0]
    nelem = IX.shape[0]

    # --- Setup Edge Numbering ---
    IX, Xrot = setup_edge_numbering(X, IX)

    # --- Setup Element DOF Numbering ---
    edofMat = np.zeros((nelem, 12), dtype=int)
    edofMat[:, 0:3] = IX[:, 0:3]*3
    edofMat[:, 3:6] = IX[:, 0:3]*3+1
    edofMat[:, 6:9] = IX[:, 0:3]*3+2
    edofMat[:, 9:12] = IX[:, 3:6]+3*nnode
    ndof = 3*nnode+Xrot.shape[0]

    return X, Xrot, IX, edofMat, nelem, nnode, ndof


def generate_block(x1, x2, y1, y2, nelx, nely):

    nelem = 8*nelx*nely
    nx = 2*nelx + 1
    ny = 2*nely + 1
    nnode = nx*ny

    # --- Node coordinates along each axis ---
    x1 = np.linspace(x1, x2, nx)
    y1 = np.linspace(y1, y2, ny)

    # --- Define nodal coordinate ---
    X = np.zeros((nnode, 3))
    for iy in range(ny):
        for ix in range(nx):
            id_ = ix + iy*nx
            X[id_, 0] = x1[ix]
            X[id_, 1] = y1[iy]
            X[id_, 2] = 0.0

    # --- Define Element connectivity: ---
    IX = np.zeros((nelem, 3), dtype=int)
    row = 0
    for iy in range(nely):
        for ix in range(nelx):
            n0 = (2*iy + 1)*nx + (2*ix + 1)
            IX[row + 0, 0:3] = [n0,         n0-1,         n0-(nx)-1        ]
            IX[row + 1, 0:3] = [n0,         n0-(nx)-1,    n0-(nx)          ]
            IX[row + 2, 0:3] = [n0,         n0-(nx),      n0-(nx)+1        ]
            IX[row + 3, 0:3] = [n0,         n0-(nx)+1,    n0+1             ]
            IX[row + 4, 0:3] = [n0,         n0+1,         n0+(nx)+1        ]
            IX[row + 5, 0:3] = [n0,         n0+(nx)+1,    n0+(nx)          ]
            IX[row + 6, 0:3] = [n0,         n0+(nx),      n0+(nx)-1        ]
            IX[row + 7, 0:3] = [n0,         n0+(nx)-1,    n0-1             ]
            row = row + 8

    return X, IX


def merge_nodes(X, IX):
    tol = 1e-10
    Xkey = np.round(X/tol)
    _, ia, nodeMap = np.unique(Xkey, axis=0, return_index=True, return_inverse=True)
    nodeMap = nodeMap.ravel()
    X = X[ia, :]
    IX = nodeMap[IX]

    s1 = X[IX[:, 2], :] - X[IX[:, 1], :]
    s2 = X[IX[:, 0], :] - X[IX[:, 2], :]
    normals = np.cross(s1, s2, axis=1)

    ref = np.flatnonzero(np.linalg.norm(normals, axis=1) > tol)
    if ref.size > 0:
        flip = normals @ normals[ref[0], :] < 0
        IX[np.ix_(flip, [1, 2])] = IX[np.ix_(flip, [2, 1])]

    return X, IX


def setup_edge_numbering(X, IX):
    nelem = IX.shape[0]

    # Columns 4-6 of IX hold the edge ids
    if IX.shape[1] < 6:
        IX = np.hstack([IX[:, 0:3], np.zeros((nelem, 3), dtype=int)])

    # Hash table: 3*nelem rows, 3 columns = [edge_id, n1, n2]
    hash_data_base = np.full((3 * nelem, 3), -1, dtype=int)

    # Preallocate Xrot to the worst-case number of edges: 3*nelem.
    Xrot = np.zeros((3 * nelem, 6))

    count = 0
    for e in range(nelem):
        # Local triangle vertices (0-based)
        v1 = int(IX[e, 0])
        v2 = int(IX[e, 1])
        v3 = int(IX[e, 2])

        # Process the 3 edges: (v1,v2), (v2,v3), (v3,v1)
        v = [v1, v2, v3, v1]

        for ei in range(3):
            n1 = v[ei]
            n2 = v[ei + 1]

            # Swap order if n2 is smaller than n1
            if n2 < n1:
                n1, n2 = n2, n1

            # Hash function
            hash_ = (17 * n1 + 1051 * n2) % (3 * nelem)
            i = hash_

            while True:
                if hash_data_base[i, 0] == -1:
                    # Empty slot: create new edge
                    hash_data_base[i, 0] = count  # edge_id
                    hash_data_base[i, 1] = n1
                    hash_data_base[i, 2] = n2

                    IX[e, ei + 3] = count  # store 0-based edge id

                    # Midpoint of the edge
                    pt = 0.5 * (X[n1, :] + X[n2, :])
                    t = X[n2, :] - X[n1, :]
                    t = t / np.linalg.norm(t)
                    Xrot[count, :] = np.concatenate([pt, t])

                    count = count + 1
                    break

                elif hash_data_base[i, 1] == n1 and hash_data_base[i, 2] == n2:
                    # Slot matches existing edge
                    IX[e, ei + 3] = hash_data_base[i, 0]  # existing ID
                    break

                else:
                    # Move to next slot (linear probing with wrap)
                    if i == 3 * nelem - 1:
                        i = 0
                    else:
                        i = i + 1

    # Trim Xrot to the number of unique edges created
    Xrot = Xrot[0:count, :]
    return IX, Xrot
