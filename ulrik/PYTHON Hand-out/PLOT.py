import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri


from FEM import shapefunc

def plot_mesh(X, IX, local_coor_cells, colors, scale):
    """
    Plot a triangular mesh and local element coordinate system vectors (3 per element).
    """
    nelem = IX.shape[0]

    # Create axis if needed
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=IX[:, :3], facecolor='white', edgecolor='k', alpha=0.3)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    ax.axis('equal')

    # Collect direction vectors for each of the 3 local axes
    centers = np.zeros((nelem, 3))
    V1 = np.zeros((nelem, 3))
    V2 = np.zeros((nelem, 3))
    V3 = np.zeros((nelem, 3))

    for elem in range(nelem):
        xe = X[IX[elem, 0:3], :]            # (3,3)
        cent = sum(xe,0)/3
        [e, a, b, Ae] = shapefunc(xe, local_coor_cells[elem])
        centers[elem,0:3] = cent

        V1[elem,0:3] = e[0,0:3]
        V2[elem,0:3] = e[1,0:3]
        V3[elem,0:3] = e[2,0:3]
    
    ax.quiver(centers[:,0], centers[:,1], centers[:,2], V1[:,0], V1[:,1], V1[:,2], length=scale, normalize=True, color=colors[0,:])
    ax.quiver(centers[:,0], centers[:,1], centers[:,2], V2[:,0], V2[:,1], V2[:,2], length=scale, normalize=True, color=colors[1,:])
    ax.quiver(centers[:,0], centers[:,1], centers[:,2], V3[:,0], V3[:,1], V3[:,2], length=scale, normalize=True, color=colors[2,:])
    
    plt.show()
    return

def plotBC(X, Xrot, IX, fixeddofs, F, colors, scale):
    """
    Plot boundary conditions and loads in 3D.

    Parameters:
        X (np.ndarray): Node coordinates (nnode x 3)
        Xrot (np.ndarray): Rotational node data (nnode x 6)
        IX (np.ndarray): Element connectivity (nelem x 3)
        fixeddofs (np.ndarray): Fixed degrees of freedom
        F (np.ndarray): Force/moment vector
        colors (list): List of RGB tuples for coloring
        scale (float): Scaling factor for arrows
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot mesh
    ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=IX[:, :3], facecolor='white', edgecolor='k', alpha=0.3)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    set_axes_equal(ax)

    ndoftran = 3 * X.shape[0]
    ndofrot = Xrot.shape[0]
    
    # Plot translational BCs
    
    for i in range(3):
        selection = ((fixeddofs - i) % 3 == 0) & (fixeddofs <= ndoftran)
        nsel = ((fixeddofs[selection]  - i) / 3).astype(int)
        UVW = np.zeros((len(nsel), 3))
        UVW[:, i] = 1
        ax.quiver(X[nsel, 0], X[nsel, 1], X[nsel, 2],UVW[:, 0], UVW[:, 1], UVW[:, 2], length=scale, normalize=True, color=colors[0,:])
    
    # Plot rotational BCs
    nsel = (fixeddofs[fixeddofs > ndoftran] - ndoftran).astype(int)
    ax.quiver(Xrot[nsel, 0], Xrot[nsel, 1], Xrot[nsel, 2],
              Xrot[nsel, 3], Xrot[nsel, 4], Xrot[nsel, 5],
              length=0.9 * scale, color=colors[0,:])
    ax.quiver(Xrot[nsel, 0], Xrot[nsel, 1], Xrot[nsel, 2],
              Xrot[nsel, 3], Xrot[nsel, 4], Xrot[nsel, 5],
              length=1.2 * scale, color=colors[0,:])
    ax.scatter(Xrot[nsel, 0], Xrot[nsel, 1], Xrot[nsel, 2],
               color=colors[0,:], s=1 * scale)
    
    # Plot loads
    for i in range(3):
        nsel = np.arange(X.shape[0])
        fsel = F[i:ndoftran:3]
        fsel = np.array(fsel).flatten()
        nsel = np.array(nsel).flatten()
        nsel = nsel[fsel != 0]
        fsel = fsel[fsel != 0]
        UVW = np.zeros((len(nsel), 3))
        UVW[:, i] = fsel
        ax.quiver(X[nsel, 0], X[nsel, 1], X[nsel, 2],
                  UVW[:, 0], UVW[:, 1], UVW[:, 2],
                  length=scale*1e-2, color=colors[1,:])
    
    # Plot moments
    nsel = np.arange(ndofrot)
    fsel = F[ndoftran:]
    fsel = np.array(fsel).flatten()
    nsel = np.array(nsel).flatten()
    nsel = nsel[fsel != 0]
    fsel = fsel[fsel != 0]
    print(Xrot[nsel,:])
    ax.quiver(Xrot[nsel, 0], Xrot[nsel, 1], Xrot[nsel, 2],
              Xrot[nsel, 3]* fsel, Xrot[nsel, 4]* fsel, Xrot[nsel, 5]* fsel,
              length=0.9 * scale*1e-2, color=colors[2,:])
    ax.quiver(Xrot[nsel, 0], Xrot[nsel, 1], Xrot[nsel, 2],
              Xrot[nsel, 3]* fsel, Xrot[nsel, 4]* fsel, Xrot[nsel, 5]* fsel,
              length=1.2 * scale*1e-2, color=colors[2,:])
    ax.scatter(Xrot[nsel, 0], Xrot[nsel, 1], Xrot[nsel, 2],
               color=colors[0,:], s=1 * scale)
    
    plt.show()

def plot_deformed(X, IX, U, colors, scale):
    """
    Plot a triangular mesh and local element coordinate system vectors (3 per element).
    """
    nnode = X.shape[0]
    Xdef = X
    Xdef[:,0] += U[0:3*nnode:3]*scale
    Xdef[:,1] += U[1:3*nnode:3]*scale
    Xdef[:,2] += U[2:3*nnode:3]*scale

    # Create axis if needed
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(Xdef[:, 0], Xdef[:, 1], Xdef[:, 2], triangles=IX[:, :3], facecolor='white', edgecolor='k', alpha=0.3)
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    ax.axis('equal')

    plt.show()
    return

def plot_failure_criteria(X, IX, fval, layer):
    """
    Plot failure criteria on a triangular mesh.
    """

    nnode = X.shape[0]
   
    # Convert connectivity to 0-based indexing (MATLAB -> Python)
    triangles = IX[:, :3]

    # Column indices for the requested layer (0-based in Python)
    col_top = 2 * (layer - 1)       # MATLAB: 2*(layer-1)+1
    col_bottom = 2 * layer - 1      # MATLAB: 2*layer

    # Triangulation in the XY plane (equivalent to MATLAB view(2))
    tri = mtri.Triangulation(X[:, 0], X[:, 1], IX[:, :3])

    C_top = fval[:, col_top]
    C_bottom = fval[:, col_bottom]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Left: Top
    im0 = axes[0].tripcolor(tri, C_top)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlabel('x-axis')
    axes[0].set_ylabel('y-axis')
    axes[0].set_title(f'layer {layer} top')
    cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.02)
    cbar0.set_label('Failure criterion')

    # Right: Bottom
    im1 = axes[1].tripcolor(tri, C_bottom)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlabel('x-axis')
    axes[1].set_ylabel('y-axis')
    axes[1].set_title(f'layer {layer} bottom')
    cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.85, pad=0.02)
    cbar1.set_label('Failure criterion')

    plt.show()


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


