import numpy as np

def shapefunc(x, alignVec):
    """
    Function to calculate shape functions.

    Parameters:
        x (ndarray): Nodal coordinates (3x3)
        alignVec (ndarray): Coordinate system to align local element coordinate system (3x3)

    Returns:
        Te (ndarray): Transformation matrix from element to global coordinate system (3x3)
        a (ndarray): Triangle coefficients (3x1)
        b (ndarray): Triangle coefficients (3x1)
        Ae (float): Element surface area
    """
    e = np.zeros((3, 3))
    s = np.zeros((3, 3))
    a = np.zeros(3)
    b = np.zeros(3)

    # Find element normal vector
    e[0] = x[1] - x[0]
    e[1] = x[2] - x[0]
    e[0] /= np.linalg.norm(e[0])
    e[1] /= np.linalg.norm(e[1])
    e[2] = np.cross(e[0], e[1])
    e[2] /= np.linalg.norm(e[2])

    # Check angle between first-axis and normal
    theta = np.arccos(np.dot(alignVec[0], e[2]) / (np.linalg.norm(alignVec[0]) * np.linalg.norm(e[2])))

    # Project alignment axis onto plane
    if (3/4)*np.pi >= theta >= np.pi/4:
        AonN = np.dot(alignVec[0], e[2]) / np.sum(e[2]**2) * e[2]
        e[0] = alignVec[0] - AonN
        e[0] /= np.linalg.norm(e[0])
        e[1] = np.cross(e[2], e[0])
        e[1] /= np.linalg.norm(e[1])
    else:
        AonN = np.dot(alignVec[1], e[2]) / np.sum(e[2]**2) * e[2]
        e[1] = alignVec[1] - AonN
        e[1] /= np.linalg.norm(e[1])
        e[0] = np.cross(e[1], e[2])
        e[0] /= np.linalg.norm(e[0])

    if abs(np.dot(e[0], e[2])) > 1e-10:
        print(np.dot(e[0], e[2]))
        raise ValueError("Element vectors are not orthogonal")

    s[0] = x[2] - x[1]
    s[1] = x[0] - x[2]
    s[2] = x[1] - x[0]

    for i in range(3):
        a[i] = np.dot(s[i], e[0])
        b[i] = -np.dot(s[i], e[1])

    Ae = np.linalg.norm(np.cross(s[0], s[1])) / 2
    Te = e
    return Te, a, b, Ae


def Bmat(nodeNum, Xe, elemAlign):
    """
    Compute B matrix, rotation matrix T, and element area Ae.

    Parameters:
        nodeNum (np.ndarray): Node numbers (3,)
        Xe (np.ndarray): Element coordinates (3 x 3)
        elemAlign (any): Alignment info passed to shapefunc

    Returns:
        B (np.ndarray): 6x12 stiffness matrix
        T (np.ndarray): 12x12 rotation matrix
        Ae (float): Element area
    """
    Te, a, b, Ae = shapefunc(Xe, elemAlign)

    # Rotation matrix
    T = np.eye(12)
    T[0:7:3, 0:7:3] = Te
    T[1:8:3, 1:8:3] = Te
    T[2:9:3, 2:9:3] = Te

    # Flip node logic
    checknode = np.array([
        nodeNum[0] > nodeNum[1],
        nodeNum[1] > nodeNum[2],
        nodeNum[2] > nodeNum[0]
    ])
    flipNode = (checknode == 0).astype(int) - checknode.astype(int)

    # Membrane stiffness
    d1 = (1 / (2 * Ae)) * b
    d2 = (1 / (2 * Ae)) * a
    Bm = np.block([
        [d1, np.zeros(3)],
        [np.zeros(3), d2],
        [d2, d1]
    ])

    # Bending stiffness
    b = -b
    a = -a

    s12 = b[2]**2 + a[2]**2
    s23 = b[0]**2 + a[0]**2
    s31 = b[1]**2 + a[1]**2
    s12s = np.sqrt(s12)
    s23s = np.sqrt(s23)
    s31s = np.sqrt(s31)

    val1 = b[1]*a[1]/s31 - b[2]*a[2]/s12
    val2 = b[2]*a[2]/s12 - b[0]*a[0]/s23
    val3 = b[0]*a[0]/s23 - b[1]*a[1]/s31

    Bb = np.zeros((3, 6))
    Bb[:, 0] = [val1, -val1, 2*(b[1]**2/s31 - b[2]**2/s12)]
    Bb[:, 1] = [val2, -val2, 2*(b[2]**2/s12 - b[0]**2/s23)]
    Bb[:, 2] = [val3, -val3, 2*(b[0]**2/s23 - b[1]**2/s31)]
    Bb[:, 3] = flipNode[0] * np.array([-b[2]**2/s12s, -a[2]**2/s12s, 2*b[2]*a[2]/s12s])
    Bb[:, 4] = flipNode[1] * np.array([-b[0]**2/s23s, -a[0]**2/s23s, 2*b[0]*a[0]/s23s])
    Bb[:, 5] = flipNode[2] * np.array([-b[1]**2/s31s, -a[1]**2/s31s, 2*b[1]*a[1]/s31s])
    Bb = Bb / Ae

    # Combine membrane and bending
    B = np.block([
        [Bm, np.zeros((3, 6))],
        [np.zeros((3, 6)), Bb]
    ])

    return B, T, Ae

