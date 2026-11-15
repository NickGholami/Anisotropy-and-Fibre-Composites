import numpy as np


def select_nodes(selection, X, type_, shape=None, var1=None, var2=None, var3=None,
                 var4=None, var5=None, var6=None):
    # Function for selecting nodes
    #
    #   Input:
    #       Selection:      Current Selection                (Logical array)
    #       X:              Nodal Coordinates                (nnode x 3)
    #       type:           Selection type                   (String)
    #       shape:          Selection shape
    #       var1-6:         Variables for selection region   (num)
    #
    #   Output:
    #       Selection:      Current Selection                (vector)

    NewSelection = np.arange(X.shape[0])

    if type_.lower() in ['new', 'add', 'remove']:
        if shape.lower() == 'box':
            # var1,var2: x-limits;
            # var3,var4: y-limits;
            # var5,var6: z-limits
            NewSelection = NewSelection[(var1 <= X[:, 0]) & (var2 >= X[:, 0]) &
                                        (var3 <= X[:, 1]) & (var4 >= X[:, 1]) &
                                        (var5 <= X[:, 2]) & (var6 >= X[:, 2])]

        elif shape.lower() == 'cylinder':
            # var1,var2,var3: cylinder center coordinates
            # var4: radius;
            # var5: cylinder height;
            # var6: axis (1=x, 2=y, 3=z)
            if var6 not in [1, 2, 3]:
                raise ValueError('Cylinder axis must be 1, 2, or 3')
            center = np.array([var1, var2, var3], dtype=float)
            axial = X[:, var6-1] - center[var6-1]
            radial = X - center
            radial[:, var6-1] = 0
            NewSelection = NewSelection[(axial >= -var5/2) & (axial <= var5/2) &
                                        (np.sum(radial**2, axis=1) <= var4**2)]

        elif shape.lower() == 'sphere':
            # var1,var2,var3: sphere center coordinates
            # var4: radius;
            # var5,var6: unused
            center = np.array([var1, var2, var3], dtype=float)
            NewSelection = NewSelection[np.sum((X - center)**2, axis=1) <= var4**2]

        else:
            raise ValueError('Invald selection shape')

    if type_.lower() in ['new', 'all']:
        selection = NewSelection
    elif type_.lower() == 'add':
        selection = np.union1d(selection, NewSelection)
    elif type_.lower() == 'remove':
        selection = np.setdiff1d(selection, NewSelection)
    elif type_.lower() == 'inverse':
        selection = np.setdiff1d(NewSelection, selection)
    else:
        raise ValueError('unkown slection type')

    return selection


def select_elem(selection, X, IX, type_, shape=None, var1=None, var2=None, var3=None,
                var4=None, var5=None, var6=None):
    # Function for selecting elements
    #
    #   Input:
    #       Selection:      Current Selection                (Logical array)
    #       X:              Nodal Coordinates                (nnode x 3)
    #       IX:             Element Connectivity             (nelem x 3)
    #       type:           Selection type                   (String)
    #       shape:          Selection shape
    #       var1-6:         Variables for selection region   (num)
    #
    #   Output:
    #       Selection:      Current Selection                (vector)

    NewSelection = np.arange(IX.shape[0])

    # Compute element centroid coordinates for geometric selection
    IXc = np.mean(X[IX[:, 0:3], :], axis=1)

    if type_.lower() in ['new', 'add', 'remove']:
        if shape.lower() == 'box':
            # var1,var2: x-limits;
            # var3,var4: y-limits;
            # var5,var6: z-limits
            NewSelection = NewSelection[(var1 <= IXc[:, 0]) & (var2 >= IXc[:, 0]) &
                                        (var3 <= IXc[:, 1]) & (var4 >= IXc[:, 1]) &
                                        (var5 <= IXc[:, 2]) & (var6 >= IXc[:, 2])]

        elif shape.lower() == 'cylinder':
            # var1,var2,var3: cylinder center coordinates
            # var4: radius;
            # var5: cylinder height;
            # var6: axis (1=x, 2=y, 3=z)
            if var6 not in [1, 2, 3]:
                raise ValueError('Cylinder axis must be 1, 2, or 3')
            center = np.array([var1, var2, var3], dtype=float)
            axial = IXc[:, var6-1] - center[var6-1]
            radial = IXc - center
            radial[:, var6-1] = 0
            NewSelection = NewSelection[(axial >= -var5/2) & (axial <= var5/2) &
                                        (np.sum(radial**2, axis=1) <= var4**2)]

        elif shape.lower() == 'sphere':
            # var1,var2,var3: sphere center coordinates
            # var4: radius;
            # var5,var6: unused
            center = np.array([var1, var2, var3], dtype=float)
            NewSelection = NewSelection[np.sum((IXc - center)**2, axis=1) <= var4**2]

        else:
            raise ValueError('Invald selection shape')

    if type_.lower() in ['new', 'all']:
        selection = NewSelection
    elif type_.lower() == 'add':
        selection = np.union1d(selection, NewSelection)
    elif type_.lower() == 'remove':
        selection = np.setdiff1d(selection, NewSelection)
    elif type_.lower() == 'inverse':
        selection = np.setdiff1d(NewSelection, selection)
    else:
        raise ValueError('unkown slection type')

    return selection
