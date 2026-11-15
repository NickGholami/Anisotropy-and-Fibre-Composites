import numpy as np
def select_nodes(selection, X, type_, shape, var1, var2, var3, var4, var5, var6):
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

    n_nodes = X.shape[0]
    new_selection = np.arange(n_nodes)

    if type_.lower() in ['new', 'add', 'remove']:
        if shape.lower() == 'box':
            mask = (
                (X[:, 0] >= var1) & (X[:, 0] <= var2) &
                (X[:, 1] >= var3) & (X[:, 1] <= var4) &
                (X[:, 2] >= var5) & (X[:, 2] <= var6)
            )
            new_selection = new_selection[mask]
        elif shape.lower() == 'cylinder':
            raise NotImplementedError("Cylinder selection shape not implemented")
        elif shape.lower() == 'sphere':
            raise NotImplementedError("Sphere selection shape not implemented")
        else:
            raise ValueError("Invalid selection shape")

    if type_.lower() in ['new', 'all']:
        selection = new_selection
    elif type_.lower() == 'add':
        selection = np.union1d(selection, new_selection)
    elif type_.lower() == 'remove':
        selection = np.setdiff1d(selection, new_selection, assume_unique=True)
    elif type_.lower() == 'inverse':
        selection = np.setdiff1d(new_selection, selection, assume_unique=True)
    else:
        raise ValueError("Unknown selection type")

    return selection