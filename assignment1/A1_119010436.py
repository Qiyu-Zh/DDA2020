

import numpy as np
# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_119010436(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :w type: numpy.ndarray
    :XT type: numpy.ndarray
    :InvXTX type: numpy.ndarray
   
    """
    # your code goes here
    try:
        b=np.linalg.inv(X.T@X)
        w=b@X.T@y
        return w, X.T, b
    except:
        return 'X.T@X is not invertible'
    # return in this order

# X = np.array([[1, 2], [4, 3], [5, 6], [3, 8], [9, 10]])
# y = np.array([-1, 0, 1, 0, 0])

# print(A1_119010436(X, y))

    
