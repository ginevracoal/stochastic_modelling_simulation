"""
This file contains useful functions and imports used for stochastic modelling
"""
import numpy as np

def check_stochastic_matrix(matrix):
    """
    This function checks if a matrix is stochastic.
    A matrix is said to be stochastic if:
        1)every element is positive
        2)rows sums to one
    """
    rowSum = np.allclose(np.sum(matrix,axis=1),1)
    allGreater=np.all(matrix >= 0)
    return (rowSum and allGreater)