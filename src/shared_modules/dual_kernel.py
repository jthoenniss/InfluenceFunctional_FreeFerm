"""
Module that contains the necessary functions to convert a 4x4 fermionic operator to a "dual" Grassmann kernel.
This is needed for single-orbital problems in real and imaginary time.
"""

import numpy as np


def transform_backward_kernel(kernel: np.ndarray) -> np.ndarray:
    """
    For dual gates on the backward branch, the ingoing and outgoing Grassmann variables are reversed,
    i.e. the labels for half/full (or odd/even) time steps are swapped.
    Hence, the basis elements need to be reordered for each spin species.
    This means that:
    1) the rows and columns with indices 1 and 2 in the dual gate are swapped, and
    2) the last row and last column are multiplied with -1, respectively.

    Parameters:
    - kernel (np.ndarray): a 4x4 matrix representing the original kernel.

    Returns:
    - np.ndarray: The transformed kernel matrix.
    """
    #check size of input
    assert kernel.shape == (4,4), "The input must be a 4x4 matrix."

    # Index mapping based on the backward_order logic:
    # Swap second and third rows and columns, negate the fourth row and column
    index_order = [0, 2, 1, 3]
    sign_changes = np.array([1, 1, 1, -1])

    # Reorder and apply sign changes
    transformed_kernel = kernel[:, index_order][index_order, :]
    transformed_kernel *= sign_changes[:, np.newaxis]  # Apply sign change to rows
    transformed_kernel *= sign_changes  # Apply sign change to columns

    return transformed_kernel

def dual_kernel(gate_coeffs: np.ndarray) -> np.ndarray:

    """
    Compute the "dual" kernel from the coefficients of a gate. This introduces signs that come from reordering Grassmann variables.
    The dual kernel is constructed in the convention:
    kernel_Grassmann = \eta_down^T @ kernel @ \eta_up,
    where \eta_down = [1, \bar{\eta}_down, \eta_down, \bar{\eta}_down \eta_down],
       and \eta_up = [1, \bar{\eta}_up, \eta_up, \bar{\eta}_up \eta_up].

    The transformation is the following:
    [[a00, a01, a02, a03],
     [a10, a11, a12, a13],
     [a20, a21, a22, a23],
     [a30, a31, a32, a33]] -> [[a00, a10, a01, a11],
                               [a20, a30, a21, a31],
                               [a02, -a12, -a03, a13],
                               [a22, -a32, -a23, a33]].

    Parameters:
    - gate_coeffs (np.ndarray): A 4x4 matrix representing the coefficients of the many-body gate.

    Returns:
    - np.ndarray: The dual kernel.
    
    """

    #check that the input is a 4x4 matrix
    assert gate_coeffs.shape == (4,4), "The input must be a 4x4 matrix."

    kernel_dual = np.zeros((4,4),dtype=np.complex_)

    # The kernel is constructed based on the coefficients of the many-body gate.
    # Note: The signs that come from the variable substitutions are already included here!
    # The sign changes are equivalent on the forward and backward branch. 
    kernel_dual[0,0] = gate_coeffs[0,0]
    kernel_dual[0,1] = gate_coeffs[1,0]
    kernel_dual[0,2] = gate_coeffs[0,1]
    kernel_dual[0,3] = gate_coeffs[1,1]
    kernel_dual[1,0] = gate_coeffs[2,0]
    kernel_dual[1,1] = gate_coeffs[3,0]
    kernel_dual[1,2] = gate_coeffs[2,1]
    kernel_dual[1,3] = gate_coeffs[3,1]
    kernel_dual[2,0] = gate_coeffs[0,2]
    kernel_dual[2,1] = - gate_coeffs[1,2]
    kernel_dual[2,2] = - gate_coeffs[0,3]
    kernel_dual[2,3] = gate_coeffs[1,3]
    kernel_dual[3,0] = gate_coeffs[2,2]
    kernel_dual[3,1] = - gate_coeffs[3,2]
    kernel_dual[3,2] = - gate_coeffs[2,3]
    kernel_dual[3,3] = gate_coeffs[3,3]

    return kernel_dual


def overlap_signs(kernel_dual: np.ndarray) -> np.ndarray:

    """
    Adjust the DUAL kernel by the signs that result from bringing the path integral into overlap form.
    This alters the signs of all entires in the rows with indices 2 and 3 and all entries in the columns with indices 1 and 3.

    Parameters:
    - kernel_dual (np.ndarray): A 4x4 matrix representing the dual kernel.

    Returns:
    - np.ndarray: The adjusted dual kernel.
    """
    #check size of input
    assert kernel_dual.shape == (4,4), "The input must be a 4x4 matrix."

    kernel_dual[2,:] *= -1
    kernel_dual[:,1] *= -1
    kernel_dual[3,:] *= -1
    kernel_dual[:,3] *= -1

    return kernel_dual



def operator_to_kernel(gate_coeffs: np.ndarray, string: bool = False, branch = 'f', boundary: bool = False) -> np.ndarray:
    """
    Function to convert a many-body gate to the 'dual' kernel representation that is used for the MPS-MPO-MPS contraction.

    Parameters:
    - gate_coeffs (numpy.ndarray): A (4x4) numpy array representing the matrix of the many-body gate.
      The matrix must adhere to the following convention for the usual quantum many-body gate:
      gate = v^\dagger @ gate_coeffs @ v, with: v = [<0|, <0|c_up , <0|c_down, <0|c_upc_down].

    - branch (string): The branch of the impurity gate. Must be set to 'f' for the forward branch and 'b' for the backward branch.
                        For imaginary time, use 'f' (default).

    - string (bool): The single-down-Grassmann entries in the gates that lie between to impurity operators 
        that contains an odd number of Grassmann variables must be multiplied by -1 in order to give a correct global sign.
        For these gates, string (because this corresponds to a fermionic Jordan-Wigner string) must be set to True.

    - boundary (bool): If True, the kernel will be constructed with antiperiodic boundary conditions. Must be set to True for the last impurity gate.
       
    Returns:
    - numpy.ndarray: A (4x4) numpy array representing the kernel of the many-body gate in the 'dual' representation.
        The kernel is constructed in the convention:
        kernel_Grassmann = \eta_down^T @ kernel @ \eta_up,
        where \eta_down = [1, \bar{\eta}_down, \eta_down, \bar{\eta}_down \eta_down],
           and \eta_up = [1, \bar{\eta}_up, \eta_up, \bar{\eta}_up \eta_up].
    """

    #check that the input is a 4x4 matrix
    assert gate_coeffs.shape == (4,4), "The input must be a 4x4 matrix."

    # The kernel is constructed based on the coefficients of the many-body gate.
    kernel_dual = dual_kernel(gate_coeffs)

    # The kernel is adjusted by the signs that result from bringing the path integral into overlap form.
    kernel_dual = overlap_signs(kernel_dual)

    
    if branch == 'b':
        # For dual gates on the backward branch, the ingoing and outgoing Grassmann variables are reversed,
        # i.e. the labels for half/full (or odd/even) time steps are swapped.
        # Hence, the basis elements need to be reordered for each spin species.
        # This means that the rows and columns with indices 1 and 2 are swapped.
        # and the last row and last column are multiplied with -1, respectively.
        kernel_dual = transform_backward_kernel(kernel_dual)

    
    if boundary == True: #antiperiodic boundary conditions: All entries with exactly one 'barred' Grassmann variable must be multiplied by -1.
        kernel_dual[[i for i in [1,3]],:] *= -1
        kernel_dual[:,[i for i in [1,3]]] *= -1

    # If the gate is a string gate, the single-down-Grassmann entries must be multiplied by -1.
    # Factor of complex i is included here for the reversing of the spin-down Grassmann variables. 
    # This is only relevant if one deals with spin-hopping Hamiltonians, i.e. Hamiltonians that can be odd within a single spin species. 
    string_var = -1. if string == True else 1.

    kernel_dual = np.diag([1, string_var * 1.j,string_var * 1.j,1]) @ kernel_dual
   
    return kernel_dual




def inverse_dual_kernel(kernel_dual: np.ndarray) -> np.ndarray:
    """
    Performs the inverse of the function 'dual_kernel'.
    Typically used to transfrom a dual kernel back to the original gate in the fermionic many-body basis.

    This is the inverse operation of the transformation:
    [[a00, a01, a02, a03],
     [a10, a11, a12, a13],
     [a20, a21, a22, a23],
     [a30, a31, a32, a33]] <- [[a00, a10, a01, a11],
                               [a20, a30, a21, a31],
                               [a02, -a12, -a03, a13],
                               [a22, -a32, -a23, a33]].

    Parameters:
    - kernel_dual (np.ndarray): A 4x4 matrix representing the dual kernel.

    Returns:
    - np.ndarray: The inverse of the 'dual_kernel' function applied to the kernel. 
                    If the input is a valid dual kernel, the result is the original fermionic gate.
    """
    #check size of input
    assert kernel_dual.shape == (4,4), "The input must be a 4x4 matrix."

    gate_coeffs = np.zeros((4, 4), dtype=np.complex_)

    # Reconstruct the original matrix
    gate_coeffs[0, 0] = kernel_dual[0, 0]
    gate_coeffs[1, 0] = kernel_dual[0, 1]
    gate_coeffs[0, 1] = kernel_dual[0, 2]
    gate_coeffs[1, 1] = kernel_dual[0, 3]
    gate_coeffs[2, 0] = kernel_dual[1, 0]
    gate_coeffs[3, 0] = kernel_dual[1, 1]
    gate_coeffs[2, 1] = kernel_dual[1, 2]
    gate_coeffs[3, 1] = kernel_dual[1, 3]
    gate_coeffs[0, 2] = kernel_dual[2, 0]
    gate_coeffs[1, 2] = -kernel_dual[2, 1]
    gate_coeffs[0, 3] = -kernel_dual[2, 2]
    gate_coeffs[1, 3] = kernel_dual[2, 3]
    gate_coeffs[2, 2] = kernel_dual[3, 0]
    gate_coeffs[3, 2] = -kernel_dual[3, 1]
    gate_coeffs[2, 3] = -kernel_dual[3, 2]
    gate_coeffs[3, 3] = kernel_dual[3, 3]

    return gate_coeffs