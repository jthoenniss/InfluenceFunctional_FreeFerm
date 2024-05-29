"""
Module that contains the necessary functions to convert a 4x4 fermionic operator to a "dual" Grassmann kernel.
This is needed for single-orbital problems in real and imaginary time.
"""

import numpy as np
from .many_body_operator import indices_odd_and_even, idx_sign_under_reverse


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



def overlap_signs(kernel_dual: np.ndarray) -> np.ndarray:

    """
    Adjust the DUAL kernel by the signs that result from bringing the path integral into overlap form.
    Valid for single-orbital problems in real and imaginary time.
    This alters the signs of all entries in the rows with indices 2 and 3 and all entries in the columns with indices 1 and 3.

    Parameters:
    - kernel_dual (np.ndarray): A 4x4 matrix representing the dual kernel.

    Returns:
    - np.ndarray: The adjusted dual kernel.
    """
    #check size of input
    assert kernel_dual.shape == (4,4), "The input must be a 4x4 matrix."

    kernel_dual[2,:] *= -1 #sign from substitution for spin down on right side of gate
    kernel_dual[:,1] *= -1 #sign from substitution for spin up on left side of gate
    kernel_dual[3,:] *= -1 #sign from substitution for spin down on right side of gate
    kernel_dual[:,3] *= -1 #sign from substitution for spin up on left side of gate

    return kernel_dual


def imaginary_i_for_global_reversal(kernel: np.ndarray) -> np.ndarray:
    """
    Adjust the kernel by the factors of imaginary i needed to determine the sign associated with 
    the reversal of a global string. This function uses vectorized operations for efficiency.

    Parameters:
    - kernel (np.ndarray): A matrix kernel.

    Returns:
    - np.ndarray: The adjusted kernel, with each row multiplied by (1j)**n, where n is the
      number of 1's in the binary representation of the row index.
    """
    # Ensure the array is of a complex type to handle complex operations
    kernel = kernel.astype(np.complex128)
    
    # Number of rows in the kernel
    dim_kernel = kernel.shape[0]

    # Pre-compute the power of 1.j for each row index
    # Create an array of row indices
    row_indices = np.arange(dim_kernel)

    # Count the number of set bits (1s) in each row index
    # np.unpackbits works on uint8, thus ensure dtype is uint8 and reshape after unpacking
    bit_counts = np.unpackbits(row_indices.astype(np.uint8)[:, np.newaxis], axis=1).sum(axis=1)

    # Compute the factor (1.j) raised to the number of 1s in the binary representation of each index
    factors = (1.j) ** bit_counts

    # Apply the computed factors to each row
    kernel *= factors[:, np.newaxis]  # Make factors a column vector to broadcast along rows

    return kernel

def inverse_imaginary_i_for_global_reversal(kernel: np.ndarray) -> np.ndarray:
    """
    Inverse operation of the function 'imaginary_i_for_global_reversal'.
    This is achieved by a triple application of the function 'imaginary_i_for_global_reversal'.

    Parameters:
    - kernel (np.ndarray): A matrix kernel.

    Returns:
    - np.ndarray: The adjusted kernel.
    """

    # Ensure the array is of a complex type to handle complex operations
    kernel = kernel.astype(np.complex128)
    
    # Apply imaginary_i_for_global_reversal to the kernel three times
    for _ in range(3):
        kernel = imaginary_i_for_global_reversal(kernel)

    return kernel



def string_in_kernel(kernel: np.ndarray) -> np.ndarray:
    """
    Adjust the kernel by the signs that result from a fermionic Jordan-Wigner string.
    For this, one needs to determine all row indices of the kernel that contain an odd number of Grassmann variables.

    Parameters:
    - kernel (np.ndarray): A matrix representing the kernel.

    Returns:
    - np.ndarray: The adjusted dual kernel.
    """ 
    #log with basis 2
    n_ferms = np.log2(kernel.shape[0]).astype(int)

    #determine indices of rows with odd number of fermionic variables
    indices_odd, _ = indices_odd_and_even(n_ferms)

    #multiply the rows with odd number of fermionic variables with -1
    kernel[indices_odd,:] *= -1

    return kernel


def sign_for_local_reversal(kernel: np.ndarray) -> np.ndarray:
    """
    Adjust the kernel by the factors of -1 needed to reverse the outgoing Grassmann variables for a local gate.

    Parameters:
    - kernel (np.ndarray): A matrix kernel.

    Returns:
    - np.ndarray: The adjusted dual kernel, with each row that changes sign under local reversal multiplied by -1.
    """

    #determine the indices of the rows that change sign under local reversal
    indices_sign_change = idx_sign_under_reverse(kernel.shape[0])

    #multiply the rows that change sign under local reversal with -1
    kernel[indices_sign_change,:] *= -1

    return kernel
    
    

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
        kernel_dual[[1,3],:] *= -1
        kernel_dual[:,[1,3]] *= -1


    #include the string in the kernel 
    if string == True:
        kernel_dual = string_in_kernel(kernel_dual)

    #swap the order of the variables in the outgoing space (corresponds to sign changes in corresponding rows of kernel)
    kernel_dual = sign_for_local_reversal(kernel_dual)

    #apply the factors of imaginary i needed to determine the sign associated with the reversal of a global string
    kernel_dual = imaginary_i_for_global_reversal(kernel_dual)
   
    return kernel_dual

