#Module to convert the input matrix B to a many-body state.

import numpy as np
from pfapack import pfaffian as pf


def IF_many_body(B):
    """
    Computes the coefficients of the many-body wavefunction based on the Slater determinant.
    
    Parameters:
    - B (numpy.ndarray): A square numpy array representing the matrix whose submatrices' Pfaffians
      determine the coefficients of the many-body wavefunction. 

    Returns:
    - numpy.ndarray: An array of complex numbers representing the coefficients of the many-body
      wavefunction in binary order. The element coeffs[0] thus desribes the coefficient of the vacuum state,
        coeffs[1] desribes the coefficient for the state with binary index [0,0,0,...,1], etc.
        Note: Only entries with an even number of '1's will be non-zero as the IF if Gaussian so all Grassmann variables come in pairs.
    """
    # Validate input
    if not isinstance(B, np.ndarray) or len(B.shape) != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("Input B must be a square numpy array.")

    n_qubits = B.shape[0]#number of variables in the exponent. Note that this is twice the number of time steps as each time step and an ingoing and an outgoing variable.
    len_IM = 2**n_qubits  # Total number of entries of the IF viewed as a MB-wavefunction
    coeffs = np.zeros(len_IM, dtype=np.complex_)  # Store the components of the (unnormalized) many-body wavefunction
    coeffs[0] = 1  # Initialize the first coefficient: The vaccum state has coefficient 1.

    for i in range(1, len_IM):
        # Binary representation of the wavefunction index to determine the submatrix of B.
        # The binary representation is padded with zeros from the left to have the correct number of bits.
        binary_index = np.binary_repr(i, width=n_qubits)  # e.g. '0011' for i = 3 with n_qubits = 4

        # Indices of '1's in binary_index indicate which rows/columns of B to use.
        read_out_indices = [j for j, bit in enumerate(binary_index) if bit == '1'] # This array will look something like e.g. [0, 2, 3] for binary_index = '1011'

        if len(read_out_indices) % 2 == 0:
            #compute the submatrix of B based on the positions of the '1's in the binary representation
            B_sub = B[np.ix_(read_out_indices, read_out_indices)]
            # compute the pfaffian of the submatrix and store it as the coefficient of the many-body wavefunction
            coeffs[i] = pf.pfaffian(B_sub)

    return coeffs

