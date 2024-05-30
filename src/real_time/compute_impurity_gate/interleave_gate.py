"""
Module that contains various functions needed to construct the interleaved impurity gates.
Needed for single- and multi-orbitals problems in real time. 
"""

import numpy as np
from typing import Tuple# for type hints


def map_interleave(idx_int: int, bin_length: int) -> Tuple[int, int]:
    """
    Generates the map between indices for the interleaving of two gates. 
    For example, if a string of Grassmann variables 'a b c d e f g h' is reordered to 'a e b f c g d h',
    this function determines the new index and the sign change due to fermionic anticommutation properties.

    Parameters:
    - idx_int (int): Original basis element index.
    - bin_length (int): Length of the binary representation.

    Returns:
    - Tuple[int, int]: New index after reordering and the associated sign.


    Example of how the sign is computed:
    consider a bit string, e.g. [a,b,c,d,e,f,g,h].
    Divide into two: [a,b,c,d] and [e,f,g,h]
    Move the first three entries from second half to the left past the last entry from first half:
    new string: [a,b,c,e,f,g,d,h]. This comes with a sign if d = 1 and e,f,g contain an odd number of 1s.
    Next, move e and f to the left past c: [a,b,e,f,c,g,d,h]. This comes with a sign if c = 1 and e,f contain an odd number of 1s.
    Finally, move e past b to the left past b. New string: [a,e,b,f,c,g,d,h]. This comes with a sign if b = 1 and e =1.
    The total sign is the product of the individual signs.
    In the list comprehension below, the signs are computed in inverse order from the steps describes here, which doesn't matter as they are all summed.
    """

    # Convert the integer index to a binary string of fixed length
    idx_bin = format(idx_int, f'0{bin_length}b')

    # Split the binary string into two halves
    half_length = bin_length // 2
    first_half, second_half = idx_bin[:half_length], idx_bin[half_length:]

    # Compute the sign change due to the reordering of Grassmann variables
    sign_changes = [first_half[i] == '1' and second_half[:i].count('1') % 2 for i in range(half_length)]
    sign = (-1) ** sum(sign_changes)

    # Interleave the bits from both halves to form the new index
    new_idx_bin = ''.join(a + b for a, b in zip(first_half, second_half))
    new_idx_int = int(new_idx_bin, 2)
    print(sign, "  ",idx_bin, "  ",new_idx_bin)
    return new_idx_int, sign


def dict_interleave(bin_length: int) -> dict:
    """
    Function that generates a dictionary containing the map for all indices needed for the interleaving of two gates.
    
    Parameters:
    - bin_length (int): Length of the binary representation.

    Returns:
    - dict: Dictionary with entries "idx" and "sign" that specifies the order of the rows and columns in the interleaved gate, as well as the signs.
    """

    # Generate the map between the indices for the interleaving of two gates
    map = {"idx": [], "sign": []}
    for idx_int in range(2 ** bin_length):
        new_idx_int, sign = map_interleave(idx_int, bin_length)
        map["idx"].append(new_idx_int)
        map["sign"].append(sign)

    return map

def interleave(forward_gate: np.ndarray, backward_gate: np.ndarray, mapping: dict) -> np.ndarray:
    """
    Interleaves two square matrices (gates) according to a specified mapping and signs.
    Useful for quantum computing simulations involving tensor products and permutation of basis.

    Parameters:
    - forward_gate (np.ndarray): Square matrix representing the gate on the forward branch.
    - backward_gate (np.ndarray): Square matrix representing the gate on the backward branch.
    - mapping (dict): Contains two keys 'idx' and 'sign' where 'idx' is a list of indices specifying the new
      order of the rows and columns, and 'sign' is a list specifying the sign changes due to permutation.

    Returns:
    - np.ndarray: The interleaved gate, which is a square matrix of size (n^2, n^2) where n is the size of the input gates.
    """
    
    # Validate input gates
    if forward_gate.shape[0] != forward_gate.shape[1] or backward_gate.shape[0] != backward_gate.shape[1]:
        raise ValueError("Both gates must be square matrices.")
    if forward_gate.shape != backward_gate.shape:
        raise ValueError("Both gates must have the same dimensions.")
    
    # Compute the Kronecker product
    tensor_prod = np.kron(forward_gate, backward_gate)

    # Initialize the interleaved gate with complex data type
    n_squared = tensor_prod.shape[0]
    interleaved_gate = np.zeros((n_squared, n_squared), dtype=np.complex_)

    # Apply the permutation and signs to the tensor product matrix
    idx, sign = mapping['idx'], mapping['sign']
    for i in range(n_squared):
        for j in range(n_squared):
            interleaved_gate[idx[i], idx[j]] = sign[i] * sign[j] * tensor_prod[i, j]

    return interleaved_gate
