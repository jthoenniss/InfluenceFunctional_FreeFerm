"""
Module to convert a single-body operator to a many-body gate.
"""

import numpy as np
#to use type hints
from typing import Tuple, List
    
def many_body_operator(output_ferms: list, input_ferms: list) -> np.ndarray:
    """
    Function to convert a fermionic operator to a many-body gate. 
    The operator is given in the form of two lists, each containing the indices of the fermionic operators.
    The fermions are ordered as c_1^\dag, c_2^\dagger,..c_1, c_2,.. (increasing order).

    Parameters:
    - output_ferms (list or np.ndarray): List containing the indices of the fermionic operators in the output state in increasing order
    - input_ferms (list or np.ndarray): List containing the indices of the fermionic operators in the input state in increasing order.

    Returns:
    - np.ndarray: The many-body gate representing the operator.
                  Note: the output gate is in the convention:
                  gate = v^\dagger @ gate_coeffs @ v, with: v = [<0|, <0|c_{n_ferm} , <0|c_{n_ferm -1}, <0|c_{n_ferm}c_{n_ferm -1}...]

    Example:
    For the operator c_1^\dag c_3^\dagger c_2 c_4, the arguments are:
    output_ferms = [1,0,1,0] (describing the 'daggered' variables, i.e. corresponding to  c_1^\dag c_3^\dagger)
    input_ferms = [0,1,0,1] (describing the 'non-daggered' variables, i.e. corresponding to c_2 c_4)


    Note: The derivation is as follows (here for illustration for 6 fermion flavors in the system):
    - Take a given operator, i.e. c_2^\dag c_6^\dag c_4 c_5.
    - Insert identity between the 'daggered' and 'non-daggered' variables: c_2^\dag c_6^\dag * Identity *  c_4 c_5. (*)
    - Identity = sum_i |i><i|, where i is a binary string of length n.
    - For all states |i> that contain the fermion 2, 4, 5, or 6, the matrix element is zero. (**)
    - The remaining terms are:  c_2^\dag c_6^\dag                   |0><0|         c_4 c_5, 
                                c_2^\dag c_6^\dag          (c_1^\dag|0><0|c_1)     c_4 c_5, 
                                c_2^\dag c_6^\dag          (c_3^\dag|0><0|c_3 )    c_4 c_5, 
                                c_2^\dag c_6^\dag (c_1^\dag c_3^\dag|0><0|c_3 c_1) c_4 c_5.
    - The fermions need to be brought into the correct order.  (***)
    - For the output space, we move the output_ferms to the right into the string from the output fermions of state_bin (*** a)
    - For the input space, we first determine the sign associated with reversal of the order of the daggered fermions from state_bin, such that all input fermions are in increasing order. (*** b)
        Note that the reversed bit string <i| for the input state is simply the string associated with |i>
    - Then, we move the fermions from state_bin to the right into the correct position in the string input_ferms and count the minus signs picked up (*** c)
    - Then, invert all input variables, such that they are in increasing order. (*** d)

    For n_ferm = 2, the annihilation operators are:
    c_1 = [[0.+0.j 0.+0.j 1.+0.j 0.+0.j]
            [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
            [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
            [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]

    c_2 = [[ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
            [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
            [ 0.+0.j  0.+0.j  0.+0.j -1.+0.j]
            [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]

    In the notes for imaginary time, c_1 = c_down and c_2 = c_up.
    """

    #if input and output fermion are lists, convert to numpy arrays to allow for advanced indexing:
    output_ferms = np.array(output_ferms)
    input_ferms = np.array(input_ferms)

    # Determine the number of fermions in the system
    n_ferms = len(output_ferms)
    dim = 2**n_ferms #dimension of local Hilbert space

    # Initialize the many-body gate
    many_body_gate = np.zeros((dim, dim), dtype=np.complex_)
    
    #iterator through all elements of the identity. (*)
    for state in range (dim):
        #binary representation of state:
        state_bin = np.array(list(map(int, np.binary_repr(state, width=n_ferms))))

        #check if any fermions that are contained in 'state' also appear in input_ferms or output_ferms.
        #If yes, the matrix element is zero. (**)
        if np.any(np.logical_and(state_bin, output_ferms)) or np.any(np.logical_and(state_bin, input_ferms)):
            continue

        #otherwise, initialize the sign of the matrix element
        sign = 1

        # determine the sign that we get by ordering all output space variables (daggered) correctly (*** a)
        sign_changes = (-1) ** np.cumsum(state_bin) #array of minus signs that we pick up when moving the output_ferms to the right into the string or state
        sign *= np.prod(sign_changes[output_ferms == 1])

        # compute sign from reversing the order of input fermions from 'state' (*** b)
        #change sign if an odd number of fermions pairs are in the state (for 2, 3, 6, 7, 10, 11, ...)
        if (np.sum(state_bin)//2)%2 == 1: 
            sign *= -1

        # determine the sign that we get by ordering all input space variables (non-daggered) correctly (*** c)
        sign_changes = (-1) ** np.cumsum(input_ferms) #array of minus signs that we pick up when moving the the input fermions from state_bin into the string input_ferms
        sign *= np.prod(sign_changes[state_bin == 1])

        # Compute the new states in input and output space
        input_state = np.add(state_bin, input_ferms) 
        output_state = np.add(state_bin, output_ferms)

        # change sign if an odd number of fermions pairs are in the final input state (for 2, 3, 6, 7, 10, 11, ...)
        if (np.sum(input_state)//2)%2 == 1:
            sign *= -1
        
        #convert input and output state to integer index
        idx_out = np.dot(output_state, 2**np.arange(n_ferms)[::-1])
        idx_in = np.dot(input_state, 2**np.arange(n_ferms)[::-1])

        #add the matrix element to the many-body gate
        many_body_gate[idx_out, idx_in] += sign

    return many_body_gate



def annihilation_ops(n_ferms: int) -> list:
    """
    Function to generate a list of annihilation operators for a given number of fermions.

    Parameters:
    - n_ferms (int): The number of fermions in the system.

    Returns:
    - list: A list of annihilation operators for the fermions in the system.
    """

    # Initialize the list of annihilation operators
    annihilation_operators = []

    # Loop over the fermions and add the annihilation operator for each fermion to the list
    for i in range(n_ferms):
        output_ferms = [0] * n_ferms #no daggered variables, i.e. all output fermions have index 0.
        input_ferms = [0] * n_ferms
        input_ferms[i] = 1 #set i-th input fermion to 1, this corresponds to the annihilation operator for flavor i.

        #create many-body operator and append to list
        annihilation_operators.append(many_body_operator(output_ferms=output_ferms, input_ferms=input_ferms))

    return annihilation_operators



def idx_sign_under_reverse(kernel_dim: int) -> list:
    """
    Determine the indices of many-body basis states which change sign under reversal of the order of the fermions.
    These are all states which contain an odd number of fermions pairs (and/or one an additional fermion).
    Parameters:
    - kernel_dim: The dimension of the kernel. For valid many-body operator a power of 2.

    Returns:
    - list: A list of indices of many-body basis states which change sign under reversal of the order of the fermions.
    """
    
    return [i for i in range(kernel_dim) if (bin(i).count('1')//2)%2 == 1]
    


def indices_odd_and_even(dim_kernel: int) -> Tuple[List[int], List[int]]:
    """
    Determine the indices of many-body basis states which contain an odd and an even number of fermions, respectively.
    This is equivalent to those integers whose binary representations contain an odd/even number of '1'.
    Parameters:
    - dim_kernel (int): Dimension of kernel. For valid many-body operator a power of 2.

    Returns:
    - Tuple: A tuple of two lists of indices of many-body basis states which contain an odd and an even number of fermions, respectively.
    """

    #binary_representation of dim_kernel
    num_ones = np.array([bin(num).count('1') for num in range(dim_kernel)])
  
    # Determine odd and even indices based on the parity of the number of 1s
    indices_odd = np.where(num_ones % 2 == 1)[0].tolist()
    indices_even = np.where(num_ones % 2 == 0)[0].tolist()

    return indices_odd, indices_even


def fermion_parity(operator: np.ndarray) -> int:
    """
    Function to determine the parity of a fermionic operator fermionic many-body basis or a "dual" Grassmann kernel.
    Parameters:
    - operator (np.ndarray): A fermionic operator or a "dual" Grassmann kernel.
    Returns:
    - int: The parity of the operator. Returns 1 if the operator is even, and -1 if the operator is odd.
    """

    indices_odd, indices_even = indices_odd_and_even(dim_kernel = operator.shape[0])

    even_sum = np.sum(np.abs(operator[indices_even][:, indices_even])) + np.sum(np.abs(operator[indices_odd][:, indices_odd]))
    odd_sum = np.sum(np.abs(operator[indices_even][:, indices_odd])) + np.sum(np.abs(operator[indices_odd][:, indices_even]))

    assert even_sum == 0 or odd_sum == 0, 'operator has no definite parity'
    return (-1 if odd_sum != 0 else 1) 



if __name__ == "__main__":

    annihilation_ops_twosite = annihilation_ops(n_ferms = 2)
    print("The annihilation operators for 2 sites are:")
    print("c_1 = ")
    print(annihilation_ops_twosite[0])
    print("c_2 = ")
    print(annihilation_ops_twosite[1])

    #Test index_signs_under_reverse for 4 sites
    idx_signs_foursite = idx_sign_under_reverse(kernel_dim= 2**4)
    print("The idx_signs_under_reverse for 4 sites are:")
    print(idx_signs_foursite)

    #Test index_signs_under_reverse for 2  sites
    idx_signs_twosite = idx_sign_under_reverse(kernel_dim= 2**2)
    print("The idx_signs_under_reverse for 2 sites are:")
    print(idx_signs_twosite)

    print([i for i in range(2**4) if (bin(i).count('1')//2)%2 == 1])
