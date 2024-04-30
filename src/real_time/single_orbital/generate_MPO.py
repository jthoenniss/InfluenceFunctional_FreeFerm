"""
Module to compute the impurity MPO for given observables/computables and a given evolution Hamiltonian
"""
import numpy as np
import os, sys
parent_dir = os.path.join(os.path.dirname(__file__), "../../..")
sys.path.append(parent_dir)
from src.real_time.compute_impurity_gate.interleave_gate import interleave, dict_interleave
from src.shared_modules.dual_kernel import operator_to_kernel
from src.shared_modules.many_body_operator import fermion_parity
from src.shared_modules.Keldysh_contour import position_to_Keldysh_idx, Keldysh_idx_to_position
from typing import List

def adjust_string(string: List[bool], Keldysh_idx: int, parity: int) -> List[bool]:
    """
    Adjust the string along the Keldysh contour based on the parity of an operator.
    The function flips all boolean value between the first index and the index of the operator with odd parity.

    Parameters:
    - string (List[bool]): A list of booleans that indicate which gates must be modified due to the presence of a string.
    - Keldysh_idx (int): The index on the unfolded Keldysh contour of the operator.
    - parity (int): The parity of the operator. Must be +1 for even operators and -1 for odd operators.
    """
    

    if parity == -1: # the string is only affected by odd operators

        nbr_time_steps = len(string)//2 - 1

        #determine the Kelddysh index of the operator   
        time_point, branch = Keldysh_idx_to_position(Keldysh_idx, nbr_time_steps)

        #up to the time point, flip all boolean values.
        for time in range (time_point):
            #forward
            idx = position_to_Keldysh_idx(time, 'f', nbr_time_steps)
            if parity == -1:
                string[idx] = not string[idx]
            #backward
            idx = position_to_Keldysh_idx(time, 'b', nbr_time_steps)
            if parity == -1:
                string[idx] = not string[idx]
        #for the time point itself, flip the boolean value corresponding to the forward gate if the operator is on the backward branch
        if branch == 'b':
            idx = position_to_Keldysh_idx(time_point, 'f', nbr_time_steps)
            if parity == -1:
                string[idx] = not string[idx]
        
        return string
    
    else:
        return string
    


def operator_at_Keldysh_idx(Keldysh_idx: int, U_evol: np.ndarray, operator_a: np.ndarray, Keldysh_idx_a: int, operator_b: np.ndarray, Keldysh_idx_b: int, nbr_time_steps: int) -> np.ndarray:
    """
    Determine the operator at a given Keldysh index on the unfolded Keldysh contour.
    Returns the evolution gate if no operator is present at the given Keldysh index.
    Otherwise, returns the operator at the given Keldysh index.

    Parameters:
    - Keldysh_idx (int): The index on the unfolded Keldysh contour.
    - U_evol (numpy.ndarray): An (4x4) array of complex numbers representing the evolution operator of the impurity model.
    - operator_a (numpy.ndarray): A (4x4) numpy array representing the operator at position_a on the Keldysh contour.
    - Keldysh_idx_a (int): an index on the unfolded Keldysh contour.
    - operator_b (numpy.ndarray): A (4x4) numpy array representing the operator at position_b on the Keldysh contour.
    - Keldysh_idx_b (int): an index on the unfolded Keldysh contour.
    - nbr_time_steps (int): The total number of time steps in the influence functional.

    Returns:
    - numpy.ndarray: The operator at the given Keldysh index.
    """
    #determine the time point and branch of the Keldysh index
    time_point, branch = Keldysh_idx_to_position(Keldysh_idx, nbr_time_steps)

   
    if Keldysh_idx == Keldysh_idx_a and Keldysh_idx == Keldysh_idx_b: #if both operators are at the same Keldysh index
        return operator_b @ operator_a #operator b is always behind operator a on the unfolded Keldysh contour
    elif Keldysh_idx == Keldysh_idx_a: #if idx coincides with the Keldysh index of operator a
        return operator_a
    elif Keldysh_idx == Keldysh_idx_b:#if idx coincides with the Keldysh index of operator b
        return operator_b

    #If no operator is inserted at the given Keldysh index, return the evolution operator (except for the initial and final time points)
    else:
        if time_point != 0 and time_point != nbr_time_steps:#if the Keldysh index is not the initial or final time point (which require special care)
            if branch == 'f':
                return U_evol#return evolution gate for bulk time step if no operator is inserted here.
            else:
                return U_evol.T.conj()#return evolution gate (hermitian conjugate) for bulk time step if no operator is inserted here.
        else:
            return np.eye(U_evol.shape[0])#return identity gate for final and initial time step if no operator is inserted here.

def impurity_MPO(U_evol: np.ndarray, initial_density_matrix: np.ndarray, nbr_time_steps: int, operator_a: np.ndarray = np.eye(4), Keldysh_idx_a: int = 0, operator_b: np.ndarray = np.eye(4), Keldysh_idx_b: int = 0):
    """
    Function to compute the impurity gate for a given set of operators and evolution Hamiltonian for the single-orbital case. 
    The function computes and the impurity gates with and without operator insertions. 

    Parameters:
    - U_evol (numpy.ndarray): An (4x4) array of complex numbers representing the evolution operator of the impurity model.
    - initial_density_matrix (numpy.ndarray): A (4x4) numpy array representing the initial density matrix of the impurity model in the fermionic operator basis.
    - nbr_time_steps (int): The total number of time steps in the influence functional.
    - operator_a (numpy.ndarray): A (4x4) numpy array representing the operator at position_a on the Keldysh contour. Default: Identity gate.
    - Keldysh_idx_a (int): an index on the unfolded Keldysh contour. Default value 0 (time point zero on forward branch, in density matrix)
    - operator_b (numpy.ndarray): A (4x4) numpy array representing the operator at position_b on the Keldysh contour. Default: Identity gate.
    - Keldysh_idx_b (int): an index on the unfolded Keldysh contour.  Default value 0 (time point zero on forward branch, in density matrix)
    

    Returns:
    - dict: A dictionary containing the impurity gates with and without operator insertions, the global sign, and the initial and final states of the impurity model.
            The keys are:
            - 'boundary_condition': The boundary condition of the impurity gate.
            - 'init_state': The initial state of the impurity model.
            - 'gates': The impurity gates with operator insertions.
            - 'global_sign': The global sign of the impurity gate.
            - 'boundary_condition_Z': The boundary condition of the impurity gate without operator insertions.
            - 'init_state_Z': The initial state of the impurity model without operator insertions.
            - 'gates_Z': The impurity gates without operator insertions.

    
   
    """
    #Validate input
    assert isinstance(nbr_time_steps,int) and nbr_time_steps > 0, 'Total time must be a positive integer'
    assert U_evol.shape[0] ==4 and U_evol.shape[0] == U_evol.shape[1] and U_evol.shape == operator_a.shape  and U_evol.shape == operator_b.shape , 'Dimensions of specified operators incompatible. '
    #convert Keldysh indices to time points and branches
    time_point_a, branch_a = Keldysh_idx_to_position(Keldysh_idx_a, nbr_time_steps)
    time_point_b, branch_b = Keldysh_idx_to_position(Keldysh_idx_b, nbr_time_steps)
    
    assert time_point_a <= nbr_time_steps and time_point_b <= nbr_time_steps, 'Time points of operators must be less or equal to total time'
    
    #determine the parity of the operators
    parity_a = fermion_parity(operator_a) #is +1 for even operators and -1 for odd operators
    parity_b = fermion_parity(operator_b) #is +1 for even operators and -1 for odd operators

    #create array with booleans that contains "True" for all indices on the unfolded Keldysh contour that lies between two operators with odd parity
    string = np.zeros(2*nbr_time_steps+2, dtype=bool)
    string = adjust_string(string, Keldysh_idx_a, parity_a)
    string = adjust_string(string, Keldysh_idx_b, parity_b)

    #create mapping for interleave function
    mapping = dict_interleave(bin_length = U_evol.shape[0])

    #__________Gate with operator insertions__________
    #Define lambda function that returns the operator at a given Keldysh index (inserted operator or evolution operator)
    operator = lambda idx: operator_at_Keldysh_idx(idx, U_evol, operator_a, Keldysh_idx_a, operator_b, Keldysh_idx_b, nbr_time_steps)
    
    #Boundary condition with antiperiodic boundary condition
    idx_final_fw = position_to_Keldysh_idx(nbr_time_steps, 'f', nbr_time_steps = nbr_time_steps)
    idx_final_bw = position_to_Keldysh_idx(nbr_time_steps, 'b', nbr_time_steps = nbr_time_steps)
    MPO_boundary_condition = operator_to_kernel(operator(idx_final_bw) @ operator(idx_final_fw), branch = 'f', boundary = True)
    
    #Initial state
    idx_0_fw = position_to_Keldysh_idx(time_point=0, branch = 'f', nbr_time_steps = nbr_time_steps)
    idx_0_bw = position_to_Keldysh_idx(time_point=0, branch = 'b', nbr_time_steps = nbr_time_steps)
    MPO_init_state = operator_to_kernel(operator(idx_0_fw) @ initial_density_matrix @ operator(idx_0_bw), branch='b', string=string[idx_0_bw]) 

    #Impurity gates with operator insertions
    MPO_gates = np.zeros((nbr_time_steps-1, U_evol.shape[0]**2, U_evol.shape[0]**2), dtype = np.complex128)#holds interleaved gates
    #iterate through time points:
    for time in range(1,nbr_time_steps):
        idx_fw = position_to_Keldysh_idx(time, 'f', nbr_time_steps)
        idx_bw = position_to_Keldysh_idx(time, 'b', nbr_time_steps)
        kernel_fw = operator_to_kernel(operator(idx_fw), string=string[idx_fw], branch='f')
        kernel_bw = operator_to_kernel(operator(idx_bw), string=string[idx_bw], branch='b')
        MPO_gates[time - 1] = interleave(kernel_fw, kernel_bw, mapping=mapping)
    
    #determine global sign:
    global_sign = 1
    if branch_a == branch_b and branch_a == 'b':#if both operators are on the backward branch
        if time_point_a != time_point_b and parity_a == -1 and parity_b == -1:#sign flips if both operators have odd parity and are not inserted at the same point
            global_sign *= -1
    elif branch_a != branch_b:#if the operators are on different branches
        #flip sign if the operator in the forward branch has a smaller time_step than the one on the backward branch
        if branch_a == 'f' and time_point_a < time_point_b:
            global_sign *= -1
        elif branch_b == 'f' and time_point_b < time_point_a:
            global_sign *= -1


    return {'boundary_condition': MPO_boundary_condition, 'init_state': MPO_init_state, 'gates': MPO_gates, 'global_sign': global_sign}
           