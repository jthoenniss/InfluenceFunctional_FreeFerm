"""
Module that contains the necessary functions to deal with the 'dual' kernel of the density matrix.
"""

import numpy as np

#import parent directory
import os,sys
parent_dir = os.path.join(os.path.dirname(__file__),"../../..")
#append parent directory to path
sys.path.append(parent_dir)
from src.shared_modules.dual_kernel import overlap_signs, operator_to_kernel, inverse_dual_kernel, inverse_imaginary_i_for_global_reversal, sign_for_local_reversal, transform_backward_kernel
from src.real_time.compute_impurity_gate.interleave_gate import interleave, dict_interleave


def half_evolve_dual_density_matrix(dual_density_matrix: np.ndarray, step_type: str = "imp") -> np.ndarray:
    """
    Evolve the dual density matrix by half a time step with trivial evolution.
    It can either be evolved by a trivial (vacuum) IF-step or by a trivial (vacuum) impurity gate.

    Parameters:
    - dual_density_matrix (np.ndarray): A 4x4 matrix representing the 'dual' kernel of the density matrix.
    - step_type (str): The type of time step that is used. Must be set to 'imp' for the impurity gate or 'IF' for the IF-step.
    
    Returns:
    - np.ndarray: The evolved dual density matrix.
    """

    #check size of dual density matrix
    assert dual_density_matrix.shape == (4,4), "The input must be a 4x4 matrix."

    if step_type == "IF":
        #The many-body wavefunction of the vaccum state, reshaped to a 4x4 matrix:
        #the diagonal entries correspond to (in this order): vaccum, overlap backward, overlap forward, both overlaps
        coeffs_single_time_vac = np.diag([1, 1, -1, 1])

        #evolve the dual density matrix by half a time step using the trivial 1-step IF of the vaccum
        dual_density_matrix = np.einsum('ij,ik,kl->jl', coeffs_single_time_vac, dual_density_matrix, coeffs_single_time_vac, optimize=True)

    elif step_type == "imp":
        #trivial impurity gate:
       
        id_fw = operator_to_kernel(np.eye(4), branch='f')#identity kernel on forward branch
        id_bw = operator_to_kernel(np.eye(4), branch='b')#identity kernel on backward branch
        dual_impurity_gate_trivial = interleave(id_fw, id_bw, mapping = dict_interleave(4)) #find corresponding tensor 

        #reshape
        dual_impurity_gate_trivial = dual_impurity_gate_trivial.reshape(4,4,4,4)

        #evolve the dual density matrix by half a time step using the trivial impurity gate
        dual_density_matrix = np.einsum('ik,ijkl->jl', dual_density_matrix, dual_impurity_gate_trivial, optimize=True)

    return dual_density_matrix


def dual_density_matrix_to_operator(dual_density_matrix: np.ndarray, step_type: str = 'full') -> np.ndarray:
    """
    Converts the 'dual' (and sign-adjusted) kernel of the a density matrix of the corresponding operator in the fermionic basis.
    This conversion depends on the step type that is used. For the full time step, the conversion is straightforward.
    For the half time step, the conversion is started by evolving the density matrix by half a time step using the trivial 1-step IF of the vaccum.

    Parameters:
    - dual_density_matrix (np.ndarray): A 4x4 matrix representing the 'dual' kernel of the density matrix.
    - step_type (str): The type of time step that is used. Must be set to 'full' or 'half'.

    Returns:
    - np.ndarray: The matrix of the corresponding operator in the fermionic basis.
    """
    
    #check size of dual density matrix
    assert dual_density_matrix.shape == (4,4), "The input must be a 4x4 matrix."

    if step_type == 'half':#if the dual density matrix corresponds to a half time step, evolve by trivial impurity gate
        dual_density_matrix = half_evolve_dual_density_matrix(dual_density_matrix, step_type='imp')

    #Convert dual density matrix to density matrix:

    # undo factors of imaginary i
    dual_density_matrix = inverse_imaginary_i_for_global_reversal(dual_density_matrix)

    # swap the order of the variables in the outgoing space (corresponds to sign changes in corresponding rows of kernel)
    dual_density_matrix = sign_for_local_reversal(dual_density_matrix)
    
    # invert trafo from backward branch which is the same as forward trafo
    dual_density_matrix = transform_backward_kernel(dual_density_matrix)

    # apply inverse of 'overlap_signs' to the kernel which is the same as forward trafo
    dual_density_matrix = overlap_signs(dual_density_matrix)

    # apply inverse of 'dual_kernel' to the kernel
    density_matrix = inverse_dual_kernel(dual_density_matrix)

    return density_matrix


if __name__ == "__main__":

    #set up a random 4x4 density matrix
    density_matrix = np.random.rand(4,4)

    #compute the dual density matrix
    dual_density_matrix = operator_to_kernel(density_matrix, branch='b')

    #transform the dual density matrix back to the density matrix 
    density_matrix_recover = dual_density_matrix_to_operator(dual_density_matrix=dual_density_matrix, step_type='full')

    #check if it is the same as the original density matrix
    print("recovered DM: \n", np.real(density_matrix_recover))
    print("original DM: \n", density_matrix)
    print("recovery for full time step successful: ", np.allclose(density_matrix_recover, density_matrix))


    print("Testing the half time step evolution of the dual density matrix")
    print(dual_density_matrix)
    rho_half = half_evolve_dual_density_matrix(dual_density_matrix, step_type='IF')
    rho_full = half_evolve_dual_density_matrix(rho_half, step_type='imp')
    print("zero ",dual_density_matrix)
    print("half ",rho_half)
    print("half-step wise evolution successful: ", np.allclose(rho_full, dual_density_matrix))
    print("full ", rho_full)
