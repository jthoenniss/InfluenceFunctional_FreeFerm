"""
Script containing all necessary functions to compute the exact many-body state overlap of IF-MPS -- impurity-MPO -- IF-MPS
"""

import numpy as np

import os,sys
parent_dir = os.path.join(os.path.dirname(__file__),"../../..")
sys.path.append(parent_dir)
from src.shared_modules.many_body_operator import annihilation_ops
from src.real_time.compute_IF.real_time_IF import read_IF
from src.shared_modules.IF_many_body import IF_many_body
from src.real_time.single_orbital.generate_MPO import impurity_MPO
from src.shared_modules.Keldysh_contour import position_to_Keldysh_idx

from scipy.linalg import expm





def compute_propagator(IF_MB: np.ndarray, U_evol: np.ndarray, init_density_matrix: np.ndarray, operator_0, operator_tau, nbr_time_steps: int) -> np.ndarray:
    """
    Compute the propagator of the impurity model based on the many-body wavefunction of the influence functional and the time-evolution operator of the impurity model.

    Parameters:
    - IF_MB (numpy.ndarray): An array of complex numbers representing the many-body wavefunction of the influence functional.
    - U_evol (numpy.ndarray): A (4x4) numpy array representing the 'dual' and 'sign-adjusted' time-evolution operator defined by the impurity Hamiltonian.
    - init_density_matrix (numpy.ndarray): A (4x4) numpy array representing the initial density matrix of the impurity model in the fermionic operator basis
    - nbr_steps (int): The number of time steps for which we want to compute the propagator.
    - operator_0 (numpy.ndarray): A (4x4) numpy array representing the operator at time 0. Expected to be odd in Grassmann variables, e.g. a creation/anihilation operator. Otherwise strings below should be adapted
    - operator_tau (numpy.ndarray): A (4x4) numpy array representing the operator at time tau. Expected to be odd in Grassmann variables, e.g a creation/anihilation operator. Otherwise strings below should be adapted

    Returns:
    - numpy.ndarray: A vector representing the propagator of the impurity model, <operator_tau(tau) operator_0(0)>.
    """

    G_up_up_ff = []
    #generate impurity MPO
    #compute the Keldysh index corresponding to the initial time point
    Keldysh_idx_0 = position_to_Keldysh_idx(0, 'f', nbr_time_steps)
    for tau in range(nbr_time_steps):
        #compute the Keldysh index corresponding at time tau
        Keldysh_idx_tau = position_to_Keldysh_idx(tau, 'f', nbr_time_steps)
        operator_a = operator_0
        operator_b = operator_tau  @ U_evol if tau > 0 else operator_tau
        MPO = impurity_MPO(U_evol = U_evol, initial_density_matrix=init_density_matrix, nbr_time_steps=nbr_time_steps, operator_a=operator_a, Keldysh_idx_a=Keldysh_idx_0, operator_b=operator_b, Keldysh_idx_b=Keldysh_idx_tau)

        #construct the explicit many-body representation of the MPO
        #partition sum:
        MPO_Z = MPO["init_state_Z"]#initialize MPO with initial state
        for gate in MPO["gates_Z"]:
            MPO_Z = np.kron(MPO_Z, gate)
        MPO_Z = np.kron(MPO_Z, MPO["boundary_condition_Z"])#add final state

        #MPO with operators:
        MPO_operator = MPO["init_state"]#initialize MPO with initial state
        for gate in MPO["gates"]:
            MPO_operator = np.kron(MPO_operator, gate)
        MPO_operator = np.kron(MPO_operator, MPO["boundary_condition"])#add final state

        #compute the propagator
        Z = IF_MB @ MPO_Z @ IF_MB#parition sum
        propag = IF_MB @ MPO_operator @ IF_MB#unnormaized propagator

        propag = propag / Z #normalize the propagator

        G_up_up_ff.append(propag)

    return G_up_up_ff






if __name__ == "__main__":
    #Set parameters:
    delta_t = 0.1 #time step
    E_up = 3 #energy of the up fermion
    E_down = 4 #energy of the down fermion
    t_spinhop = 5 #spin hopping term
    nbr_time_steps = 3 #number of time steps in the influence functional

    #array containing the many-body representations of all annihilation operators for two fermions species
    c_down, c_up = annihilation_ops(n_ferms=2)


    #read out matrix B from file
    filename = '/Users/julianthoenniss/Documents/PhD/code/InfluenceFunctional_FreeFerm/data/benchmark_delta_t=0.1_Tren=5_globalgamma=1.0_beta=50.0_T=3'

    #read the influence matrix B from disk
    B = read_IF(filename)

    #convert the influence matrix to a many-body state
    IF_MB = IF_many_body(B)

    #initial density matrix of the system
    init_density_matrix = np.eye(4)

    Ham_onsite = E_down * c_down.T.conj() @ c_down + E_up * c_up.T.conj() @ c_up 
    Ham_spinhop = t_spinhop* (c_up.T @ c_down + c_down.T @ c_up)
    U_evol =  expm(1j*Ham_onsite * delta_t)  @ expm(1j * Ham_spinhop * delta_t)#time-evolution operator of the impurity model

    #compute the propagator
    G_upup_ff = compute_propagator(IF_MB, U_evol, init_density_matrix, operator_0=c_up.T.conj(), operator_tau= c_up  , nbr_time_steps=nbr_time_steps)

    print("Dual Overlap -- Forward-forward propagator for spin up:")
    for tau in range (len(G_upup_ff)):
        print(f'G_upup_ff({tau}) = {G_upup_ff[tau]}')


   
