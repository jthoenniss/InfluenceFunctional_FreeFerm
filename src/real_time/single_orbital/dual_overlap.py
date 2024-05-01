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
from src.real_time.single_orbital.dual_density_matrix import dual_density_matrix_to_operator
from scipy.linalg import expm
from typing import Tuple


def evolve_density_matrix_MB(IF_MB: np.ndarray, U_evol: np.ndarray, init_density_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the time-evolved density matrix at all half time steps, i.e. after application of IF and after following application of impurity gate.
    The evolution is computed by explicitly contracting the many-body state with the gates.

    Parameters:
    - IF_MB (numpy.ndarray): An array of complex numbers representing the many-body wavefunction of the influence functional.
    - U_evol (numpy.ndarray): A (4x4) numpy array representing the 'dual' and 'sign-adjusted' time-evolution operator defined by the impurity Hamiltonian.
    - init_density_matrix (numpy.ndarray): A (4x4) numpy array representing the initial density matrix of the impurity model in the fermionic operator basis.
    
    Returns:
    - numpy.ndarray: A list of density matrices at all half time steps, including the original density matrix at time 0.
    """

    #infer the number of timesteps form the length of the many-body representation of the IF
    len_IF = len(IF_MB)
    nbr_time_steps = int(np.log2(len_IF)/4)
    

    #generate MPO (without any operator insertions). In this case the gates for the partition sum coincide with the gates used for the evolution.
    MPO = impurity_MPO(U_evol = U_evol, initial_density_matrix=init_density_matrix, nbr_time_steps=nbr_time_steps)

    #generate full many-body representaton of impurity gate 
    imp_gate = MPO["init_state"]#initialize MPO with initial state
    for gate in MPO["gates"]:
        imp_gate = np.kron(imp_gate, gate)
    imp_gate = np.kron(imp_gate, MPO["boundary_condition"])#add final state

    density_matrices = []#list to hold the density matrices at all time steps
    #successively evolve the initial state:
    for tau in range (0,2*nbr_time_steps):
        
        #reshape such that the ingoing legs (forward and backward) are separate
        IF_MB = IF_MB.reshape(4 ** tau, 4, -1)
        #reshape impurity gate accordingly
        imp_gate = imp_gate.reshape(*IF_MB.shape, *IF_MB.shape)
        #contract the many-body state with the impurity gate, except at one time point
        rho_dual_temp = np.einsum('iaj, ibjkcl, kdl -> abcd', IF_MB, imp_gate, IF_MB, optimize=True)
    
        if tau%2 == 1: #if the uncontracted legs are at a half time step:
            #Connect the forward and backward legs of the uncontracted impurity gate variables
            rho_dual_temp = rho_dual_temp.reshape(4,2,2,2,2,4) 
            rho_dual_temp = np.einsum('ammnnc->ac', rho_dual_temp, optimize=True)/4 #connect forward and backward leg
            #convert the dual density matrix to the density matrix and append it to the list
            density_matrices.append(dual_density_matrix_to_operator(rho_dual_temp, step_type="half"))
          
        else: #if the uncontracted legs are at a full time step:
            #Connect the forward and backward legs of the uncontracted IF variables
            rho_dual_temp = rho_dual_temp.reshape(2,2,4,4,2,2) 
            rho_dual_temp = np.einsum('mmacnn->ac',rho_dual_temp, optimize=True)/4#connect forward and backward leg
            #convert the dual density matrix to the density matrix and append it to the list
            density_matrices.append(dual_density_matrix_to_operator(rho_dual_temp, step_type="full"))

    return np.array(density_matrices)

    

def compute_propagator_MB(IF_MB: np.ndarray, U_evol: np.ndarray, init_density_matrix: np.ndarray, operator_0, operator_tau) -> np.ndarray:
    """
    Compute the propagator of the impurity model based on the many-body wavefunction of the influence functional and the time-evolution operator of the impurity model.

    Parameters:
    - IF_MB (numpy.ndarray): An array of complex numbers representing the many-body wavefunction of the influence functional.
    - U_evol (numpy.ndarray): A (4x4) numpy array representing the 'dual' and 'sign-adjusted' time-evolution operator defined by the impurity Hamiltonian.
    - init_density_matrix (numpy.ndarray): A (4x4) numpy array representing the initial density matrix of the impurity model in the fermionic operator basis.
    - operator_0 (numpy.ndarray): A (4x4) numpy array representing the operator at time 0. Expected to be odd in Grassmann variables, e.g. a creation/anihilation operator. Otherwise strings below should be adapted
    - operator_tau (numpy.ndarray): A (4x4) numpy array representing the operator at time tau. Expected to be odd in Grassmann variables, e.g a creation/anihilation operator. Otherwise strings below should be adapted

    Returns:
    - numpy.ndarray: A vector representing the propagator of the impurity model, <operator_tau(tau) operator_0(0)>.
    """

    #infer the number of timesteps form the length of the many-body representation of the IF
    nbr_time_steps = int(np.log2(len(IF_MB))/4)

    propagator = []#empty array to hold the values of the propagator

    #construct the explicit many-body representation of the MPO for the partition sum (without operator insertions)
    MPO = impurity_MPO(U_evol = U_evol, initial_density_matrix=init_density_matrix, nbr_time_steps=nbr_time_steps)
    imp_gate_Z = MPO["init_state"]#initialize MPO with initial state
    for gate in MPO["gates"]:
        imp_gate_Z = np.kron(imp_gate_Z, gate)
    imp_gate_Z = np.kron(imp_gate_Z, MPO["boundary_condition"])#add final state

    #______compute propagator for spin up on forward branch________
    #compute the Keldysh index corresponding to the initial time point
    Keldysh_idx_0 = position_to_Keldysh_idx(0, 'f', nbr_time_steps)
    for tau in range(nbr_time_steps):
        #compute the Keldysh index corresponding at time tau
        Keldysh_idx_tau = position_to_Keldysh_idx(tau, 'f', nbr_time_steps)
        operator_a = operator_0
        operator_b = operator_tau  @ U_evol if tau > 0 else operator_tau
        #contruct impurity gates with operators
        MPO = impurity_MPO(U_evol = U_evol, initial_density_matrix=init_density_matrix, nbr_time_steps=nbr_time_steps, operator_a=operator_a, Keldysh_idx_a=Keldysh_idx_0, operator_b=operator_b, Keldysh_idx_b=Keldysh_idx_tau)
        
        #impurity gate with operators:
        imp_gate_operators = MPO["init_state"]#initialize MPO with initial state
        for gate in MPO["gates"]:
            imp_gate_operators = np.kron(imp_gate_operators, gate)
        imp_gate_operators = np.kron(imp_gate_operators, MPO["boundary_condition"])#add final state

        #compute the propagator
        propag = IF_MB @ imp_gate_operators @ IF_MB#unnormaized propagator
        Z = IF_MB @ imp_gate_Z @ IF_MB#parition sum

        propag = propag / Z #normalize the propagator

        propagator.append(propag)

    return np.array(propagator)



def setup_MB_operators(E_up: float = 0, E_down: float = 0, t_spinhop: float = 0, U:float = 0, beta_up: float = 0, beta_down: float = 0, delta_t: float = 0.1) -> dict:
    """
    Set up the many-body operators needed to compute the propagator and evolved density matrix.

    Parameters:
    - E_up (float): The energy of the up fermion.
    - E_down (float): The energy of the down fermion.
    - t_spinhop (float): The spin hopping term.
    - U (float): The interaction term.
    - beta_up (float): The inverse temperature of the up fermion.
    - beta_down (float): The inverse temperature of the down fermion.
    - delta_t (float): The time step.

    Returns:
    - dict: A dictionary containing the initial density matrix, the time-evolution operator, and the annihilation operators for the up and down fermions.
    """
    #array containing the many-body representations of all annihilation operators for two fermions species
    c_down, c_up = annihilation_ops(n_ferms=2)

    #____INITIAL DENSITY MATRIX____
    init_density_matrix = expm(- beta_up * c_up.T @ c_up) @ expm(- beta_down * c_down.T @ c_down)
    Z = np.trace(init_density_matrix)
    init_density_matrix = init_density_matrix / Z#normalize

    #___SET UP HAMILTONIAN
    Ham_onsite = E_up * c_up.T @ c_up  + E_down * c_down.T @ c_down 
    Ham_spinhop = t_spinhop * (c_up.T @ c_down + c_down.T @ c_up)
    Ham_interaction = U * c_up.T @ c_up @ c_down.T @ c_down 

    #by convention (and analogy with the Grassmann code), define the time-evolution operator as the product of the spin-diagonal and spin-offdiagoanl evolution operators.
    U_evol =  expm(1j*(Ham_onsite + Ham_interaction) * delta_t)  @ expm(1j * Ham_spinhop * delta_t)#time-evolution operator of the impurity model

    return {"init_density_matrix": init_density_matrix, "U_evol": U_evol, "c_down": c_down, "c_up": c_up}

if __name__ == "__main__":

    # Set parameters for the impurity model
    params = {"E_up": 3, "E_down": 4, "t_spinhop": 5, "beta_up": 1, "beta_down": 2, "delta_t": 0.1}

    #read out matrix B from file
    filename = '/Users/julianthoenniss/Documents/PhD/code/InfluenceFunctional_FreeFerm/data/benchmark_delta_t=0.1_Tren=5_beta=50.0_T=3'

    #read the influence matrix B from disk
    B = read_IF(filename)

    #convert the influence matrix to a many-body state
    IF_MB = IF_many_body(B)
    
    #set up the many-body operators
    MB_operators = setup_MB_operators(**params)
    init_density_matrix = MB_operators['init_density_matrix']
    U_evol = MB_operators['U_evol']
    c_down = MB_operators['c_down']
    c_up = MB_operators['c_up']

    #compute the propagator
    #spin up:
    G_upup_ff = compute_propagator_MB(IF_MB, U_evol, init_density_matrix, operator_0=c_up.T, operator_tau= c_up)
    #spin down:
    G_downdown_ff = compute_propagator_MB(IF_MB, U_evol, init_density_matrix, operator_0=c_down.T, operator_tau= c_down)

    print("MANY-BODY OVERLAP:")
    for tau in range (len(G_upup_ff)):
        print(f'G_upup_ff({tau}) = {np.round(G_upup_ff[tau],6)}')
    for tau in range (len(G_downdown_ff)):
        print(f'G_downdown_ff({tau}) = {np.round(G_downdown_ff[tau],6)}')


    #evolve the density matrix
    density_matrices = evolve_density_matrix_MB(IF_MB, U_evol = U_evol, init_density_matrix = init_density_matrix)
    #normalize density matrices
    for tau in range (0,len(density_matrices),2):
        density_matrices[tau] = density_matrices[tau] / np.trace(density_matrices[tau])
    #print density matrices
    print("Density matrices at full time steps:")
    for tau in range (0,len(density_matrices),2):
        print(f'rho({tau//2}) = \n{np.real(density_matrices[tau])}')

