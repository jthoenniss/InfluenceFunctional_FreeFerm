"""
This module contains the necessary functions in order to evaluate the non-interacting path integral for a given input matrix B.
It can be used to compute the propagator and the density matrix.
"""

import numpy as np
from scipy import linalg
from pfapack import pfaffian as pf
#add parent directory to path in order to import the read_IF function
import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
parent_dir = os.path.join(current_dir, '../../..')
sys.path.append(parent_dir)
#import the function to read the influence matrix from disk
from compute_IF.real_time_IF import read_IF, folded_to_Grassmann
from src.shared_modules.IF_many_body import IF_many_body


def construct_grassmann_exponential(B: np.ndarray, delta_t: float, t_spinhop: float = 0, E_up: float = 0, E_down: float = 0, beta_up : float = 0, beta_down: float = 0) -> np.ndarray:
    """
    Construct the full action of the path integral for the non-interacting case.
    
    Input:
    - B: np.ndarray, shape=(dim_B, dim_B), influence matrix
    - delta_t: float, time step
    - t_spinhop: float, spin hopping parameter
    - E_up: float, chemical potential for spin up
    - E_down: float, chemical potential for spin down
    - beta_up: float, inverse temperature for initial state of spin up impurity
    - beta_down: float, inverse temperature for initial state of spin down impurity

    Output:
    - exponent: np.ndarray, shape=(4*dim_B, 4*dim_B), full action of the path integral
    """

    dim_B = B.shape[0]
    #bring matrix B into Grassmann order: reverse rows and columns
    B = folded_to_Grassmann(B)
    
    # adjust signs that make the influence matrix a vectorized state.
    # Absorb it here instead of in the impurity in order to have a very independent check.
    #change the sign of all ingoing variables for spin up and all outgoing variables for spin down, i.e. all elements with an odd sum of indices
    for i in range (dim_B):
        for j in range (dim_B):
            if (i+j)%2 == 1:
                B[i,j] *= -1

    #this exponent will contain the exponents of both spin species as well as the impurity dynamics
    exponent = np.zeros((4*dim_B, 4*dim_B), dtype=np.complex_)

    #the basis is ordered as follows: up, down, \bar{up}, \bar{down}
    
    # Influence matices for both spin species
    #spin down
    exponent[dim_B:2*dim_B, dim_B:2*dim_B] = B[:, :]
    #spin up
    exponent[2*dim_B:3*dim_B, 2*dim_B:3*dim_B] = B[:, :]

    # integration measure
    # spin up
    exponent[:dim_B, 2*dim_B :3*dim_B] += np.identity(dim_B)# up - \bar{up}
    exponent[2*dim_B :3*dim_B, :dim_B] += -np.identity(dim_B)#transpose
    # spin down
    exponent[dim_B:2*dim_B, 3*dim_B:4*dim_B ] += np.identity(dim_B)#down - \bar{down}
    exponent[3*dim_B:4*dim_B , dim_B:2*dim_B] += -np.identity(dim_B) #transpose
  
    
    # Initial state of impurity (thermal with temperature beta)
    #spin up
    exponent[dim_B//2 - 1, dim_B//2 ] += np.exp(- beta_up)
    #Transpose (antisymm)
    exponent[dim_B//2, dim_B//2 - 1] -= np.exp(- beta_up)
    #spin down
    exponent[3 * dim_B + dim_B//2 - 1, 3 * dim_B + dim_B//2] += np.exp(- beta_down)
    #Transpose (antisymm)
    exponent[3 * dim_B + dim_B//2, 3 * dim_B + dim_B//2 - 1] -= np.exp(- beta_down)
    

    # temporal boundary condition for measure
    # sign because substituted in such a way that all kernels are the same.
    #spin up
    exponent[dim_B - 1, 0] += -1
    #Transpose (antisymm)
    exponent[0, dim_B - 1] -= -1
    #spin down
    exponent[3 * dim_B + dim_B - 1, 3 * dim_B] += -1
    #Transpose (antisymm)
    exponent[3 * dim_B, 3 * dim_B + dim_B - 1] -= -1

    #_______Add impurity gates to the action_________
    for i in range(dim_B//4-1):

        T=1+np.tan(t_spinhop * delta_t/2)**2
        # forward 
        # (matrix elements between up -> down)
        exponent[dim_B//2 - 3 - 2*i, 3*dim_B + dim_B//2 - 2 - 2*i] += 1.j * np.tan(t_spinhop * delta_t/2) *2/T *np.exp(1.j * E_up * delta_t)
        exponent[dim_B//2 - 2 - 2*i, 3*dim_B + dim_B//2 - 3 - 2*i] -= 1.j * np.tan(t_spinhop * delta_t/2)*2/T *np.exp(1.j * E_down * delta_t)
        #(matrix elements between up -> up)
        exponent[dim_B//2 - 3 - 2*i, dim_B//2 - 2 - 2*i] += 1 *np.cos(t_spinhop * delta_t) *np.exp(1.j * E_up * delta_t)
        exponent[3*dim_B + dim_B//2 - 3 - 2*i, 3*dim_B + dim_B//2 - 2 - 2*i] += 1 *np.cos(t_spinhop * delta_t) *np.exp(1.j * E_down * delta_t)

        # forward Transpose (antisymm)
        exponent[3*dim_B + dim_B//2 - 2 - 2*i, dim_B//2 - 3 - 2*i] += -1.j * np.tan(t_spinhop * delta_t/2)*2/T *np.exp(1.j * E_up * delta_t)
        exponent[3*dim_B + dim_B//2 - 3 - 2*i, dim_B//2 - 2 - 2*i] -= -1.j * np.tan(t_spinhop * delta_t/2)*2/T *np.exp(1.j * E_down * delta_t)
        exponent[dim_B//2 - 2 - 2*i,dim_B//2 - 3 - 2*i] += -1 *np.cos(t_spinhop * delta_t) *np.exp(1.j * E_up * delta_t)
        exponent[3*dim_B + dim_B//2 - 2 - 2*i, 3*dim_B + dim_B//2 - 3 - 2*i] += -1 *np.cos(t_spinhop * delta_t) *np.exp(1.j * E_down * delta_t)

        # backward
        exponent[dim_B//2 + 1 + 2*i, 3*dim_B + dim_B//2 + 2 + 2*i] += - 1.j * np.tan(t_spinhop * delta_t/2)*2/T *np.exp(-1.j * E_down * delta_t)
        exponent[dim_B//2 + 2 + 2*i, 3*dim_B + dim_B//2 + 1 + 2*i] -= - 1.j * np.tan(t_spinhop * delta_t/2)*2/T *np.exp(-1.j * E_up * delta_t)
        exponent[dim_B//2 + 1 + 2*i, dim_B//2 + 2 + 2*i] += 1 *np.cos(t_spinhop * delta_t) * np.exp(-1.j * E_up * delta_t)
        exponent[3*dim_B + dim_B//2 + 1 + 2*i, 3*dim_B + dim_B//2 + 2 + 2*i] += 1 *np.cos(t_spinhop * delta_t) * np.exp(-1.j * E_down * delta_t)

        # backward Transpose (antisymm)
        exponent[3*dim_B + dim_B//2 + 2 + 2*i, dim_B//2 + 1 + 2*i] += + 1.j * np.tan(t_spinhop * delta_t/2)*2/T *np.exp(-1.j * E_down * delta_t)
        exponent[3*dim_B + dim_B//2 + 1 + 2*i, dim_B//2 + 2 + 2*i] -= + 1.j * np.tan(t_spinhop * delta_t/2)*2/T *np.exp(-1.j * E_up * delta_t)
        exponent[dim_B//2 + 2 + 2*i, dim_B//2 + 1 + 2*i] += -1 *np.cos(t_spinhop * delta_t) * np.exp(-1.j * E_up * delta_t)
        exponent[3*dim_B + dim_B//2 + 2 + 2*i, 3*dim_B + dim_B//2 + 1 + 2*i] += -1 *np.cos(t_spinhop * delta_t) * np.exp(-1.j * E_down * delta_t)
    
    return exponent
    
def evolve_density_matrix_grassmann(exponent_full: np.ndarray) -> np.ndarray:

    """
    Function to compute the evolved density matrix for the non-interacting case at each full time step.
    Parameters:
    - exponent_full: np.ndarray, shape=(4*dim_B, 4*dim_B), full action of the path integral

    Returns:
    - density_matrices: list, list containing the density matrices at each full time step, starting at time 0 (i.e. with the initial state).
    """
    
    dim_B = exponent_full.shape[0]//4
    total_time = dim_B//4

    density_matrices = [] 

    for intermediate_time_dm in range(1,total_time+1):#index 1 here corresponds to time 0, i.e. to initial state
        #deepcopy exponent matrix
        exponent = np.copy(exponent_full)

        #Replace the evolution gates above cuts by the following if perfect depolarizer is used above cut (remove the impurity gates there):
        """
        for i in range(intermediate_time_dm, dim_B//4-1):
            exponent[dim_B//2 - 3 - 2*i, dim_B//2 + 2 + 2*i] += 1 
            exponent[dim_B//2 - 2 - 2*i, dim_B//2 + 1 + 2*i] += 1 

            exponent[3*dim_B + dim_B//2 - 3 - 2*i,3*dim_B +  dim_B//2 + 2 + 2*i] += 1 
            exponent[3*dim_B + dim_B//2 - 2 - 2*i,3*dim_B +  dim_B//2 + 1 + 2*i] += 1 

            #antisymm. transpose
            exponent[dim_B//2 + 2 + 2*i, dim_B//2 - 3 - 2*i] += -1 
            exponent[dim_B//2 + 1 + 2*i, dim_B//2 - 2 - 2*i] += -1 

            exponent[3*dim_B +  dim_B//2 + 2 + 2*i, 3*dim_B + dim_B//2 - 3 - 2*i] += -1 
            exponent[3*dim_B +  dim_B//2 + 1 + 2*i, 3*dim_B + dim_B//2 - 2 - 2*i] += -1 """

        delt = 2 * (total_time - intermediate_time_dm)

        # ingoing legs of IF above cut
        idx_IF_up_fw = 2*dim_B + delt+1 #index for spin up IF leg, ingoing forward
        idx_IF_up_bw = 3*dim_B - delt - 2 #index for spin up IF leg, ingoing backward
        idx_IF_down_fw = dim_B + delt+1 #index for spin down IF leg, ingoing forward
        idx_IF_down_bw = 2*dim_B - delt - 2 #index for spin down IF leg, ingoing backward

        #identify open legs that determine density matrix
        idx_imp_up_fw = delt+1 #index for spin up impurity leg, outgoing forward
        idx_imp_up_bw = dim_B - delt - 2 #index for spin up impurity leg, outgoing backward
        idx_imp_down_fw = 3*dim_B + delt+1 #index for spin down impurity leg, outgoing forward
        idx_imp_down_bw = 4*dim_B - delt - 2 #index for spin down impurity leg, outgoing backward
        cut_indices = [idx_imp_up_fw, idx_imp_up_bw, idx_imp_down_fw, idx_imp_down_bw]

    
        #take out measure connecting cut legs at cut
        remove_measure = np.zeros_like(exponent)
        #spin up
        remove_measure[idx_imp_up_fw, idx_IF_up_fw] -= 1
        remove_measure[idx_imp_up_bw, idx_IF_up_bw] -= 1
        #spin down
        remove_measure[idx_IF_down_fw, idx_imp_down_fw] -= 1
        remove_measure[idx_IF_down_bw, idx_imp_down_bw] -= 1
        #antisymm
        remove_measure -= remove_measure.T

        exponent += remove_measure
        

        #connect the ingoing legs of IF above cut
        connect_ingoing_legs = np.zeros_like(exponent)
        #spin up
        connect_ingoing_legs[idx_IF_up_bw, idx_IF_up_fw] += 1
        #spin down
        connect_ingoing_legs[idx_IF_down_bw, idx_IF_down_fw] += 1
        #antisymm
        connect_ingoing_legs -= connect_ingoing_legs.T

        exponent += connect_ingoing_legs

    
        #elements that are not connected to the cut legs
        A = np.delete(exponent, cut_indices, axis=0)#remove the rows that are connected to the cut legs
        A = np.delete(A, cut_indices, axis=1)#remove the columns that are connected to the cut legs

        #elements that connect the cut legs to the uncut legs  
        R_without_columns = np.delete(exponent, cut_indices, axis=1)#remove the columns that are connected to the cut legs
        R_without_columns = R_without_columns[np.ix_(cut_indices, np.arange(R_without_columns.shape[1]))]

        #element that contain only cut legs
        C = np.zeros((4,4),dtype=np.complex_)
        C = exponent[np.ix_(cut_indices, cut_indices)]


        #invert A (integrate out all uncut legs)
        A_inv = linalg.inv(A)
        
        rho_exponent_evolved = 0.5*(R_without_columns @ A_inv @ R_without_columns.T + C )

        a1 = 2 * rho_exponent_evolved[2,3]#down forward, down backward
        a2 = +2 * rho_exponent_evolved[2,0]#down forward, up forward
        a3 = 2 * rho_exponent_evolved[2,1]#down forward, up backward
        a4 = 2 * rho_exponent_evolved[3,0]#down backward, up foward
        a5 = +2 * rho_exponent_evolved[3,1]#down backward, up backward
        a6 = 2 * rho_exponent_evolved[0,1]#up forward, up backward
        

        #construct the full density matrix from the entries of the exponential
        rho_evolved = np.zeros((4,4),dtype=np.complex_)
        rho_evolved[0,0] = 1
        rho_evolved[0,3] = - a5 

        rho_evolved[1,1] = a6
        rho_evolved[1,2] = - a4

        rho_evolved[2,1] = a3
        rho_evolved[2,2] = a1

        rho_evolved[3,0] = a2
        rho_evolved[3,3] = a1*a6 - a2*a5 + a3*a4

        #append to array of density matrices
        density_matrices.append(rho_evolved)

    return density_matrices


def compute_propagator_grassmann(exponent: np.ndarray, trotter_convention = ''):
    """
    Function to compute different components of the propagator for the non-interacting case.
    Parameters:
    - exponent: np.ndarray, shape=(4*dim_B, 4*dim_B), full action of the path integral
    - trotter_convention: str, convention for the Trotter decomposition of the path integral

    Returns:
    - G_upup_ff: list, forward-forward propagator for spin up
    - G_downdown_ff: list, forward-forward propagator for spin down
    - G_updown_ff: list, forward-forward propagator between spin up and down
    - G_upup_fb: list, forward-backward propagator for spin up
    - G_downdown_fb: list, forward-backward propagator for spin down
    - G_updown_fb: list, forward-backward propagator between spin up and down
    """

    dim_B = exponent.shape[0]//4
    nbr_time_steps = dim_B//4 
   

    G_upup_ff = [] #forward-forward
    G_downdown_ff = [] #forward-forward
    G_updown_ff = [] #forward-forward
    G_upup_fb = [] #forward-backward
    G_downdown_fb = [] #forward-backward
    G_updown_fb = [] #forward-backward

    exponent_inv = linalg.inv(exponent)

    #tau 0 :
    for tau in range (nbr_time_steps):
        G_upup_ff.append(pf.pfaffian(np.array(exponent_inv.T[np.ix_([dim_B//2 -1, 2*dim_B +dim_B//2 -1 - 2*tau], [dim_B//2 -1, 2*dim_B +dim_B//2 -1- 2*tau])]))) #c(0) c^\dag(0)


    return G_upup_ff, G_downdown_ff, G_updown_ff, G_upup_fb, G_downdown_fb, G_updown_fb 


if __name__ == "__main__":

    

    #import file containing the IF (must exist. If not, run the file 'real_time_IM.py)
    
    filename = '/Users/julianthoenniss/Documents/PhD/code/InfluenceFunctional_FreeFerm/data/benchmark_delta_t=0.1_Tren=5_beta=50.0_T=2'

    #read the influence matrix B from disk
    B = read_IF(filename)




    #Set parameters:
    delta_t = 0.1 #time step
    E_up = 3 #energy of the up fermion
    E_down = 4 #energy of the down fermion
    t_spinhop = 5 #spin hopping term
    beta_up = 1 #inverse temperature of the up fermion
    beta_down = 2 #inverse temperature of the down fermion
    #compute the full action of the path integral
    exponent = construct_grassmann_exponential(B = B, delta_t=delta_t, t_spinhop=t_spinhop, E_up=E_up, E_down=E_down, beta_up=beta_up, beta_down=beta_down)

    #compute the propagator
    G_upup_ff, G_downdown_ff, G_updown_ff, G_upup_fb, G_downdown_fb, G_updown_fb = compute_propagator_grassmann(exponent)
    
    print("Grassmann Overlap -- Forward-forward propagator for spin up:")
    for tau in range (len(G_upup_ff)):
        print(f'G_upup_ff({tau}) = {G_upup_ff[tau]}')
        #print(f'G_downdown_ff({tau}) = {G_downdown_ff[tau]}')
        #print(f'G_updown_ff({tau}) = {G_updown_ff[tau]}')
        #print(f'G_upup_fb({tau}) = {G_upup_fb[tau]}')
        #print(f'G_downdown_fb({tau}) = {G_downdown_fb[tau]}')
        #print(f'G_updown_fb({tau}) = {G_updown_fb[tau]}')

    #compute the density matrix
    density_matrices = evolve_density_matrix_grassmann(exponent)
    print("Grassmann Overlap -- Density matrix:")
    for tau in range (len(density_matrices)):
        print(f'rho({tau}) = \n{np.real(density_matrices[tau] / np.trace(density_matrices[tau] ))}')