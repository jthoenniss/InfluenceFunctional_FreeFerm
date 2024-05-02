"""
This file contains a demo of the real-time IM algorithm. It contains the following steps:
1. Read out a given IF
2. Compute the propagator using the exact Grassmann path integral formalism
3. Compute the propagator using the explicit many-body state representation of the IF
"""
import numpy as np
import os,sys
#append parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__),"../../.."))
from src.real_time.single_orbital.print_comparison import print_propagators, print_matrices
from src.real_time.compute_IF.real_time_IF import read_IF
from src.real_time.single_orbital.overlap import construct_grassmann_exponential, compute_propagator_grassmann, evolve_density_matrix_grassmann
from src.real_time.single_orbital.dual_overlap import compute_propagator_MB, evolve_density_matrix_MB, setup_MB_operators
from src.shared_modules.IF_many_body import IF_many_body




if __name__ == "__main__":
    #Step 1: Read out the IF from file
    filename = '/Users/julianthoenniss/Documents/PhD/code/InfluenceFunctional_FreeFerm/data/benchmark_delta_t=0.1_Tren=5_beta=50.0_T=3'
    B = read_IF(filename)

    # Set parameters for the impurity model
    params = {"E_up": 3., "E_down": 4., "t_spinhop": 5., "beta_up": 1., "beta_down": 2., "delta_t": 0.1}

    #Set up path integral for Grassmann calculation__________
    exponent = construct_grassmann_exponential(B = B, **params)

    #Set up many-body calculation__________
    #Many-body representation of IF
    IF_MB = IF_many_body(B)
    #set up many-body operators
    MB_operators = setup_MB_operators(**params)
    init_density_matrix = MB_operators['init_density_matrix']
    U_evol = MB_operators['U_evol']
    c_down = MB_operators['c_down']
    c_up = MB_operators['c_up']

 
    #_______COMPUTE PROPAGATORS
    #Grassmann:
    GM_G_upup_ff, GM_G_downdown_ff = compute_propagator_grassmann(exponent)
    #Many-body overlap:
    MB_G_upup_ff = compute_propagator_MB(IF_MB, U_evol, init_density_matrix, operator_0=c_up.T, operator_tau= c_up)
    MB_G_downdown_ff = compute_propagator_MB(IF_MB, U_evol, init_density_matrix, operator_0=c_down.T, operator_tau= c_down)
            

    #_______EVOLVE DENSITY MATRIX
    #Grassmann:
    GM_density_matrix = evolve_density_matrix_grassmann(exponent)
    #Many-body:
    MB_density_matrix = evolve_density_matrix_MB(IF_MB, U_evol, init_density_matrix)[::2]#extract only density matrices at ful time steps


    #______Print output
    print(f"Results for parameters: {params}\n")
    print_propagators(GM_G_upup_ff, GM_G_downdown_ff, MB_G_upup_ff, MB_G_downdown_ff)
    print_matrices(GM_density_matrix, MB_density_matrix)

