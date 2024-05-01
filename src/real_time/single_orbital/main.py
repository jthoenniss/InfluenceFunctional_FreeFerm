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
from scipy.linalg import expm
from src.real_time.compute_IF.real_time_IF import read_IF
from src.real_time.single_orbital.overlap import construct_grassmann_exponential, compute_propagator_grassmann, evolve_density_matrix_grassmann
from src.real_time.single_orbital.dual_overlap import compute_propagator_MB, evolve_density_matrix_MB, setup_MB_operators
from src.shared_modules.IF_many_body import IF_many_body




def print_propagators(GM_upup, GM_downdown, MB_upup, MB_downdown):
    num_time_steps = len(GM_upup)
    print(f"{'Time Step':>10} | "
          f"{'G_upup(t) (Grassmann)':^25} <-> {'G_upup(t) (many-body)':^25} | "
          f"{'G_downdown(t) (Grassmann)':^25} <-> {'G_downdown(t) (many-body)':^25} || "
          f"{'Coincidence up':^15} | {'Coincidence down':^15}")
    print("-" * 165)  # Adjust the line length based on the new width for better alignment

    for t in range(num_time_steps):
        GM_upup_fmt = format_complex(GM_upup[t])
        MB_upup_fmt = format_complex(MB_upup[t])
        GM_downdown_fmt = format_complex(GM_downdown[t])
        MB_downdown_fmt = format_complex(MB_downdown[t])

        coincidence_upup = check_coincidence(GM_upup[t], MB_upup[t])
        coincidence_downdown = check_coincidence(GM_downdown[t], MB_downdown[t])

        print(f"{t:>10} | {GM_upup_fmt:^25} <-> {MB_upup_fmt:^25} | "
              f"{GM_downdown_fmt:^25} <-> {MB_downdown_fmt:^25} || "
              f"{coincidence_upup:^15} | {coincidence_downdown:^15}")

    print("\n")

def format_complex(c):
    """Ensures complex numbers are formatted uniformly."""
    return f"({c.real:.6f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.6f}j)"

def check_coincidence(a, b):
    """Checks for equivalence of two complex numbers, returns 'Match' or 'Differ'."""
    return "Match" if np.allclose(a, b, atol=1e-6) else "Differ"


def format_number(n):
    """ Format a number or complex number to ensure clear presentation. """
    if np.isclose(n, 0, atol=1e-6):
        return " 0.         "  # Uniform zero presentation
    elif np.isclose(n.imag, 0, atol=1e-6):  # If imaginary part is zero
        return f"{n.real:+11.6f} "  # Format real part only
    else:
        return f"{n.real:+11.6f}{n.imag:+11.6f}j".replace('+-', '-')  # Format complex numbers

def print_matrices(GM_matrices, MB_matrices):
    t = 0
    for GM_DM, MB_DM in zip(GM_matrices, MB_matrices):

        #normalize both density matrices by their trace
        GM_DM /= np.trace(GM_DM)
        MB_DM /= np.trace(MB_DM)


        if np.allclose(GM_DM, MB_DM):
            print(f"Time Step {t} -- Coincidence: Match_________________________\n")
        else:
            print(f"Time Step {t} -- Coincidence: DIFFER________________________\n")
    
        #print them one below the other. Arrange their real and imaginary parts in an easily readable way
        print("Density Matrix from Grassmann path integral (real part):")
        print(np.real(GM_DM))
        print("Density Matrix from many-body overlap (real part):")
        print(np.real(MB_DM))
        print("\n")
        t += 1

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

