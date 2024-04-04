import numpy as np
import h5py
from scipy.linalg import expm

import config as cfg
from imag_time_compact import B_from_spec_dens
from overlap_imag import compute_propagator_grassmann, construct_grassmann_exponential
from dual_overlap_imag import make_first_entry_last, IF_many_body, Hamiltonian, compute_propagator
from create_correlation_block import create_correlation_block


#____create the exponent matrix B from the spectral density
# Parameters
N_timesteps = 6#number of time steps
delta_tau = 1.0# time step
int_lim_low = -1#lower frequency integration limit 
int_lim_up = 1#upper frequency integration limit
beta = 4.#inverse temperature
dim_B = 2 * N_timesteps#dimension of exponent matrix B

# Create exponent matrix B
B = B_from_spec_dens(dim_B = dim_B, beta = beta, delta_tau = delta_tau,  int_lim_low=int_lim_low, int_lim_up=int_lim_up)


#____compute the propagator by explicitly solving the Grassmann path integral and evaluating pfaffians
# Impurity parameters
E_up = 0.3
E_down = 0.1
t = 0.2

# Construct the full action of the path integral
exponent = construct_grassmann_exponential(B, E_up, E_down, t, delta_tau)

# ____Compute the propagator of the impurity model
time_grid = np.arange(1, dim_B//2) * delta_tau #define the time grid for printed output
G_up, G_down = compute_propagator_grassmann(exponent)

print(f"Grassmann propagator for parameters: E_up = {E_up}, E_down = {E_down}, t = {t}, delta_tau = {delta_tau}" )
for tau, G_up, G_down in zip(time_grid, G_up, G_down):
        print(f"At tau = {np.round(tau,1)}: G_up = {G_up},  G_down = {G_down}")



#____compute the propagator by computing the many-body wavefunction of the influence functional and evaluating the overlap using the 'dual' gates that define the impurity MPO
#bring the first leg to the last position
B = make_first_entry_last(B)

#compute the correlation matrix that corresponds to this influence functional
corr_matrix = create_correlation_block(B) #This is the correlation matrix that is the input to the Fishman-White algorithm. 

#Here, instead of computing the MPS representation, we compute the exact many-body wavefunction of the influence functional. In this case, we don't need the correlation matrix.
#The impurity gates are the same that define the MPO. Hence, the code below can conceptually be copy-pasted with just the replacement of 
# the many-body wavefunction IF_MB by the MPS representation of the influence functional.

#Compute the many-body wavefunction of the influence functional
IF_MB = IF_many_body(B)


# ____Construct the local impurity Hamiltonian:
# create many-body Hamiltonian for the Anderson impurity model
Ham_Anderson = Hamiltonian(E_up = E_up, E_down = E_down) 
# create many-body Hamiltonian for the spin-hopping model
Ham_spin_hopping = Hamiltonian(t=t) 

# ____Define the impurity time-evolution operator. 
#Here, we define a successive application of the spin-hopping and Anderson impurity Hamiltonians to be consistent with the Grassmann calculation above
U_Anderson = expm(- Ham_Anderson * delta_tau)
U_spin_hopping = expm(- Ham_spin_hopping * delta_tau)
U_evol = U_spin_hopping @ U_Anderson #define a combined time evolution operator


# ____Compute the propagator of the impurity model. Check out the function 'compute_propagator' where you see that this amount to an overlap IF_MB @ Gate @ IF_MB. 
#Spin up
G_up = compute_propagator(IF_MB=IF_MB, U_evol=U_evol, dim_B=dim_B, operator_0=cfg.c_up_dag, operator_tau=cfg.c_up)
G_down = compute_propagator(IF_MB=IF_MB, U_evol=U_evol, dim_B=dim_B, operator_0=cfg.c_down_dag, operator_tau=cfg.c_down)

print(f"Many-body overlap propagator for parameters: E_up = {E_up}, E_down = {E_down}, t = {t}, delta_tau = {delta_tau}" )
for tau, G_up, G_down in zip(time_grid, G_up, G_down):
    print(f"At tau = {np.round(tau,1)}: G_up = {G_up},  G_down = {G_down}")



