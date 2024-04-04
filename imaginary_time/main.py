import numpy as np
import h5py

from src.imag_time_compact import B_from_spec_dens
from src.overlap_imag import compute_propagator_grassmann, construct_grassmann_exponential


#____create the exponent matrix B from the spectral density
# Parameters
N_timesteps = 6#number of time steps
delta_tau = 1.0# time step
int_lim_low = -1#lower frequency integration limit 
int_lim_up = 1#upper frequency integration limit
beta = 4.#inverse temperature
dim_B = 2 * N_timesteps#dimension of exponent matrix B

# Create B
B = B_from_spec_dens(dim_B = dim_B, beta = beta, delta_tau = delta_tau,  int_lim_low=int_lim_low, int_lim_up=int_lim_up)
filename = '/Users/julianthoenniss/Documents/PhD/data/B_imag_specdens_propag_GM'
with h5py.File(filename + '.hdf5', 'r') as f:
    B_data = f['B']
    B_read = B_data[:,:] #Note: the matrix B is stored in increasing order, i.e. the first varaible is ingoing at initial time, the last variable is outgoing at final time
    dim_B = B_read.shape[0]
print(B-B_read)
B = B_read

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
