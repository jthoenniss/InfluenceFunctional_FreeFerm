import numpy as np
import h5py
from scipy.linalg import expm

from config import c_up_dag, c_up, c_down_dag, c_down
from imag_time_compact import single_mode_GF, compute_continuous_time_IF, convert_to_simultaneous_evol_scheme
from overlap_imag import compute_propagator_grassmann, construct_grassmann_exponential
from dual_overlap_imag import make_first_entry_last, IF_many_body, Hamiltonian, compute_propagator
from create_correlation_block import create_correlation_block


#____create the exponent matrix B from the spectral density
# Parameters
nbr_steps = 6#number of time steps
int_lim_low = -1#lower frequency integration limit 
int_lim_up = 1#upper frequency integration limit
beta = 4.#inverse temperature

delta_tau = beta/nbr_steps #time step

# Create exponent matrix B for single environment mode
t_hop = np.sqrt(0.8)#hopping amplitude between bath and single environment mode
g = single_mode_GF(t_hop=t_hop,beta = beta,nbr_steps=nbr_steps)
#compute continous time IF from the single-mode GF
B_spec_dens_cont = compute_continuous_time_IF(g)

#____compute the propagator by explicitly solving the Grassmann path integral and evaluating pfaffians
# Impurity parameters
E_up = 0.3
E_down = 0.1
t = 0.2

# Construct the full action of the path integral
exponent = construct_grassmann_exponential(B_spec_dens_cont, E_up, E_down, t, delta_tau=delta_tau)

# ____Compute the propagator of the impurity model
time_grid = np.arange(1, nbr_steps) * delta_tau #define the time grid for printed output
G_up, G_down = compute_propagator_grassmann(exponent)

print(f"Grassmann propagator for parameters: E_up = {E_up}, E_down = {E_down}, t = {t}, delta_tau = {delta_tau}" )
for tau, G_up, G_down in zip(time_grid, G_up, G_down):
        print(f"At tau = {np.round(tau,1)}: G_up = {G_up},  G_down = {G_down}")



#____compute the propagator by computing the many-body wavefunction of the influence functional and evaluating the overlap using the 'dual' gates that define the impurity MPO
#bring the first leg to the last position
B_spec_dens_cont_reshuf = make_first_entry_last(B_spec_dens_cont)

#compute the correlation matrix that corresponds to this influence functional
corr_matrix = create_correlation_block(B_spec_dens_cont_reshuf) #This is the correlation matrix that is the input to the Fishman-White algorithm. 

#Here, instead of computing the MPS representation, we compute the exact many-body wavefunction of the influence functional. In this case, we don't need the correlation matrix.
#The impurity gates are the same that define the MPO. Hence, the code below can conceptually be copy-pasted with just the replacement of 
# the many-body wavefunction IF_MB by the MPS representation of the influence functional.

#Compute the many-body wavefunction of the influence functional
IF_MB = IF_many_body(B_spec_dens_cont_reshuf)


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
G_up = compute_propagator(IF_MB=IF_MB, U_evol=U_evol, nbr_steps=nbr_steps, operator_0=c_up_dag, operator_tau=c_up)
G_down = compute_propagator(IF_MB=IF_MB, U_evol=U_evol, nbr_steps=nbr_steps, operator_0=c_down_dag, operator_tau=c_down)

print(f"Many-body overlap propagator for parameters: E_up = {E_up}, E_down = {E_down}, t = {t}, delta_tau = {delta_tau}" )
for tau, G_up, G_down in zip(time_grid, G_up, G_down):
    print(f"At tau = {np.round(tau,1)}: G_up = {G_up},  G_down = {G_down}")

print("\n")


#_____Check that for a trivial impurity, the successive and simultaneous evolution give the same result
print("For a TRIVIAL IMPURITY, E_up = E_down = t = 0.0, U = 0, check that the successive and simultaneous evolution scheme yield the same propagators:")
# Impurity parameters
E_up = 0.0
E_down = 0.0
t = 0.0
#convert the continuous-time IF to the simultaneous evolution scheme
B_spec_dens_cont_sim = convert_to_simultaneous_evol_scheme(B_spec_dens_cont)
#compute the propagator from the continuous-time IF:
# 1) Directly as Grassmann integral:
exponent = construct_grassmann_exponential(B_spec_dens_cont, E_up, E_down, t, delta_tau=delta_tau, trotter_convention='a')#successive evolution, trotter_convention='a'
exponent_sim = construct_grassmann_exponential(B_spec_dens_cont_sim, E_up, E_down, t, delta_tau=delta_tau, trotter_convention='b')#simultaneous evolution, trotter_convention='b'
    
# ____Compute the Grassmann propagator of the impurity model
G_up, G_down = compute_propagator_grassmann(exponent, trotter_convention='a') #successive evolution
G_up_sim, G_down_sim = compute_propagator_grassmann(exponent_sim, trotter_convention='b') #simultaneous evolution

if np.allclose(G_up, G_up_sim) and np.allclose(G_down, G_down_sim):
        print("The Grassmann propagator is the SAME for the successive and simultaneous evolution scheme:")
        print("G_up: ", G_up)
        print("G_down: ", G_down, "\n")
else: 
        print("The Grassmann propagator is DIFFERENT for the successive and simultaneous evolution scheme.")

# 2) Using the many-body wavefunction of the IF:
B_spec_dens_cont_reshuf_sim = make_first_entry_last(B_spec_dens_cont_sim, trotter_convention='b')#"reshuffling" has actually no effect for the simultaneous evolution scheme, just adding the line here for consistency with the above.
IF_MB_sim = IF_many_body(B_spec_dens_cont_sim)#compute the many-body wavefunction of the IF for the simultaneous evolution scheme

#Compute the propagator of the impurity model. 
#define evolution operator 
Ham_trivial = Hamiltonian(E_up = E_up, E_down = E_down, t=t)
U_evol = expm(- Ham_trivial * delta_tau)

G_up = compute_propagator(IF_MB=IF_MB, U_evol=U_evol, nbr_steps=nbr_steps, operator_0=c_up_dag, operator_tau=c_up)
G_down = compute_propagator(IF_MB=IF_MB, U_evol=U_evol, nbr_steps=nbr_steps, operator_0=c_down_dag, operator_tau=c_down)
G_up_sim = compute_propagator(IF_MB=IF_MB_sim, U_evol=U_evol, nbr_steps=nbr_steps, operator_0=c_up_dag, operator_tau=c_up)
G_down_sim = compute_propagator(IF_MB=IF_MB_sim, U_evol=U_evol, nbr_steps=nbr_steps, operator_0=c_down_dag, operator_tau=c_down)

if np.allclose(G_up[:-1], G_up_sim[1:]) and np.allclose(G_down[:-1], G_down_sim[1:]):
        print("The MANY-BODY OVERLAP propagator is the SAME for the successive and simultaneous evolution scheme.")
        print("G_up: ", G_up)
        print("G_down: ", G_down, "\n")
else:
        print("The MANY-BODY OVERLAP propagator is DIFFERENT for the successive and simultaneous evolution scheme. Here are the spin-up components:")
        print(G_up)
        print(G_up_sim)

