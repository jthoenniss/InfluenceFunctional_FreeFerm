import numpy as np
from scipy import linalg
from pfapack import pfaffian as pf
import h5py


def construct_grassmann_exponential(B: np.ndarray, E_up: float, E_down: float, t:float, delta_tau:float, trotter_convention: str = 'a') -> np.ndarray:
    """
    Construct the full action of the path integral. It is quadratic and contains both, the IF, as well as the impurity dynamics.
    This function implements a successive impurity evolution with U_imp = expm( - H_spinhop * delta_tau) @ expm( - H_Anderson * delta_tau)
    where H_spinhop = t * (c^dagger_up c_down + c^dagger_down c_up) and H_Anderson = E_up  * c^dagger_up c_up + E_down * c^dagger_down c_down 
    Note that this there is no spin-spin interaction in this model which allows us to solve it exactly. (That's the purpuse of this script.)

    Parameters:
    - B: np.ndarray, shape=(dim_B, dim_B), matrix defining the influence functional
    - E_up: float, energy of the spin up state
    - E_down: float, energy of the spin down state
    - t: float, hopping parameter
    - delta_tau: float, imaginary time step
    - trotter_convention: str, convention for the Trotter decomposition, either 'a' for successive bath-impurity evolution or 'b' for simultaneous impurity-bath evolution

    Returns:
    - exponent: np.ndarray, shape=(4*dim_B, 4*dim_B), full action of the path integral
    """
    if trotter_convention not in ['a', 'b']:
        raise ValueError("Trotter convention must be either 'a' or 'b' for successive or simultaneous evolution, respectively.")
   
    B = B[::-1,::-1] #bring variables into decreasing order, i.e. the first variable it the outgoing leg at the final time and the last leg is the ingoing leg at the first time.
    
    dim_B = B.shape[0]#number of Grassmann variables

    exponent = np.zeros((4*dim_B, 4*dim_B), dtype=np.complex_)#this exponent will contain the exponents of both spin species as well as the impurity dynamics
    #the blocks of the matrix 'exponent' correspond to the following order: \bar{down}, down, \bar{up}, up

    # Influence matices for both spin species
    #spin down
    exponent[dim_B:2*dim_B, dim_B:2*dim_B] = B[:, :]
    #spin up
    exponent[2*dim_B:3*dim_B, 2*dim_B:3*dim_B] = B[:, :]


    # overlap measure connecting the impurity and the IFs.
    if trotter_convention == 'a':#for successive bath-impurity evolution
        # spin up
        exponent[2*dim_B:3*dim_B-1,1:dim_B] += -np.identity(dim_B-1)
        exponent[3*dim_B-1,0] += -1
        exponent[1:dim_B,2*dim_B:3*dim_B-1] += +np.identity(dim_B-1)
        exponent[0,3*dim_B-1] += +1
        # spin down
        exponent[3*dim_B+1:4*dim_B,dim_B:2*dim_B-1] += -np.identity(dim_B-1)
        exponent[3*dim_B,2*dim_B-1] += -1
        exponent[dim_B:2*dim_B-1,3*dim_B+1:4*dim_B] += +np.identity(dim_B-1)
        exponent[2*dim_B-1,3*dim_B] += +1

    elif trotter_convention == 'b': #for simultaneous impurity-bath evolution 
        # spin up
        exponent[2*dim_B:3*dim_B,:dim_B] += - np.identity(dim_B)
        exponent[:dim_B,2*dim_B:3*dim_B] += +np.identity(dim_B)
        # spin down
        exponent[3*dim_B:4*dim_B,dim_B:2*dim_B] += -np.identity(dim_B)
        exponent[dim_B:2*dim_B,3*dim_B:4*dim_B] += +np.identity(dim_B)


    #impurity
    #hopping between spin species (gate is easily found by taking kernel of xy gate at isotropic parameters):
    T=1-np.tanh(t * delta_tau/2)**2
    for i in range(dim_B//2 -1):
        # forward 
        # (matrix elements between up -> down), last factors of (-1) are sign changes to test overlap form
        exponent[dim_B - 2 - 2*i, 4*dim_B - 1 - 2*i] += -1. * np.tanh(t * delta_tau/2) *2/T *np.exp(-1. * E_up * delta_tau) 
        exponent[dim_B - 1 - 2*i, 4*dim_B - 2 - 2*i] -= -1. * np.tanh(t * delta_tau/2)*2/T *np.exp(-1. * E_down * delta_tau) 
        #(matrix elements between up -> up)
        exponent[dim_B - 2 - 2*i, dim_B - 1 - 2*i] += 1 *np.cosh(t * delta_tau) *np.exp(-1 * E_up * delta_tau) *(-1.) 
        #(matrix elements between down -> down)
        exponent[4*dim_B - 2 - 2*i, 4*dim_B - 1 - 2*i] += 1 *np.cosh(t * delta_tau) *np.exp(-1. * E_down * delta_tau) *(-1.)

        # forward Transpose (antisymm)
        exponent[4*dim_B - 1 - 2*i, dim_B - 2 - 2*i] += 1 * np.tanh(t * delta_tau/2)*2/T *np.exp(-1 * E_up * delta_tau) 
        exponent[4*dim_B - 2 - 2*i, dim_B - 1 - 2*i] -= 1. * np.tanh(t * delta_tau/2)*2/T *np.exp(-1. * E_down * delta_tau)
        exponent[dim_B - 1 - 2*i,dim_B - 2 - 2*i] += -1 *np.cosh(t * delta_tau) *np.exp(-1. * E_up * delta_tau) *(-1.)
        exponent[4*dim_B - 1 - 2*i, 4*dim_B - 2 - 2*i] += -1 *np.cosh(t * delta_tau) *np.exp(-1. * E_down * delta_tau) *(-1.)

    #last application contains antiperiodic bc.:
    exponent[0, 3*dim_B +1] += -1. * np.tanh(t * delta_tau/2) *2/T *np.exp(-1. * E_up * delta_tau) *(-1.) 
    exponent[1, 3*dim_B ] -= -1. * np.tanh(t * delta_tau/2)*2/T *np.exp(-1. * E_down * delta_tau) *(-1.)
    #(matrix elements between up -> up)
    exponent[0, 1] += 1 *np.cosh(t * delta_tau) *np.exp(-1 * E_up * delta_tau) *(-1.) *(-1.)
    #(matrix elements between down -> down)
    exponent[3*dim_B , 3*dim_B + 1] += 1 *np.cosh(t * delta_tau) *np.exp(-1. * E_down * delta_tau) *(-1.) *(-1.)

    # forward Transpose (antisymm)
    exponent[3*dim_B +1,0] += 1 * np.tanh(t * delta_tau/2)*2/T *np.exp(-1 * E_up * delta_tau) *(-1.) 
    exponent[3*dim_B,1] -= 1. * np.tanh(t * delta_tau/2)*2/T *np.exp(-1. * E_down * delta_tau) *(-1.)
    exponent[1,0] += -1 *np.cosh(t * delta_tau) *np.exp(-1. * E_up * delta_tau) *(-1.) *(-1.)
    exponent[3*dim_B + 1,3*dim_B] += -1 *np.cosh(t * delta_tau) *np.exp(-1. * E_down * delta_tau) *(-1.) *(-1.)

    return exponent

    

def compute_propagator_grassmann(exponent: np.ndarray, trotter_convention:str = 'a') -> np.ndarray:
    """
    Compute the propagator <c_\sigma(tau) c_\sigma(0)> from the full action of the path integral.
    This is done by using the relationship between the propagator and the Pfaffian of the inverse of the full action of the path integral.

    Parameters:
    - exponent: np.ndarray, shape=(4*dim_B, 4*dim_B), full action of the path integral
    - trotter_convention: str, convention for the Trotter decomposition, either 'a' for successive bath-impurity evolution or 'b' for simultaneous impurity-bath evolution

    Returns:
    - propagator: np.ndarray, shape=(4*dim_B, 4*dim_B), propagator of the path integral
    """
    if trotter_convention not in ['a', 'b']:
        raise ValueError("Trotter convention must be either 'a' or 'b' for successive or simultaneous evolution, respectively.")
   
   
    dim_B = exponent.shape[0]//4
    exponent_inv = linalg.inv(exponent)
 
    G_up = []
    G_down = []

    if trotter_convention == 'a':#for successive impurity-bath evolution
        for tau in range (1,dim_B//2):
            G_up.append(pf.pfaffian(exponent_inv.T[np.ix_([0,3*dim_B -1 -2*tau], [0,3*dim_B -1 -2*tau])]))
            G_down.append(pf.pfaffian(exponent_inv.T[np.ix_([2*dim_B -1 -2*tau,3*dim_B], [2*dim_B -1 -2*tau,3*dim_B])]))
    
    elif trotter_convention == 'b':#for simultaneous impurity-bath evolution
        for tau in range (1,dim_B//2):
            G_up.append(pf.pfaffian(exponent_inv.T[np.ix_([0,dim_B -1 -2*tau], [0,dim_B -1 -2*tau])]))
            G_down.append(pf.pfaffian(exponent_inv.T[np.ix_([3*dim_B,4*dim_B-1-2*tau], [3*dim_B,4*dim_B-1-2*tau])]))
 
    return G_up, G_down

if __name__ == "__main__":
    #___Import the matrix B that defines the influence functional
    filename = '/Users/julianthoenniss/Documents/PhD/data/B_imag_specdens_propag_GM'
    with h5py.File(filename + '.hdf5', 'r') as f:
        B_data = f['B']
        B = B_data[:,:]
        dim_B = B.shape[0]


    # Define the impurity parameters
    E_up = 0.3
    E_down = 0.1
    t = 0.2
    delta_tau = 1.0

    # Construct the full action of the path integral
    exponent = construct_grassmann_exponential(B, E_up, E_down, t, delta_tau)



    
    # ____Compute the propagator of the impurity model
    time_grid = np.arange(1, dim_B//2) * delta_tau #define the time grid for printed output
    G_up, G_down = compute_propagator_grassmann(exponent)

    print(f"Grassmann propagator for parameters: E_up = {E_up}, E_down = {E_down}, t = {t}, delta_tau = {delta_tau}" )
    for tau, G_up, G_down in zip(time_grid, G_up, G_down):
            print(f"At tau = {np.round(tau,1)}: G_up = {G_up},  G_down = {G_down}")