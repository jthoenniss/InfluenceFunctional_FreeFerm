# Module to compute the Influence functional from a spectral density

import numpy as np
from scipy import linalg
import scipy.integrate as integrate  # to compute frequency integral in order to evaluate hybridization function
import h5py  # to store the influence functional in a file


def spec_dens(energy: float, Gamma: float = 1.0):
    """
    Spectral denstiy of the bath for a flat band with smooth cutoffs
    Parameters:
    - energy(float) : Energy at which the spectral density is evaluated.
    - Gamma(float) : Energy scale of the bath.

    Returns:
    - float : Value of the spectral density at the given energy.
    """
    
    #e_c = 10.*Gamma 
    #nu = 10./Gamma
    #return  2 * Gamma /((1+np.exp(nu*(energy - e_c))) * (1+np.exp(-nu*(energy + e_c)))) 

    return Gamma


def Grassmann_folded_map(dim_B: int) -> np.ndarray:
    """
    Generate index map that transforms between the Grassmann and folded basis.
    Grassmann basis means, the variables are ordered as (out_T_fw, in_T_fw, out_(T-1)_fw, in_(T-1)_fw,...,out_0_fw, in_0_bw, ...in_T_bw, out_T_bw)
    Folded basis means, the variables are ordered as (in_fw, in_bw, out_fw, out_bw,...) in increasing order of time.

    Parameters:
    - dim_B(int) : Dimension of the influence matrix.

    Returns:
    - np.ndarray : Index map that transforms between the Grassmann and folded basis.
    """
    # Create index arrays for the permutation
    indices = np.zeros(dim_B, dtype=int)
    indices[::4] = np.arange(dim_B//2 - 1, -1, -2)  # Even indices for forward
    indices[1::4] = np.arange(dim_B//2, dim_B, 2)       # Odd indices for backward
    indices[2::4] = np.arange(dim_B//2 - 2, -1, -2) # Even indices for forward
    indices[3::4] = np.arange(dim_B//2 + 1, dim_B, 2)   # Odd indices for backward

    return indices

def Grassmann_to_folded(B_Grassmann: np.ndarray) -> np.ndarray:
    """
    Transformation that bring the influence matrix from the Grassmann basis to the folded basis.
    Grassmann basis means, the variables are ordered as (out_T_fw, in_T_fw, out_(T-1)_fw, in_(T-1)_fw,...,out_0_fw, in_0_bw, ...in_T_bw, out_T_bw)
    Folded basis means, the variables are ordered as (in_fw, in_bw, out_fw, out_bw,...) in increasing order of time.

    Parameters:
    - B_Grassmann(np.ndarray) : Input matrix to the real-time influence functional in the Grassmann basis.

    Returns:
    - np.ndarray : The influence matrix in the folded basis.
    """
    dim_B = B_Grassmann.shape[0]

    indices = Grassmann_folded_map(dim_B)

    # Rotate from Grassmann convention back to folded basis
    B_folded = B_Grassmann[indices, :][:, indices]

    return B_folded

def folded_to_Grassmann(B_folded: np.ndarray) -> np.ndarray:
    """
    Inverse of the transformation that bring the influence matrix from the Grassmann basis to the folded basis.
    Grassmann basis means, the variables are ordered as (out_T_fw, in_T_fw, out_(T-1)_fw, in_(T-1)_fw,...,out_0_fw, in_0_bw, ...in_T_bw, out_T_bw)
    Folded basis means, the variables are ordered as (in_fw, in_bw, out_fw, out_bw,...) in increasing order of time.

    Parameters:
    - B_folded(np.ndarray) : Input matrix to the real-time influence functional in the folded basis.

    Returns:
    - np.ndarray : The influence matrix in the Grassmann basis.
    
    """
    dim_B = B_folded.shape[0]

    indices = Grassmann_folded_map(dim_B)

    # Create inverse index array
    inverse_indices = np.zeros_like(indices)
    inverse_indices[indices] = np.arange(dim_B)

    # Rotate from folded basis into Grassmann basis
    B_Grassmann = B_folded[inverse_indices, :][:, inverse_indices]

    return B_Grassmann





def hybridization_to_B(
    integs_a: np.ndarray,
    integs_b: np.ndarray,
    nbr_time_steps: int,
    delta_t: float,
    nbr_substeps: int,
) -> np.ndarray:
    """
    Function that constructs the B-matrix that defines the influence functional from a given hybridization function (integs_a, integs_b).
    The B-matrix is constructed in the folded basis, i.e. the basis is (in-fw, in-bw, out-fw, out-bw,...) which is needed for the correlation matrix.

    Parameters:
    - integs_a (np.ndarray): hole propagation-component of the hybridization function.
    - integs_b (np.ndarray): particle propagation-component of the hybridization function.
    - nbr_time_steps (int): The number of time steps in the influence functional.
    - delta_t (float): The time step in the environment.
    - nbr_substeps (int): The number of substeps per time step delta_t that are integrated out.

    Returns:
    - np.ndarray: The B-matrix that defines the influence functional.
    """

    delta_t_fine = delta_t / nbr_substeps
    
    B = np.zeros(
        (4 * nbr_time_steps * nbr_substeps, 4 * nbr_time_steps * nbr_substeps),
        dtype=np.complex_,
    )
    for j in range(nbr_time_steps * nbr_substeps):
        for i in range(j + 1, nbr_time_steps * nbr_substeps):

            integ_a = integs_a[i - j]
            integ_b = integs_b[i - j]

            B[4 * i, 4 * j + 1] = -np.conj(integ_b) * delta_t_fine**2
            B[4 * i, 4 * j + 2] = -np.conj(integ_a) * delta_t_fine**2
            B[4 * i + 1, 4 * j] = integ_b * delta_t_fine**2
            B[4 * i + 1, 4 * j + 3] = integ_a * delta_t_fine**2
            B[4 * i + 2, 4 * j] = integ_b * delta_t_fine**2
            B[4 * i + 2, 4 * j + 3] = integ_a * delta_t_fine**2
            B[4 * i + 3, 4 * j + 1] = -np.conj(integ_b) * delta_t_fine**2
            B[4 * i + 3, 4 * j + 2] = -np.conj(integ_a) * delta_t_fine**2

        # for equal time
        integ_a = integs_a[0]
        integ_b = integs_b[0]

        B[4 * j + 1, 4 * j] = integ_b * delta_t_fine**2
        B[4 * j + 2, 4 * j] = integ_b * delta_t_fine**2
        B[4 * j + 3, 4 * j + 1] = -np.conj(integ_b) * delta_t_fine**2
        B[4 * j + 3, 4 * j + 2] = -np.conj(integ_a) * delta_t_fine**2

        # the plus and minus one here come from the overlap of GMs
        B[4 * j + 2, 4 * j] += 1
        B[4 * j + 3, 4 * j + 1] -= 1

    B += -B.T  # like this, one obtains 2*exponent, needed for Grassmann code.
    # here, the IF is in the folded basis: (in-fw, in-bw, out-fw, out-bw,...) which is needed for correlation matrix

    if (nbr_substeps > 1):  # if physical time steps are subdivided, integrate out the "auxiliary" legs
        # rotate from folded basis into Grassmann basis
        B = folded_to_Grassmann(B)

        #add intermediate integration measure to integrate out internal legs
        for i in range (2*nbr_time_steps):
            for j in range (nbr_substeps-1):
                B[2*i*nbr_substeps + 1 + 2*j,2*i*nbr_substeps+2+ 2*j] += 1  
                B[2*i*nbr_substeps+2+ 2*j,2*i*nbr_substeps + 1 + 2*j] += -1  

        #select submatrix that contains all intermediate times that are integrated out
        B_sub =  np.zeros((4*nbr_time_steps*(nbr_substeps-1) , 4*nbr_time_steps*(nbr_substeps-1)),dtype=np.complex_)
        for i in range (2*nbr_time_steps):
            for j in range (2*nbr_time_steps):
                B_sub[i*(2*nbr_substeps-2):i*(2*nbr_substeps-2 )+2*nbr_substeps-2,j*(2*nbr_substeps-2):j*(2*nbr_substeps-2 )+2*nbr_substeps-2] = B[2*i*nbr_substeps+1:2*(i*nbr_substeps + nbr_substeps)-1,2*j*nbr_substeps+1:2*(j*nbr_substeps + nbr_substeps)-1]

        #matrix coupling external legs to integrated (internal) legs
        B_coupl =  np.zeros((4*(nbr_substeps-1)*nbr_time_steps,4*nbr_time_steps),dtype=np.complex_)
        for i in range (2*nbr_time_steps):
            for j in range (2*nbr_time_steps):
                B_coupl[i*(2*nbr_substeps-2):i*(2*nbr_substeps-2 )+2*nbr_substeps-2,2*j] = B[2*i*nbr_substeps+1:2*(i*nbr_substeps + nbr_substeps)-1,2*j*nbr_substeps]
                B_coupl[i*(2*nbr_substeps-2):i*(2*nbr_substeps-2 )+2*nbr_substeps-2,2*j+1] = B[2*i*nbr_substeps+1:2*(i*nbr_substeps + nbr_substeps)-1,2*(j+1)*nbr_substeps-1]

        #part of matriy that is neither integrated nor coupled to integrated variables
        B_ext = np.zeros((4*nbr_time_steps,4*nbr_time_steps),dtype=np.complex_)
        for i in range (2*nbr_time_steps):
            for j in range (2*nbr_time_steps):
                B_ext[2*i,2*j] = B[2*i*nbr_substeps,2*j*nbr_substeps]
                B_ext[2*i+1,2*j] = B[2*(i+1)*nbr_substeps-1,2*j*nbr_substeps]
                B_ext[2*i,2*j+1] = B[2*i*nbr_substeps,2*(j+1)*nbr_substeps-1]
                B_ext[2*i+1,2*j+1] = B[2*(i+1)*nbr_substeps-1,2*(j+1)*nbr_substeps-1]

        #print(f"B_sub is antisymmetric: {np.allclose(B_sub, -B_sub.T)}")
     
        B = B_ext + B_coupl.T @ linalg.inv(B_sub) @ B_coupl
        
        #rotate back to folded basis
        B = Grassmann_to_folded(B)

    return B

def cont_integral(
    integrand: np.ndarray, lower_cutoff: float, upper_cutoff: float
) -> np.ndarray:
    """
    Calculate the integral of a given function over a specified energy range.

    This function performs a numerical integration of the provided integrand
    function over the continuous energy range specified by the lower and upper
    bounds. It is optimized for handling complex-valued functions.

    Parameters:
    - integrand (callable): A function to be integrated. This function should
                            accept a single float argument (energy) and return
                            an array of complex numbers.
    - lower_bound (float): The lower limit of the energy range for integration.
    - upper_bound (float): The upper limit of the energy range for integration.

    Returns:
    - np.ndarray: The result of the integration, scaled by 1/(2π), as an array
                  with one element per time step.

    Notes:
    - The function assumes that 'integrand' can handle vectorized inputs for
      efficient computation.
    - The integral is calculated using `scipy.integrate.quad_vec`, which is
      suitable for vectorized integration of real and complex functions.
    """

    result, _ = integrate.quad_vec(
        lambda energy: integrand(energy), lower_cutoff, upper_cutoff
    )

    result *= 1.0 / (2 * np.pi)  # normalize

    return result


def compute_IF(
    spectral_density: callable, nbr_time_steps: int, delta_t: float, nbr_substeps: int, mu: float, beta: float, int_lim_low: float, int_lim_up: float
) -> np.ndarray:
    """
    Function to calculate the "B-matrix" for a given spectral density.

    Parameters:
    - spectral_density (callable): A function that returns the spectral density
                                   for a given energy.
    - delta_t(float, default = 0.1) : Time step in the environment.
    - T_ren(int, default = 1) : Number of "sub"-timesteps into which a physical timestep in the environment is divided.
    - min_time(int, default = 4) : Minimal time for which IF should be computed and stored.
    - max_time(int, default = 5) : Maximal time for which IF should be computed and stored.
    - interval(int, default = 1) : Interval with which time steps should be computed.
    - mu(float, default = 0.) : Chemical potential.
    - beta(float, default = 0.) : Inverse temperature.
    - int_lim_low(float, default = -12.) : Lower integration limit.
    - int_lim_up(float, default = 12.) : Upper integration limit.
 

    Returns:
    - np.ndarray: The B-matrix that defines the influence functional.
        The order of variables is (in-fw, in-bw, out-fw, out-bw,...) in increasing time order which is needed for the correlation matrix.
    """

    delta_t_fine = delta_t / nbr_substeps
    T_max = delta_t * nbr_time_steps
    time_grid_fine = np.arange(0, T_max, delta_t_fine)  # fine time grid for integration with time step delta_t_fine

    def integrand_a(energy: float) -> np.ndarray:
        """
        Exploit vectorization of spectral density to calculate integrand for all values of t in time_grid at once.
        Returns an np.ndarray: Function to be integrated for each value t in time_grid.
        """
        fermi_factor = 1.0 / (1 + np.exp(beta * (energy - mu)))
        return (
            fermi_factor * np.exp(-1.0j * energy * time_grid_fine) * spectral_density(energy)
        )

    def integrand_b(energy: float) -> np.ndarray:
        """
        Exploit vectorization of spectral density to calculate integrand for all values of t in time_grid at once.
        Returns an np.ndarray: Function to be integrated for each value t in time_grid.
        """
        fermi_factor = 1.0 / (1 + np.exp(beta * (energy - mu))) - 1
        return (
            fermi_factor * np.exp(-1.0j * energy * time_grid_fine) * spectral_density(energy)
        )

    integs_a = cont_integral(integrand_a, int_lim_low, int_lim_up)
    integs_b = cont_integral(integrand_b, int_lim_low, int_lim_up)

    return hybridization_to_B(integs_a, integs_b, nbr_time_steps=nbr_time_steps, delta_t=delta_t, nbr_substeps=nbr_substeps)


def convert_to_simultaneous_evol_scheme(B: np.ndarray) -> np.ndarray:
    """
    Converts the exponent matrix B to the simultaenous evolution scheme. 
    Parameters:
    - B: np.ndarray, shape=(dim_B,dim_B), exponent of the IF in the successive evolution scheme. B is ordered in the convention with ordering: in_0, in_1, ...,in_{M-1}, out_0, out_1, ...,out_{M-1}
    - nbr_time_steps: int, number of time steps in the IF.
    Returns:
    - B_sim: np.ndarray, shape=(dim_B,dim_B), exponent of the IF in the simultaneous evolution scheme. B_sim is ordered in the convention with ordering: in_0, out_0, in_1, out_1, ...,in_{M-1}, out_{M-1}
    """

    B = np.copy(B)

    dim_B = B.shape[0]

    #rotate from folded basis into Grassmann basis
    B = folded_to_Grassmann(B)

    #subtract ones from overlap, these will be included in the impurity MPS, instead
    for i in range (dim_B//2):
        B[2*i, 2*i+1] -= 1 
        B[2*i+1, 2*i] += 1 
    
    dim_B += 4#update for embedding into larger matrix
    B_enlarged = np.zeros((dim_B,dim_B),dtype=np.complex_)
    B_enlarged[1:dim_B//2-1, 1:dim_B//2-1] = 0.5 * B[:B.shape[0]//2,:B.shape[0]//2]#0.5 to avoid double couting in antisymmetrization
    B_enlarged[dim_B//2+1:dim_B-1, 1:dim_B//2-1] = B[B.shape[0]//2:, :B.shape[0]//2]
    B_enlarged[dim_B//2+1:dim_B-1, dim_B//2+1:dim_B-1] = 0.5 * B[B.shape[0]//2:, B.shape[0]//2:]
    
    #include ones from grassmann identity-resolutions (not included in conventional IM)
    for i in range (0,dim_B//2):
        B_enlarged[2*i, 2*i+1] += 1 
    B_enlarged += - B_enlarged.T
    B_enlarged_inv = linalg.inv(B_enlarged)
    B = B_enlarged #check and update this

    #rotate from Grassmann basis to folded basis: (in-fw, in-bw, out-fw, out-bw,...)
    B = Grassmann_to_folded(B)

    return B


def store_IF(B: np.ndarray, filename: str) -> None:
    """
    Function to store the influence functional in a file.
    Parameters:
    - B(np.ndarray) : Input matrix to the real-time influence functional.
    - filename(str) : Name of the file in which the influence functional is stored.

    Returns:
    - None
    """
    #create hdf5 file to store 
    with h5py.File(filename + ".h5", 'w') as f:
        dset_IM_exponent = f.create_dataset('IM_exponent', B.shape, dtype=np.complex_)
        dset_IM_exponent[:,:] = B[:,:]#store IM exponent

    print('Influence functional stored in file: ', filename + ".h5")
    

def read_IF(filename: str) -> np.ndarray:
    """
    Function to read the influence functional from a file.
    Parameters:
    - filename(str) : Name of the file in which the influence functional is stored.

    Returns:
    - B(np.ndarray) : Input matrix to the real-time influence functional.
    """
    with h5py.File(filename + ".h5", 'r') as f:
        B = f['IM_exponent']
        B = B[:,:]
    return B



if __name__ == '__main__':

    nbr_time_steps = 2
    delta_t = 0.1 # time step in the environment
    nbr_substeps = 5 #number of "sub"-timesteps into which a physical timestep in the environment is divided
    mu = 0 #chemical potential, for Cohen 2015, set this to 0
    beta = 50. #chemical potential, for Cohen 2015, set this to 50./global_gamma
    int_lim_low = -12 #lower frequency integration limit
    int_lim_up = 12 #upper frequency integration limit
    filename = f'../../../data/benchmark_delta_t={delta_t}_Tren={nbr_substeps}_beta={beta}_T={nbr_time_steps}'
   
    B = compute_IF(spectral_density=lambda x: spec_dens(x, Gamma=1.), nbr_time_steps=nbr_time_steps, delta_t=delta_t, nbr_substeps=nbr_substeps, mu=mu, beta=beta, int_lim_low=int_lim_low, int_lim_up=int_lim_up)
    
    #to convert to simultaneous evolution scheme, uncomment the following line
    #B = convert_to_simultaneous_evol_scheme(B, nbr_time_steps_effective)

    #store IF
    store_IF(B, filename)

