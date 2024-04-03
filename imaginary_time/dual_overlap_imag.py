import numpy as np
from pfapack import pfaffian as pf #to compute the Pfaffian when evaluating the many-body wavefunction of the influence functional
import h5py #to import the matrix B from a file and possibly stroe the propagator
from scipy.linalg import expm
import pandas as pd

#_________Define fermionic cration and annihilation operators in the many-body basis
# Note: The convention for the basis of a many-body operator A is: A = v^\dagger @ a @ v, with: v = [<0|, <0|c_up , <0|c_down, <0|c_up c_down].

c_up_dag = np.zeros((4,4),dtype=np.complex_) 
c_up_dag[1,0] = 1
c_up_dag[3,2] = -1

c_down_dag = np.zeros((4,4),dtype=np.complex_) 
c_down_dag[2,0] = 1
c_down_dag[3,1] = 1

c_up = np.zeros((4,4),dtype=np.complex_) 
c_up[0,1] = 1
c_up[2,3] = -1

c_down = np.zeros((4,4),dtype=np.complex_) 
c_down[0,2] = 1
c_down[1,3] = 1



#_______Define a number of functions needed to compute the many-body wavefunction of the influence functional and the kernels of the gates.

def make_first_entry_last(B):
    dim_B = B.shape[0]
    #reorder entries such that input at time zero becomes last entry:
    B_reshuf = np.zeros((dim_B,dim_B),dtype=np.complex_)
    B_reshuf[:dim_B-1,:dim_B-1] = B[1:dim_B,1:dim_B]
    B_reshuf[dim_B-1,:dim_B-1] = B[0,1:dim_B]
    B_reshuf[:dim_B-1,dim_B-1] = B[1:dim_B,0]
    B_reshuf[dim_B-1,dim_B-1] = B[0,0]
    
    return B_reshuf

def IF_many_body(B):
    """
    Computes the coefficients of the many-body wavefunction based on the Slater determinant.
    To be consistent with the gates, it is imporant that
    -- the ingoing variable at time zero has been moved to the last position, and
    -- the ordering of the variables must be in increasing order on the imaginary time contour, 
        i.e. the entry B[0,0] corresponds to outgoing variable at time 0 (the ingoing variable has been moved to the last position).

    Parameters:
    - B (numpy.ndarray): A square numpy array representing the matrix whose submatrices' Pfaffians
      determine the coefficients of the many-body wavefunction. 

    Returns:
    - numpy.ndarray: An array of complex numbers representing the coefficients of the many-body
      wavefunction in binary order. The element coeffs[0] thus desribes the coefficient of the vacuum state,
        coeffs[1] desribes the coefficient for the state with binary index [0,0,0,...,1], etc.
        Note: Only entries with an even number of '1's will be non-zero as the IF if Gaussian so all Grassmann variables come in pairs.
    """
    # Validate input
    if not isinstance(B, np.ndarray) or len(B.shape) != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("Input B must be a square numpy array.")

    n_qubits = B.shape[0]#number of variables in the exponent. Note that this is twice the number of time steps as each time step and an ingoing and an outgoing variable.
    len_IM = 2**n_qubits  # Total number of entries of the IF viewed as a MB-wavefunction
    coeffs = np.zeros(len_IM, dtype=np.complex_)  # Store the components of the (unnormalized) many-body wavefunction
    coeffs[0] = 1  # Initialize the first coefficient: The vaccum state has coefficient 1.

    for i in range(1, len_IM):
        # Binary representation of the wavefunction index to determine the submatrix of B.
        # The binary representation is padded with zeros from the left to have the correct number of bits.
        binary_index = np.binary_repr(i, width=n_qubits)  # e.g. '0011' for i = 3 with n_qubits = 4

        # Indices of '1's in binary_index indicate which rows/columns of B to use.
        read_out_indices = [j for j, bit in enumerate(binary_index) if bit == '1'] # This array will look something like e.g. [0, 2, 3] for binary_index = '1011'

        if len(read_out_indices) % 2 == 0:
            #compute the submatrix of B based on the positions of the '1's in the binary representation
            B_sub = B[np.ix_(read_out_indices, read_out_indices)]
            # compute the pfaffian of the submatrix and store it as the coefficient of the many-body wavefunction
            coeffs[i] = pf.pfaffian(B_sub)

    return coeffs




def operator_to_kernel(gate_coeffs: np.ndarray, string: bool = False, boundary: bool = False) -> np.ndarray:
    """
    Function to convert a many-body gate to the 'dual' kernel representation that is used for the MPS-MPO-MPS contraction.

    Parameters:
    - gate_coeffs (numpy.ndarray): A (4x4) numpy array representing the matrix of the many-body gate.
      The matrix must adhere to the following convention for the usual quantum many-body gate:
      gate = v^\dagger @ gate_coeffs @ v, with: v = [<0|, <0|c_up , <0|c_down, <0|c_upc_down].

    - string (bool): The single-down-Grassmann entries in the gates that lie between to impurity operators 
        that contains an odd number of Grassmann variables must be multiplied by -1 in order to give a correct global sign.
        For these gates, string (because this corresponds to a fermionic Jordan-Wigner string) must be set to True.

    - boundary (bool): If True, the kernel will be constructed with antiperiodic boundary conditions. Must be set to True for the last impurity gate.
       
    Returns:
    - numpy.ndarray: A (4x4) numpy array representing the kernel of the many-body gate in the 'dual' representation.
        The kernel is constructed in the convention:
        kernel_Grassmann = \eta_down^T @ kernel @ \eta_up,
        where \eta_down = [1, \bar{\eta}_down, \eta_down, \bar{\eta}_down \eta_down],
           and \eta_up = [1, \bar{\eta}_up, \eta_up, \bar{\eta}_up \eta_up].
    """
    kernel = np.zeros((4,4),dtype=np.complex_)

    # The kernel is constructed based on the coefficients of the many-body gate.
    # Note: The signs that come from the variable substitutions are already included here! 
 
    kernel[0,0] = gate_coeffs[0,0]
    kernel[0,1] = - gate_coeffs[1,0]
    kernel[0,2] = gate_coeffs[0,1]
    kernel[0,3] = - gate_coeffs[1,1]
    kernel[1,0] = gate_coeffs[2,0]
    kernel[1,1] = - gate_coeffs[3,0]
    kernel[1,2] = gate_coeffs[2,1]
    kernel[1,3] = - gate_coeffs[3,1]
    kernel[2,0] = - gate_coeffs[0,2]
    kernel[2,1] = - gate_coeffs[1,2]
    kernel[2,2] = gate_coeffs[0,3]
    kernel[2,3] = gate_coeffs[1,3]
    kernel[3,0] = - gate_coeffs[2,2]
    kernel[3,1] = - gate_coeffs[3,2]
    kernel[3,2] = gate_coeffs[2,3]
    kernel[3,3] = gate_coeffs[3,3]

    
    if boundary == True: #antiperiodic boundary conditions: All entries with exactly one 'barred' Grassmann variable must be multiplied by -1.
        kernel[[i for i in [1,3]],:] *= -1
        kernel[:,[i for i in [1,3]]] *= -1
    
    # If the gate is a string gate, the single-down-Grassmann entries must be multiplied by -1.
    # Factor of complex i is included here for the reversing of the spin-down Grassmann variables. 
    # This is only relevant if one deals with spin-hopping Hamiltonians, i.e. Hamiltonians that can be odd within a single spin species. 
    # Otherwise, this is equivalent to the gates given in the "recipe"-notes.
    
    string_var = -1. if string == True else 1.

    kernel = np.diag([1, string_var * 1.j,string_var * 1.j,1]) @ kernel 
   
    return kernel

def Hamiltonian(E_up = 0, E_down = 0, t = 0, U = 0):
    """
    Construct the Hamiltonian that describes the local (interacting) impurity dynamics.

    The Hamiltonian is constructed in the following way:
    H = E_up * n_up + E_down * n_down + t * (c_up^dagger c_down + c_down^dagger c_up) + U * n_up * n_down

    Parameters:
    - E_up (float): The energy of the spin-up impurity level.
    - E_down (float): The energy of the spin-down impurity level.
    - t (float): The hopping amplitude between the impurity levels.
    - U (float): The interaction strength between the impurity levels.

    Returns:
    - numpy.ndarray: A (4x4) numpy array representing the Hamiltonian.
        The convention is:
        H = v^\dagger @ H @ v, with: v = [<0|, <0|c_up , <0|c_down, <0|c_up c_down].

    """

    H = np.zeros((4,4),dtype=np.complex_)#define time evolution Hamiltonian

    #spin hopping
    H += t * (c_up_dag @ c_down + c_down_dag @ c_up)

    #impurity energies and interaction
    H += E_up * c_up_dag @ c_up + E_down * c_down_dag @ c_down + U * (c_up_dag @ c_up) @ (c_down_dag @ c_down)

    return H


def compute_propagator(IF_MB: np.ndarray, U_evol: np.ndarray, dim_B: int, operator_0, operator_tau) -> np.ndarray:
    """
    Compute the propagator of the impurity model based on the many-body wavefunction of the influence functional and the time-evolution operator of the impurity model.

    Parameters:
    - IF_MB (numpy.ndarray): An array of complex numbers representing the many-body wavefunction of the influence functional.
    - U_evol (numpy.ndarray): A (4x4) numpy array representing the 'dual' and 'sign-adjusted' time-evolution operator defined by the impurity Hamiltonian.
    - dim_B (int): The number of variables in the influence functional.
    - operator_0 (numpy.ndarray): A (4x4) numpy array representing the operator at time 0. Expected to be odd in Grassmann variables, e.g. a creation/anihilation operator. Otherwise strings below should be adapted
    - operator_tau (numpy.ndarray): A (4x4) numpy array representing the operator at time tau. Expected to be odd in Grassmann variables, e.g a creation/anihilation operator. Otherwise strings below should be adapted

    Returns:
    - tuple(numpy.ndarray): A tuple of vectors of complex numbers representing the propagator of the impurity model for both spin species, <c_up(tau) c_up^\dagger(0)>.
    """
    
    # partition sum (needed for normalization)
    #create the local gate as tensor product of the evolution operator where the final gate contains antiperiodic boundary conditions
    gate_big_Z = operator_to_kernel(U_evol,boundary = True)#gate_boundary_cdag_up
    for _ in range (dim_B//2-1):
        gate_big_Z = np.kron(operator_to_kernel(U_evol), gate_big_Z)
    #compute partition sum as overlap of the many-body wavefunction of the influence functional with the gate
    Z = IF_MB @ gate_big_Z @ IF_MB
    
    G = []#store the propagator
    #Now we can compute the propagator of the impurity model
    for tau in range (dim_B//2-1):
        
        gate_big = operator_to_kernel(operator_0 @U_evol, boundary = True)# Start with the final evolution operator which is combined with the operator at time 0 (which has been brought to the last position trhough cyclicity of the trace). Don't forget antiperiodic boundary conditions
       
        for _ in range (dim_B//2-1 - tau - 1):#add the remaining evolution operators. These contain a string
            gate_big = np.kron(operator_to_kernel(U_evol, string = True),gate_big)
           
        gate_big = np.kron(operator_to_kernel(operator_tau @ U_evol, string = True),gate_big)#add the second operator at some intermediate time step. Also this gate, contains the string variable.
       
        for _ in range (tau):#fill up with the remaining evolution operators up to time 0. No string here.
            gate_big = np.kron(operator_to_kernel(U_evol), gate_big)
           
        #compute the propagator as the overlap of the many-body wavefunction of the influence functional with the gate
        G.append(IF_MB @ gate_big @ IF_MB)

    #normalize and return as numpy array:
    G = np.array(G)/Z

    return G

if __name__ == "__main__":

    #___Import the matrix B that defines the influence functional
    filename = '/Users/julianthoenniss/Documents/PhD/data/B_imag_specdens_propag_GM'
    with h5py.File(filename + '.hdf5', 'r') as f:
        B_data = f['B']
        B = B_data[:,:]
        dim_B = B.shape[0]
    
    #Make sure that the matrix B is in the correct format for the computation of the many-body wavefunction of the influence functional:
    #the ordering of the variables must be in increasing order on the imaginary time contour.
    B = B[::-1,::-1] #we usually store the matrix B in the reverse order, such that we 're-reverse' it here.
    # Moverover the ingoing variable at time zero must be moved to the last position.
    B = make_first_entry_last(B)

    #Compute the many-body wavefunction of the influence functional
    IF_MB = IF_many_body(B)


    # ____Construct the local impurity Hamiltonian:
    # Hamiltonian parameters:
    E_up = 0.3
    E_down = 0.1
    t = 0.2
    U = 0
    #Define time step:
    delta_tau = 1.0

    # create many-body Hamiltonian for the Anderson impurity model
    Ham_Anderson = Hamiltonian(E_up = E_up, E_down = E_down, U=U) 
    # create many-body Hamiltonian for the spin-hopping model
    Ham_spin_hopping = Hamiltonian(t=t) 

    # ____Define the impurity time-evolution operator. 
    #Here, we define a successive application of the spin-hopping and Anderson impurity Hamiltonians.
    U_Anderson = expm(- Ham_Anderson * delta_tau)
    U_spin_hopping = expm(- Ham_spin_hopping * delta_tau)
    U_evol = U_spin_hopping @ U_Anderson #define a combined time evolution operator


    
    # ____Compute the propagator of the impurity model
    time_grid = np.arange(1, dim_B//2) * delta_tau #define the time grid for printed output

    #Spin up:
    G_up = compute_propagator(IF_MB=IF_MB, U_evol=U_evol, dim_B=dim_B, operator_0=c_up_dag, operator_tau=c_up)
    G_down = compute_propagator(IF_MB=IF_MB, U_evol=U_evol, dim_B=dim_B, operator_0=c_down_dag, operator_tau=c_down)

    for tau, G_up, G_down in zip(time_grid, G_up, G_down):
        print(f"At tau = {np.round(tau,1)}: G_up = {G_up},  G_down = {G_down}")




