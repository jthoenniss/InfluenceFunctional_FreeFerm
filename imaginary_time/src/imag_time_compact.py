import numpy as np
from scipy import linalg
from imag_time_IM_funcs import g_greater, spec_dens
import scipy.integrate as integrate # to solve frequency integrals when computing the propagator from the spectral density

global_gamma = 1.#global energyscale -> set to 1


def single_mode_GF(t_hop,tau):
    #Compute the exact greens function for a single-mode environment with E_k = 0. Computed by Fourier transforming the non-interacting impurity GF as Matsubara sum.
    return -(np.exp(-t_hop * tau)/2 + np.sinh(t_hop * tau)*1/(1+np.exp(t_hop * beta)))#analytical solution of non-interacting greens function for single-mode environment with E_k = 0



def compute_continuous_time_IF(g: np.ndarray):

    """
    The part below takes the array g[] and spits out the exponent of the IF in the continuous-time limit.
    Parameters:
    - g: np.ndarray, shape=(nbr_steps+1), array containing the time-domain-version of the non-interacting impurity GF, evaluated at the discrete time-grid points: tau = 0, delta, 2* delta,...,beta-delta, beta
    

    Returns:
    - B_spec_dens_cont: np.ndarray, shape=(2*nbr_steps,2*nbr_steps), exponent of the IF in the continuous-time limit
    """
    nbr_steps = g.shape[0] - 1
    dim_B = 2 * nbr_steps # Size of exponent matrix in the IF. 

    #Create array G with values of g[]. From this array, we will extract certain submatrices of which we compute the determinants, yielding the components of the IF
    #the matrix G is constructed as
    # [[g[0], g[delta], g[2],...,g[M]],
    #  [-g[M-1], g[0], g[1],...g[M-1]],
    #  [-g[M-2], -g[M-1], g[0],...,g[M-2]]
    # ...
    G = np.zeros((nbr_steps+1,nbr_steps+1),dtype=np.complex_)
    for i in range (nbr_steps):
        G[i,:] = np.append(-g[nbr_steps-i:nbr_steps],g[:nbr_steps-i+1])
    G[nbr_steps,:] = np.append(-g[:nbr_steps],g[0])

    Z = -1/linalg.det(G[:-1,:-1])#partition sum, with minus sign included in such a way to cancel the minus sign of the entries in B_tau

    #by evaluating the determinant of certain submatrices of G, compute the elements of the IF. 
    B_tau = np.zeros(nbr_steps,dtype=np.complex_)#only nbr_steps evaluations are necessary. They are stored in the arrax B_tau, from which we later construct the exponent of the IF. 
    for m in range(nbr_steps-1):
        arr_bar = list(np.append(m+1,(np.delete(np.arange(1,nbr_steps),m))))#columns of G, corresponding to barred Grassmann variables in the multipoint correlation function
        arr_nobar = list(np.delete(np.arange(nbr_steps),m+1))#rows of G, corresponding to non-barred Grassmann variables in the multipoint correlation function
        G_det = G[np.ix_(arr_nobar,arr_bar)]#first array: bars, second array: non-bar
        B_tau[m] = Z * linalg.det(G_det)# multiply the determinant with the partition sum to obtain elements of IF.

    #repeat the same procedure with slight modification for the element between maximally separated variables.
    arr_bar = list(np.append(nbr_steps,np.arange(1,nbr_steps)))
    arr_nobar = list(np.arange(nbr_steps))
    G_det = G[np.ix_(arr_nobar,arr_bar)]#first array: bars, second array: non-bar
    B_tau[-1] =  - Z * linalg.det(G_det)# minus sign becase the sign in the determinant does not fully cancel the ones from GF because there is one GF more than before

    #construct IF-exponent B_spec_dens_cont from vector B_tau:
    B_spec_dens_cont = np.zeros((dim_B,dim_B),dtype=np.complex_)#exponent of IF (times 2, as previously). 
    for m in range (nbr_steps):
        for n in range(nbr_steps):
            if m > n-1:
                B_spec_dens_cont[2*m+1,2*n] = B_tau[m-n]
            else: 
                B_spec_dens_cont[2*m+1,2*n] = -B_tau[nbr_steps + m-n]
    B_spec_dens_cont -= B_spec_dens_cont.T

    #at this point, the matrix B_spec_dens is in the convention with ordering: in_0, out_0, in_1, out_1, ...,in_{M-1},out_{M-1}, i.e. as written in the "guide"

    return B_spec_dens_cont

def integrate_T_ren(B: np.ndarray, T_ren: int = 1) -> np.ndarray:
    """
    Combines T_ren time steps into one time step by integrating out the internal legs, i.e. by effectively inserting the identity gate on the impurity.
    By choosing the initial time step very small and integrating out many legs, one can obtain the continuous-time limit of the IF for a finite number of open legs.
    Parameters:
    - B: np.ndarray, shape=(dim_B_temp,dim_B_temp), exponent of the IF before integrating out internal legs
    - T_ren: int, number of time steps that are integrated out

    Returns:
    - B: np.ndarray, shape=(dim_B,dim_B), exponent of the IF after integrating out internal legs.
    """
    np.assert_(T_ren > 0, "T_ren must be an integer larger than 0")

    if T_ren == 1:
        return B
    
    elif T_ren > 1: 
        dim_B_temp = B.shape[0]#dimension of matrix B before integrating out internal legs
        dim_B = dim_B_temp // T_ren #dimension of matrix B after integrating out internal legs

        #add intermediate integration measure to integrate out internal legs
        for i in range (dim_B//2 ):
            for j in range (T_ren-1):
                B[2*i*T_ren + 1 + 2*j,2*i*T_ren+2+ 2*j] += -1  
                B[2*i*T_ren+2+ 2*j,2*i*T_ren + 1 + 2*j] += 1  
    
        #select submatrix that contains all intermediate times that are integrated out
        B_sub =  np.zeros((dim_B_temp - dim_B, dim_B_temp - dim_B),dtype=np.complex_)
        for i in range (dim_B//2 ):
            for j in range (dim_B//2 ):
                B_sub[i*(2*T_ren-2):i*(2*T_ren-2 )+2*T_ren-2,j*(2*T_ren-2):j*(2*T_ren-2 )+2*T_ren-2] = B[2*i*T_ren+1:2*(i*T_ren + T_ren)-1,2*j*T_ren+1:2*(j*T_ren + T_ren)-1]
    
        #matrix coupling external legs to integrated (internal) legs
        B_coupl =  np.zeros((dim_B_temp - dim_B,dim_B),dtype=np.complex_)
        for i in range (dim_B//2 ):
            for j in range (dim_B//2 ):
                B_coupl[i*(2*T_ren-2):i*(2*T_ren-2 )+2*T_ren-2,2*j] = B[2*i*T_ren+1:2*(i*T_ren + T_ren)-1,2*j*T_ren]
                B_coupl[i*(2*T_ren-2):i*(2*T_ren-2 )+2*T_ren-2,2*j+1] = B[2*i*T_ren+1:2*(i*T_ren + T_ren)-1,2*(j+1)*T_ren-1]

        #part of matriy that is neither integrated nor coupled to integrated variables
        B_ext = np.zeros((dim_B,dim_B),dtype=np.complex_)
        for i in range (dim_B//2 ):
            for j in range (dim_B//2 ):
                B_ext[2*i,2*j] = B[2*i*T_ren,2*j*T_ren]
                B_ext[2*i+1,2*j] = B[2*(i+1)*T_ren-1,2*j*T_ren]
                B_ext[2*i,2*j+1] = B[2*i*T_ren,2*(j+1)*T_ren-1]
                B_ext[2*i+1,2*j+1] = B[2*(i+1)*T_ren-1,2*(j+1)*T_ren-1]

    
        B = B_ext + B_coupl.T @ linalg.inv(B_sub) @ B_coupl #integrating out internal legs which amounts to a matrix inversion

        return B
    


def B_from_spec_dens(dim_B: int, beta:float, delta_tau: float,  int_lim_low: float, int_lim_up: float, alpha: float = 1.) -> np.ndarray:
    """
    Compute the exponent of the IF from the spectral density of the bath.
    Parameters:
    - dim_B: int, dimension of the exponent matrix in the IF
    - beta: float, inverse temperature
    - delta_tau: float, time step
    - int_lim_low: float, lower integration limit
    - int_lim_up: float, upper integration limit
    - alpha: float, parameter that the trotter evolution, defined as U_hyb = expm(- delta_tau * (1 - alpha) H_bath ) @ expm(- delta_tau * (H_hop + alpha H_bath)), 
        where H_hop is the hopping between impurity and bath and H_bath is only bath evolution

    Returns:
    - B_spec_dens: np.ndarray, shape=(dim_B,dim_B), exponent of the IF
    """

    B_spec_dens = np.zeros((dim_B,dim_B))
    #here, as an example we reproduce the example of the spectral density-result by first defining the hybridization vector:
    hyb = np.zeros(dim_B//2)#this vector is the vector coming out of the DMFT loop with hyb[0] corresponding to tau=0, hyb[1] corresponding to tau=1, and so on 

    #for our example, initialize hyb[] with the bath greens function as derived in the notes
    for n in range (2):
        hyb[n] = integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_greater(x,beta,(dim_B//2 + (n-1-alpha))*delta_tau, 0), int_lim_low, int_lim_up)[0]
    for n in range (2,dim_B//2):
        hyb[n] = - integrate.quad(lambda x: 1./(2*np.pi) * spec_dens(global_gamma,x) * g_greater(x,beta,(n-1-alpha)*delta_tau, 0), int_lim_low, int_lim_up)[0]
    hyb = np.append(hyb[1:],-hyb[0])#reshuffle first element to last position with negative sign, such that matrix an be initialized easier

    for m in range (B_spec_dens.shape[0]//2):
        B_spec_dens[2*m,2*m+1::2] = - delta_tau**2 * hyb[:len(hyb)-m]
        B_spec_dens[2*m+1,2*m+2::2] = - delta_tau**2 * hyb[len(hyb)-1:m:-1]
        B_spec_dens[2*m, 2*m + 1] += - 1. #constant from overlap at m=n
    B_spec_dens -= B_spec_dens.T#antisymmetrize. This is twice the exponent matrix 

    #at this point, the matrix B_spec_dens is in the convention with ordering: in_0, out_0, in_1, out_1, ...,in_{M-1},out_{M-1}, i.e. as written in the "guide"

    return B_spec_dens


def convert_to_simultaneous_evol_scheme(B):
    """
    Converts the exponent matrix B to the simultaenous evolution scheme. 
    Parameters:
    - B: np.ndarray, shape=(dim_B,dim_B), exponent of the IF in the successive evolution scheme. B must be ordered in the convention with ordering: in_0, out_0, in_1, out_1, ...,in_{M-1},out_{M-1}

    Returns:
    - B: np.ndarray, shape=(dim_B,dim_B), exponent of the IF in the simultaneous evolution scheme. B is still ordered in the convention with ordering: in_0, in_1, ...,in_{M-1}, out_0, out_1, ...,out_{M-1}
    """


    #remove overlap, it will be included in the impurity gates
    for m in range(B.shape[0]//2):
        B[2*m, 2*m + 1] -= - 1. #constant from overlap at m=n
        B[2*m+1, 2*m] -= + 1. #constant from overlap at m=n

    #account for changes in signs which are introduced in order to keep the impurity gates unchanged for both cases, a and b: change sign of all entries that have one conjugate variable 
    for i in range (B.shape[0]):
        B[i,(i+1)%2:B.shape[0]:2] *= -1 
    
    B[B.shape[0]-1, :] *= - 1. #antiperiodic b.c.
    B[:,B.shape[0]-1] *= - 1. #antiperiodic b.c.

    #identity measure for cont. time prescription
    id_meas = np.zeros(B.shape,dtype=np.complex_)
    for m in range(B.shape[0]//2-1):
        id_meas[2*m+1 , 2*(m+1)] += 1. 
    id_meas[0,B.shape[0]-1] -= 1. 

    id_meas -= id_meas.T
    B += id_meas
    B = linalg.inv(B)#invert the matrix. The result B on the left side defines the IF and corresponds to the matrix A^{-1} from the DMFT guide




if __name__ == "__main__":

    
    global_gamma = 1.#global energyscale -> set to 1 
    beta = 20/global_gamma# here, twice the value than Benedikt (?)
    Gamma = 1.



    nbr_steps =8#this is the number of Floquet steps that are performed on the level of MPS
    #T_ren =100#this is the number of "substeps" into which one time-step of the environment evolution is subdivided. The result becomes the exact continuous-time result when this parameters is large. This is only meaningful for method "a", i.e. for successive environment-impurity evolution

    #dim_B = 2 * nbr_steps # Size of exponent matrix in the IF. The variable dim_B is equal to 2*nbr_Floquet_layers
    #dim_B_temp = dim_B * T_ren
    #delta_tau = beta / (dim_B_temp/2 )#timestep

    #into this array, insert the Fourier transform of the non-interacting impurity GF, evaluated at the discrete time-grid points: tau = 0, delta, 2* delta,...,beta-delta, beta
    g = np.zeros((nbr_steps+1),dtype=np.complex_)


    # here (as an example) initialized with the analytical non-interacting GF gf() for the single-mode environment with E_k = 0.
    t_hop = np.sqrt(0.8)#hopping amplitude between bath and single environment mode
    for i in range (nbr_steps+1):
        g[i] = single_mode_GF(t_hop, i*beta/nbr_steps) 
    B_spec_dens_cont = compute_continuous_time_IF(g)



    
    