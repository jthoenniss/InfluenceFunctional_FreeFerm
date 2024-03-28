from rotation_matrix_for_schur import rotation_matrix_for_schur
import numpy as np
import h5py
from datetime import datetime
import pandas as pd
from pfapack import pfaffian as pf
#from memory_profiler import profile
from scipy import linalg
np.set_printoptions(suppress=False, linewidth=np.nan)
#@profile
def create_correlation_block(B, ntimes, filename):
    
    dim_B = B.shape[0]

    """
    random_part = np.random.rand(dim_B,dim_B) * 1.e-8
    B += random_part - random_part.T #add small antisymmetric part to make sure that the schur decomposition does not suffer from numerical issues to the degeneracies in B

    hermitian_matrix = B.T @ B.conj()#create hermitian matrix, whose eigenvectors define the rotation matrix that rotates B into Schur form
    eigenvalues_hermitian_matrix, R = linalg.eigh(hermitian_matrix)#compute rotation matrix as eigenvalues of hermitian_matrix
    
    B_schur_complex = R.T.conj() @ B @ R.conj() #this is the Schur form of B, where the entries are generally complex
    eigenvalues_complex = np.diag(B_schur_complex,k=1)[::2]#define Schur-eigenvalues such that the order is in correspondence with the order of the eigenvectors in R.

    #create matrix that contains the phases, which can be absorbed in R, such that  R.T.conj() @ B @ R.conj() is real and all entries in the upper right triangular matrix are positive.
    D_phases = np.zeros((dim_B, dim_B), dtype=np.complex_)
    np.fill_diagonal(D_phases[::2,::2], np.exp(0.5j * np.angle(eigenvalues_complex[:])))
    np.fill_diagonal(D_phases[1::2,1::2], np.exp(0.5j * np.angle(eigenvalues_complex[:])))

    #update rotation matrix to include phases, such that Schur-values become real
    R = R @ D_phases #R is generally complex

    B_schur_real = R.T.conj() @ B @ R.conj()#this is Schur-form of B, but now with real Schur-values
    eigenvalues_real = np.real(np.diag(B_schur_real,k=1)[::2])#define eigenvalues from Schur form of matrix such that the order is in correspondence with the order of the eigenvectors in R.

    #compute correlation block in diagonal basis with only entries this phases of fermionic operators are defined such that the eigenvalues of B are real
    corr_block_diag = np.zeros((2 * dim_B, 2 * dim_B))
    for i in range(0, dim_B // 2):
        ew = eigenvalues_real[i] 
        norm = 1 + abs(ew)**2
        corr_block_diag[2 * i, 2 * i] = 1/norm # <d_k d_k^\dagger>
        corr_block_diag[2 * i + 1, 2 * i + 1] = 1/norm # <d_{-k} d_{-k}^\dagger>
        corr_block_diag[2 * i, 2 * i + dim_B + 1] = - ew/norm # <d_k d_{-k}>
        corr_block_diag[2 * i + 1, 2 * i + dim_B] = ew/norm # <d_{-k} d_{k}>
        corr_block_diag[2 * i + dim_B, 2 * i + 1] = ew.conj()/norm #<d_{k}^dagger d_{-k}^\dagger> .. conjugation is formally correct but has no effect since eigenvalues are real anyways
        corr_block_diag[2 * i + dim_B + 1, 2 * i] = - ew.conj()/norm #<d_{-k}^dagger d_{k}^\dagger>
        corr_block_diag[2 * i + dim_B, 2 * i + dim_B] = abs(ew)**2/norm #<d_{k}^dagger d_{k}>
        corr_block_diag[2 * i + dim_B + 1, 2 * i + dim_B + 1] = abs(ew)**2/norm #<d_{-k}^dagger d_{-k}>

    #matrix that rotates the correlation block between the diagonal basis and the original fermion basis
    double_R = np.bmat([[R, np.zeros((dim_B, dim_B),dtype=np.complex_)],[np.zeros((dim_B, dim_B),dtype=np.complex_), R.conj()]])
    
    corr_block_back_rotated = double_R @ corr_block_diag @ double_R.T.conj()#rotate correlation block back from diagonal basis to original fermion basis
    """
    B_large = np.zeros((4*dim_B, 4*dim_B), dtype=np.complex_)
    B_large[:dim_B, :dim_B] = B.T.conj()*0.5
    B_large[3*dim_B:, 3*dim_B:] = 0.5*B

    for i in range(3):
        i_n = (i+1)*dim_B
        B_large[i_n:i_n+dim_B, i_n-dim_B:i_n] = -0.5 * np.eye(dim_B)
        B_large[i_n-dim_B:i_n, i_n:i_n+dim_B] = 0.5 * np.eye(dim_B)

    B_large_inv = linalg.inv(B_large)
    corr_pfaff = np.zeros((2*dim_B,2*dim_B),dtype=np.complex_)
    corr_pfaff[:dim_B,:dim_B] = np.eye(dim_B)
    for i in range (dim_B):
        for j in range (dim_B):
            corr_pfaff[i,j] +=   0.5 *pf.pfaffian(np.array(pd.DataFrame(B_large_inv.T).iloc[[i+2*dim_B,j+dim_B], [i+2*dim_B,j+dim_B]]))
            corr_pfaff[i+dim_B,j+dim_B] +=  0.5 *pf.pfaffian(np.array(pd.DataFrame(B_large_inv.T).iloc[[i+dim_B,j+2*dim_B], [i+dim_B,j+2*dim_B]]))
            corr_pfaff[i,j+dim_B] +=  0.5 *pf.pfaffian(np.array(pd.DataFrame(B_large_inv.T).iloc[[i+2*dim_B,j+2*dim_B], [i+2*dim_B,j+2*dim_B]]))
            corr_pfaff[i+dim_B,j] +=  0.5 *pf.pfaffian(np.array(pd.DataFrame(B_large_inv.T).iloc[[i+dim_B,j+dim_B], [i+dim_B,j+dim_B]]))
    corr_block_back_rotated = corr_pfaff


    #Attention: At this point, the correlation matrix "corr_block_back_rotated" is in block form [[< c c^\dagger> ,< c c>],[< c^\dagger c^\dagger> ,< c^\dagger c>]], where each of the four blocks has size (dim_b x dim_B). 
    #To bring it into form specified in pdf, we must reshuffle (below)

    corr_block_back_rotated_reshuf = np.zeros(corr_block_back_rotated.shape,dtype=np.complex_)
    corr_block_back_rotated_reshuf[::2,::2] = corr_block_back_rotated[:dim_B,:dim_B]
    corr_block_back_rotated_reshuf[::2,1::2] = corr_block_back_rotated[:dim_B,dim_B:]
    corr_block_back_rotated_reshuf[1::2,::2] = corr_block_back_rotated[dim_B:,:dim_B]
    corr_block_back_rotated_reshuf[1::2,1::2] = corr_block_back_rotated[dim_B:,dim_B:]

    #store for Benedikt:
    filename_correlations =  filename + '_correlations_Benedikt'
    with h5py.File(filename_correlations + ".hdf5", 'a') as f:
        dset_corr = f.create_dataset('corr_t='+ str(ntimes), (corr_block_back_rotated_reshuf.shape[0],corr_block_back_rotated_reshuf.shape[1]),dtype=np.complex_)
        dset_corr[:,:] = corr_block_back_rotated_reshuf[:,:]
    print('Correlations stored for Benedikt.')
    

    return corr_block_back_rotated

