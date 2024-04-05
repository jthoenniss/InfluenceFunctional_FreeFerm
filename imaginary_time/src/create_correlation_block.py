import numpy as np
import h5py
from pfapack import pfaffian as pf
from scipy import linalg

np.set_printoptions(suppress=False, linewidth=np.nan)

def create_correlation_block_Schur(B):
    
    dim_B = B.shape[0]

    #_______The following block computes the correlation matrix by explicitly evaluating the correlation functions in the basis where the matrix B is in Schur form____
    #_______Afterwards, the correlation matrix is rotated back to the original basis____
    #_______The resulting correlation matrix is in block form [[< c c^\dagger> ,< c c>],[< c^\dagger c^\dagger> ,< c^\dagger c>]], where each of the four blocks has size (dim_b x dim_B)____
    #_______This procedure works well but may be numerically unstable when the matrix B has many degenerate eigenvalues____
    #_______Therefore, in practice, it can be advantageous to use the more stable procedure below which compute the correlation matrix by evaluating Pfaffians____
    #_______Results are equivalent____

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
    #Attention: At this point, the correlation matrix "corr_block_back_rotated" is in block form [[< c c^\dagger> ,< c c>],[< c^\dagger c^\dagger> ,< c^\dagger c>]], where each of the four blocks has size (dim_b x dim_B). 

    return corr_block_back_rotated

def create_correlation_block_Pfaffian(B):
    #_______The following block computes the correlation matrix by evaluating Pfaffians____
    #_______The resulting correlation matrix is in block form [[< c c^\dagger> ,< c c>],[< c^\dagger c^\dagger> ,< c^\dagger c>]], where each of the four blocks has size (dim_b x dim_B)____
    #_______This procedure is more stable than the one above and works well even when the matrix B has many degenerate eigenvalues____
    #_______Results are equivalent to the above procedure____

    dim_B = B.shape[0]
    # We want to evaluate expectation values of the form <c_a c_b^\dagger>  = <vac| e^{\sum_{ij} (B_ij)^\dagger c_i c_j} c_a c_b^\dagger e^{\sum_{ij} B_kl c_k^\dagger c_l^\dagger} |vac>
    #Combine both, B and B^\dagger, into one large matrix B_large, such that the expectation value can be written as Pfaffian of the inverse of B_large
    B_large = np.zeros((4*dim_B, 4*dim_B), dtype=np.complex_)
    B_large[:dim_B, :dim_B] = B.T.conj()*0.5 
    B_large[3*dim_B:, 3*dim_B:] = 0.5*B 

    #add the Grassmann measure to the matrix B_large
    for i in range(3):
        i_n = (i+1)*dim_B
        B_large[i_n:i_n+dim_B, i_n-dim_B:i_n] = -0.5 * np.eye(dim_B) 
        B_large[i_n-dim_B:i_n, i_n:i_n+dim_B] = 0.5 * np.eye(dim_B) 

    #invert the large matrix B_large
    B_large_inv = linalg.inv(B_large)

    #initialize the correlation matrix in block form
    corr_pfaff = np.zeros((2*dim_B,2*dim_B),dtype=np.complex_)
    #anticommute fermions in first block which gives rise to the identity matrix, and compute remaining part via Pfaffian
    corr_pfaff[:dim_B,:dim_B] = np.eye(dim_B) 

    #Now, evaluate the different correlation functions by computing Pfaffians of the inverse of the large matrix B_large
    #Here, no change of basis is necessary as the Pfaffians are directly evaluated in the original basis
    for i in range (dim_B):
        for j in range (dim_B):
            corr_pfaff[i,j] +=   0.5 *pf.pfaffian(B_large_inv.T[np.ix_([i+2*dim_B,j+dim_B], [i+2*dim_B,j+dim_B])])
            corr_pfaff[i+dim_B,j+dim_B] +=  0.5 *pf.pfaffian(B_large_inv.T[np.ix_([i+dim_B,j+2*dim_B], [i+dim_B,j+2*dim_B])])
            corr_pfaff[i,j+dim_B] +=  0.5 *pf.pfaffian(B_large_inv.T[np.ix_([i+2*dim_B,j+2*dim_B], [i+2*dim_B,j+2*dim_B])])
            corr_pfaff[i+dim_B,j] +=  0.5 *pf.pfaffian(B_large_inv.T[np.ix_([i+dim_B,j+dim_B], [i+dim_B,j+dim_B])])

    corr_block_back_rotated = corr_pfaff # assign to new variable in order to highlight relationship to the Schur method above. No change of basis necessary here.
    
    #Attention: At this point, the correlation matrix "corr_block_back_rotated" is in block form [[< c c^\dagger> ,< c c>],[< c^\dagger c^\dagger> ,< c^\dagger c>]], where each of the four blocks has size (dim_b x dim_B). 
    return corr_block_back_rotated

def reshuffle_corr_matrix(corr_matrix):
    """
    Input:
    corr_matrix: correlation matrix in block form [[< c c^\dagger> ,< c c>],[< c^\dagger c^\dagger> ,< c^\dagger c>]], where each of the four blocks has size (dim_b x dim_B).
    Output:
    corr_block_back_rotated_reshuf: reshuffled correlation matrix in the form Lambda = [[Lambda_{00}, Lambda_{01}, ...],[Lambda_{10}, Lambda_{11}, ...], ...], where Lambda_{ij} = [[< c_i c_j^\dagger>, <c_i c_j>], [< c_i^\dagger c_j^\dagger> ,< c_i^\dagger c_j>]]. 
    """
    dim_B = corr_matrix.shape[0]//2
    corr_block_back_rotated_reshuf = np.zeros(corr_matrix.shape,dtype=np.complex_)
    corr_block_back_rotated_reshuf[::2,::2] = corr_matrix[:dim_B,:dim_B]
    corr_block_back_rotated_reshuf[::2,1::2] = corr_matrix[:dim_B,dim_B:]
    corr_block_back_rotated_reshuf[1::2,::2] = corr_matrix[dim_B:,:dim_B]
    corr_block_back_rotated_reshuf[1::2,1::2] = corr_matrix[dim_B:,dim_B:]

    return corr_block_back_rotated_reshuf

def create_correlation_block(B):
    #compute correlation matrices by using Pfaffian formula
    corr_matrix_Pfaffian = create_correlation_block_Pfaffian(B)
    #reshuffle to bring into form Lambda = [[Lambda_{00}, Lambda_{01}, ...],[Lambda_{10}, Lambda_{11}, ...], ...], where Lambda_{ij} = [[< c_i c_j^\dagger>, <c_i c_j>], [< c_i^\dagger c_j^\dagger> ,< c_i^\dagger c_j>]].
    corr_matrix_reshuf = reshuffle_corr_matrix(corr_matrix_Pfaffian)

    return corr_matrix_reshuf

def store_correlation_matrix(corr_matrix, filename):

    #store correlation matrix in hdf5 file
    filename_correlations =  filename + '_correlations'
    with h5py.File(filename_correlations + ".hdf5", 'a') as f:
        dset_corr = f.create_dataset('corr_dim='+ str(corr_matrix.shape[0]), (corr_matrix.shape[0],corr_matrix.shape[1]),dtype=np.complex_)
        dset_corr[:,:] = corr_matrix[:,:]

    print('Correlations stored successfully.')


if __name__ == "__main__":
    
    #test that both ways of computing the correlation matrix give the same result

    #generate random antisymmetric matrix B
    dim_B = 10
    B = np.random.rand(dim_B,dim_B) 
    #make B antisymmetric
    B = B - B.T

    #compute correlation matrices by:
    #1. explicitly rotating B into Schur form and then back to the original basis
    corr_block_Schur = create_correlation_block_Schur(B)
    #2. evaluating Pfaffians
    corr_block_Pfaffian = create_correlation_block_Pfaffian(B)

    #check that both ways give the same result
    corr_matrices_equal = np.allclose(corr_block_Schur, corr_block_Pfaffian)
    if (corr_matrices_equal):
        print('The correlation matrices coincide.')
    else:
        print('The correlation matrices do not coincide.')
        print('Max. difference is: ' + str(np.max(np.abs(corr_block_Schur - corr_block_Pfaffian))))


    #reshuffle correlation matrix to obtain the form Lambda = [[Lambda_{00}, Lambda_{01}, ...],[Lambda_{10}, Lambda_{11}, ...], ...], where Lambda_{ij} = [[< c_i c_j^\dagger>, <c_i c_j>], [< c_i^\dagger c_j^\dagger> ,< c_i^\dagger c_j>]].
    corr_reshuf = reshuffle_corr_matrix(corr_block_Pfaffian)

    #store correlation matrix in hdf5 file
    store_correlation_matrix(corr_reshuf, '../data/test')



