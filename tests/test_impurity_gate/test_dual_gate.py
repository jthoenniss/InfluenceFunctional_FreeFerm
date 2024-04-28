import unittest
import numpy as np
import os,sys
parent_dir = os.path.join(os.path.dirname(__file__),"../..")
#append parent directory to path
sys.path.append(parent_dir)
from src.shared_modules.dual_kernel import dual_kernel, overlap_signs, operator_to_kernel, inverse_dual_kernel, dual_density_matrix_to_operator, string_in_kernel, imaginary_i_for_global_reversal, inverse_imaginary_i_for_global_reversal

class TestDualGate(unittest.TestCase):
    
    "Test that the dual gate with overlap signs is compute correctly"

    def setUp(self) -> None:
        
        #random 4x4 matrix
        self.gate_coeffs = np.random.rand(4,4)

        #compute the dual gate
        self.dual_gate = dual_kernel(self.gate_coeffs)
        #add signs from overlap
        self.dual_gate = overlap_signs(self.dual_gate)
    
    def test_dual_gate(self):
        print("Testing the dual gate")
        
        kernel = np.zeros((4,4))

        #compute the sign-adjusted dual gate by hand:
        kernel[0,0] = self.gate_coeffs[0,0]
        kernel[0,1] = - self.gate_coeffs[1,0]
        kernel[0,2] = self.gate_coeffs[0,1]
        kernel[0,3] = - self.gate_coeffs[1,1]
        kernel[1,0] = self.gate_coeffs[2,0]
        kernel[1,1] = - self.gate_coeffs[3,0]
        kernel[1,2] = self.gate_coeffs[2,1]
        kernel[1,3] = - self.gate_coeffs[3,1]
        kernel[2,0] = - self.gate_coeffs[0,2]
        kernel[2,1] = - self.gate_coeffs[1,2]
        kernel[2,2] = self.gate_coeffs[0,3]
        kernel[2,3] = self.gate_coeffs[1,3]
        kernel[3,0] = - self.gate_coeffs[2,2]
        kernel[3,1] = - self.gate_coeffs[3,2]
        kernel[3,2] = self.gate_coeffs[2,3]
        kernel[3,3] = self.gate_coeffs[3,3]

        #compare the two kernels
        self.assertTrue(np.allclose(self.dual_gate,kernel), "The dual gate is not computed correctly")


class TestStringInKernel(unittest.TestCase):

    def setUp(self) -> None:
        self.random_kernel_16 = np.random.rand(16,16)

        self.random_kernel_32 = np.random.rand(32,32)

        #include the string in the kernel analytically
        #all rows corresponding to an odd number of Grassmann is multiplied with a factor of -1
        #diagonal matrix containing -1 in the rows with odd number of Grassmann variables
        odd_rows_16 = np.diag([(-1)**bin(i).count('1') for i in range(16)])
        odd_rows_32 = np.diag([(-1)**bin(i).count('1') for i in range(32)])

        #multiply the rows with odd number of Grassmann variables with -1
        self.kernel_with_string_manual_16 = odd_rows_16 @ self.random_kernel_16
        self.kernel_with_string_manual_32 = odd_rows_32 @ self.random_kernel_32

    def test_string_in_kernel(self):

        print("Testing the string in kernel function")
        #compute the kernel with the string included
        kernel_with_string_16 = string_in_kernel(self.random_kernel_16)
        kernel_with_string_32 = string_in_kernel(self.random_kernel_32)



        #compare the two kernels
        self.assertTrue(np.allclose(kernel_with_string_16,self.kernel_with_string_manual_16), "The string in kernel function is not computed correctly")
        self.assertTrue(np.allclose(kernel_with_string_32,self.kernel_with_string_manual_32), "The string in kernel function is not computed correctly")

class TestImaginaryIForGlobalReversal(unittest.TestCase):

    def setUp(self) -> None:
        
        #set up a random 16x16 gate
        self.gate_coeffs_16 = np.random.rand(16, 16)
        self.gate_coeffs_32 = np.random.rand(32, 32)

        #include factors of imaginary i for global reversal
        diag_matrix_16 = np.diag([1.j**bin(i).count('1') for i in range(16)])
        diag_matrix_32 = np.diag([1.j**bin(i).count('1') for i in range(32)])

        #multiply the diagonal matrix with the gate coefficients
        self.gate_coeffs_with_i_16 = diag_matrix_16 @ self.gate_coeffs_16
        self.gate_coeffs_with_i_32 = diag_matrix_32 @ self.gate_coeffs_32

    def test_imaginary_i_for_global_reversal(self):
        print("Testing the imaginary i for global reversal function")
        #compute the gate coefficients with imaginary i included
        gate_coeffs_with_i_16 = imaginary_i_for_global_reversal(self.gate_coeffs_16)
        gate_coeffs_with_i_32 = imaginary_i_for_global_reversal(self.gate_coeffs_32)

        #compare the two gate coefficients
        self.assertTrue(np.allclose(gate_coeffs_with_i_16,self.gate_coeffs_with_i_16), "The imaginary i for global reversal function is not computed correctly")
        self.assertTrue(np.allclose(gate_coeffs_with_i_32,self.gate_coeffs_with_i_32), "The imaginary i for global reversal function is not computed correctly")
        
class TestInverseImaginaryIForGlobalReversal(unittest.TestCase):

    def setUp(self) -> None:
        self.random_kernel_16 = np.random.rand(16,16)
        self.random_kernel_32 = np.random.rand(32,32)

        #include factors of imaginary i for global reversal
        self.matrix_16 = imaginary_i_for_global_reversal(self.random_kernel_16)
        self.matrix_32 = imaginary_i_for_global_reversal(self.random_kernel_32)

    def test_inverse_imaginary_i_for_global_reversal(self):

        print("Testing the inverse imaginary i for global reversal function")
        #compute the kernel without imaginary i included
        kernel_without_i_16 = inverse_imaginary_i_for_global_reversal(self.matrix_16)
        kernel_without_i_32 = inverse_imaginary_i_for_global_reversal(self.matrix_32)

        #compare the two kernels
        self.assertTrue(np.allclose(kernel_without_i_16,np.real(self.random_kernel_16)), "The inverse imaginary i for global reversal function is not computed correctly")
        self.assertTrue(np.allclose(kernel_without_i_32,np.real(self.random_kernel_32)), "The inverse imaginary i for global reversal function is not computed correctly")

class TestOperatorToKernel(unittest.TestCase):
    """
    Test full construction of the operator-to-kernel function by comparing to manually computed function.
    """

    def setUp(self) -> None:
        
        #generate random 4x4 matrix
        self.gate_coeffs = np.random.rand(4,4)

        #compute the operator-to-kernel result
        self.kernel = operator_to_kernel(self.gate_coeffs)

    def test_operator_to_kernel(self):
        
        print("Testing the operator-to-kernel function")
        #analytically derived kernel from the gate coefficients:
        # [[a00, a01, a02, a03],
        #  [a10, a11, a12, a13],
        #  [a20, a21, a22, a23],
        #  [a30, a31, a32, a33]] -> [[a00, -a10, a01, -a11],
        #                            [i a20, -i a30, i a21, -i a31],
        #                            [-i a02, -i a12, i a03, i a13],
        #                            [-a22, -a32, a23, a33]].

        kernel = np.array([[self.gate_coeffs[0,0], - self.gate_coeffs[1,0], self.gate_coeffs[0,1], - self.gate_coeffs[1,1]],
                                [1.j * self.gate_coeffs[2,0], -1.j * self.gate_coeffs[3,0], 1.j * self.gate_coeffs[2,1], -1.j * self.gate_coeffs[3,1]],
                                [-1.j * self.gate_coeffs[0,2], -1.j * self.gate_coeffs[1,2], 1.j * self.gate_coeffs[0,3], 1.j * self.gate_coeffs[1,3]],
                                [- self.gate_coeffs[2,2], - self.gate_coeffs[3,2], self.gate_coeffs[2,3], self.gate_coeffs[3,3]]])


        #compare the two kernels
        self.assertTrue(np.allclose(self.kernel,kernel), "The operator-to-kernel function is not computed correctly")
        
 

class TestInverseDualKernel(unittest.TestCase):
     
    def setUp(self) -> None:
        
        #set up a random 4x4 gate
        self.gate_coeffs = np.random.rand(4,4)

        #compute the dual gate
        self.dual_gate = dual_kernel(self.gate_coeffs)

    def test_inverse_dual_kernel(self):
        print("Testing the inverse dual kernel function")
        #compute the inverse dual kernel
        inv_dual_gate = inverse_dual_kernel(self.dual_gate)

        #check that inv_dual_gate indeed recovers the original gate, self.gate_coeffs
        self.assertTrue(np.allclose(inv_dual_gate,self.gate_coeffs), "The inverse dual kernel function is not computed correctly")

class TestDualDensityMatrix(unittest.TestCase):

    def setUp(self) -> None:
        
        #set up a random 4x4 density matrix
        self.density_matrix = np.random.rand(4,4)

        #compute the dual density matrix
        self.dual_density_matrix = operator_to_kernel(self.density_matrix, branch='b')

    def test_dual_density_matrix(self):

        #transform the dual density matrix back to the density matrix 
        density_matrix = dual_density_matrix_to_operator(dual_density_matrix=self.dual_density_matrix)

        #check if it is the same as the original density matrix
        self.assertTrue(np.allclose(density_matrix,self.density_matrix), f"The dual density matrix is not computed correctly:\n {np.real(density_matrix)} \n!= {self.density_matrix}")

if __name__ == "__main__":
        unittest.main()