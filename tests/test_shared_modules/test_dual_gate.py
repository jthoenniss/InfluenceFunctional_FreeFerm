import unittest
import numpy as np
import os,sys
parent_dir = os.path.join(os.path.dirname(__file__),"../..")
#append parent directory to path
sys.path.append(parent_dir)
from src.shared_modules.dual_kernel import transform_backward_kernel, dual_kernel, overlap_signs, operator_to_kernel, inverse_dual_kernel, string_in_kernel, imaginary_i_for_global_reversal, inverse_imaginary_i_for_global_reversal, sign_for_local_reversal, inverse_operator_to_kernel

class TestBackwardKernel(unittest.TestCase):

    def setUp(self) -> None:

        #set up a random 4x4 gate
        self.gate_coeffs = np.random.rand(4,4)

        #compute the backward kernel
        self.backward_kernel = transform_backward_kernel(self.gate_coeffs)

    def test_backward_kernel(self):

        #compare to analytically known backward kernel
        backward_kernel = np.array([[self.gate_coeffs[0,0], self.gate_coeffs[0,2], self.gate_coeffs[0,1], - self.gate_coeffs[0,3]],
                                    [self.gate_coeffs[2,0], self.gate_coeffs[2,2], self.gate_coeffs[2,1], - self.gate_coeffs[2,3]],
                                    [self.gate_coeffs[1,0], self.gate_coeffs[1,2], self.gate_coeffs[1,1], - self.gate_coeffs[1,3]],
                                    [- self.gate_coeffs[3,0], - self.gate_coeffs[3,2], - self.gate_coeffs[3,1], self.gate_coeffs[3,3]]])

        #compare the two kernels
        self.assertTrue(np.allclose(self.backward_kernel,backward_kernel), "The backward kernel is not computed correctly")



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


class TestSignForLocalReversal(unittest.TestCase):

    def setUp(self) -> None:

        #set up a random 16x16 gate
        self.gate_coeffs_16 = np.random.rand(16, 16)
        self.gate_coeffs_34 = np.random.rand(34, 34)

        #include signs for local reversal (signs for which reversal of output fermions picks up a sign)
        signed_idcs = lambda size: np.array([-1 if (bin(i).count('1')//2)%2 == 1 else 1 for i in range(size)])

        #multiply the rows with odd number of Grassmann variables with -1
        self.gate_coeffs_16_expected = self.gate_coeffs_16 * signed_idcs(16)[:,np.newaxis]
        self.gate_coeffs_34_expected = self.gate_coeffs_34 * signed_idcs(34)[:,np.newaxis]
    
    def test_sign_for_local_reversal(self):

        print("Testing the sign for local reversal function")
        #compute the gate coefficients with signs for local reversal included
        gate_coeffs_with_sign_16 = sign_for_local_reversal(self.gate_coeffs_16)
        gate_coeffs_with_sign_34 = sign_for_local_reversal(self.gate_coeffs_34)

        #compare the two gate coefficients
        self.assertTrue(np.allclose(gate_coeffs_with_sign_16,self.gate_coeffs_16_expected), f"The sign for local reversal function is not computed correctly (16), got {gate_coeffs_with_sign_16} and expected {self.gate_coeffs_16_expected}")
        self.assertTrue(np.allclose(gate_coeffs_with_sign_34,self.gate_coeffs_34_expected), f"The sign for local reversal function is not computed correctly (34), got {gate_coeffs_with_sign_34} and expected {self.gate_coeffs_34_expected}")




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



class Test_Inverse_Operator_To_Kernel(unittest.TestCase):
    def setUp(self) -> None:
        
        #set up a random 4x4 gate
        self.gate_coeffs = np.random.rand(4,4)

        #compute the dual gate
        self.dual_kernel1 = operator_to_kernel(self.gate_coeffs)
        self.dual_kernel2 = operator_to_kernel(self.gate_coeffs, boundary= True)
        self.dual_kernel3 = operator_to_kernel(self.gate_coeffs, branch='b')
        self.dual_kernel4 = operator_to_kernel(self.gate_coeffs, boundary=True, branch='b')
        self.dual_kernel5 = operator_to_kernel(self.gate_coeffs, string=True)
        self.dual_kernel6 = operator_to_kernel(self.gate_coeffs, string=True, boundary=True)
        self.dual_kernel7 = operator_to_kernel(self.gate_coeffs, string=True, branch='b')
        self.dual_kernel8 = operator_to_kernel(self.gate_coeffs, string=True, boundary=True, branch='b')

    def test_inverse_dual_kernel(self):
        print("Testing the inverse dual kernel function")

        #invert operator to kernel for every case
        kernel1 = inverse_operator_to_kernel(self.dual_kernel1)
        kernel2 = inverse_operator_to_kernel(self.dual_kernel2, boundary=True)
        kernel3 = inverse_operator_to_kernel(self.dual_kernel3, branch='b')
        kernel4 = inverse_operator_to_kernel(self.dual_kernel4, boundary=True, branch='b')
        kernel5 = inverse_operator_to_kernel(self.dual_kernel5, string=True)
        kernel6 = inverse_operator_to_kernel(self.dual_kernel6, string=True, boundary=True)
        kernel7 = inverse_operator_to_kernel(self.dual_kernel7, string=True, branch='b')
        kernel8 = inverse_operator_to_kernel(self.dual_kernel8, string=True, boundary=True, branch='b')

        #check that inv_dual_gate indeed recovers the original gate, self.gate_coeffs
        self.assertTrue(np.allclose(kernel1,self.gate_coeffs), "The inverse dual kernel function is not computed correctly")
        self.assertTrue(np.allclose(kernel2,self.gate_coeffs), "The inverse dual kernel function is not computed correctly")
        self.assertTrue(np.allclose(kernel3,self.gate_coeffs), "The inverse dual kernel function is not computed correctly")
        self.assertTrue(np.allclose(kernel4,self.gate_coeffs), "The inverse dual kernel function is not computed correctly")
        self.assertTrue(np.allclose(kernel5,self.gate_coeffs), "The inverse dual kernel function is not computed correctly")
        self.assertTrue(np.allclose(kernel6,self.gate_coeffs), "The inverse dual kernel function is not computed correctly")
        self.assertTrue(np.allclose(kernel7,self.gate_coeffs), "The inverse dual kernel function is not computed correctly")
        self.assertTrue(np.allclose(kernel8,self.gate_coeffs), "The inverse dual kernel function is not computed correctly")
       
if __name__ == "__main__":
        unittest.main()