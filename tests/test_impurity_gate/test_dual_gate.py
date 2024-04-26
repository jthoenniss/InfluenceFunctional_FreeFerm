import unittest
import numpy as np
import os,sys
parent_dir = os.path.join(os.path.dirname(__file__),"../..")
#append parent directory to path
sys.path.append(parent_dir)
from src.shared_modules.dual_kernel import dual_kernel, overlap_signs, operator_to_kernel, inverse_dual_kernel

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

if __name__ == "__main__":
        unittest.main()