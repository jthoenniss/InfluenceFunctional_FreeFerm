import unittest
import numpy as np
import os,sys
parent_dir = os.path.join(os.path.dirname(__file__),"../..")
#append parent directory to path
sys.path.append(parent_dir)
from src.real_time.single_orbital.dual_density_matrix import half_evolve_dual_density_matrix, dual_density_matrix_to_operator
from src.shared_modules.dual_kernel import operator_to_kernel

class TestDualDensityMatrix(unittest.TestCase):

    def setUp(self) -> None:
        
        #set up a random 4x4 density matrix
        self.density_matrix = np.random.rand(4,4)

        #compute the dual density matrix
        self.dual_density_matrix = operator_to_kernel(self.density_matrix, branch='b')

    def test_dual_density_matrix(self):

        print("Testing the dual density matrix")
        #transform the dual density matrix back to the density matrix 
        density_matrix = dual_density_matrix_to_operator(dual_density_matrix=self.dual_density_matrix)

        #check if it is the same as the original density matrix
        self.assertTrue(np.allclose(density_matrix,self.density_matrix), f"The dual density matrix is not computed correctly:\n {np.real(density_matrix)} \n!= {self.density_matrix}")



class Testhalf_evolve_dual_density_matrix(unittest.TestCase):
     
    def setUp(self) -> None:
         
        #set up a random 4x4 density matrix
        self.density_matrix = np.random.rand(4,4)

        #compute the dual density matrix
        self.dual_density_matrix = operator_to_kernel(self.density_matrix, branch='b')

    def test_half_evolve_dual_density_matrix(self):
        print("Testing the half time step evolution of the dual density matrix")
        #evolve the dual density matrix by two half a time step and assert that is returns the original density matrix

        #evolve the dual density matrix by half a trivial IF step
        dual_density_matrix_half = half_evolve_dual_density_matrix(self.dual_density_matrix, step_type='IF')
        #evolve the dual density matrix by half a trivial impurity gate
        dual_density_matrix = half_evolve_dual_density_matrix(dual_density_matrix_half, step_type='imp')

        #compate to self.dual_density_matrix
        self.assertTrue(np.allclose(dual_density_matrix,self.dual_density_matrix), f"The dual density matrix is not computed correctly:\n {np.real(dual_density_matrix)} \n!= {self.dual_density_matrix}")


class TestDensityMatrixToOperator(unittest.TestCase):
     
    def setUp(self) -> None:
         
        #set up a random 4x4 density matrix
        self.density_matrix = np.random.rand(4,4)

        #compute the dual density matrix
        self.dual_density_matrix = operator_to_kernel(self.density_matrix, branch='b')
        #half-step evolved DM
        self.half_evolved_dual_density_matrix = half_evolve_dual_density_matrix(self.dual_density_matrix, step_type='IF')

    def test_dual_density_matrix(self):
        print("Testing the dual density matrix to operator conversion")
        #transform the dual density matrix back to the density matrix 
        density_matrix = dual_density_matrix_to_operator(dual_density_matrix=self.dual_density_matrix, step_type='full')
        #transform the half-evolved dual density matrix back to the density matrix 
        half_evolved_density_matrix = dual_density_matrix_to_operator(dual_density_matrix=self.half_evolved_dual_density_matrix, step_type='half')

        #check if it is the same as the original density matrix
        self.assertTrue(np.allclose(density_matrix,self.density_matrix), f"The dual density matrix is not computed correctly:\n {np.real(density_matrix)} \n!= {self.density_matrix}")
        self.assertTrue(np.allclose(half_evolved_density_matrix,self.density_matrix), f"The half-evolved dual density matrix is not computed correctly:\n {np.real(half_evolved_density_matrix)} \n!= {self.density_matrix}")
          

          
if __name__ == "__main__":
        unittest.main()