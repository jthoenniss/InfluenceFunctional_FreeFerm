import unittest
import numpy as np
import os,sys
#append parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__),"../.."))
from src.imaginary_time.dual_overlap_imag import Hamiltonian, operator_to_kernel
from scipy.linalg import expm


class TestHamiltonian(unittest.TestCase):
    """
    Check that the Hamiltonian is correctly set up by comparing it to a known Hamiltonian where we intialize the elements manually.
    """
    
    def setUp(self):
        #set up parameters
        self.t= 0.2
        self.E_up = .3
        self.E_down = 0.1
        self.U = 0.23

        
        self.H = np.zeros((4,4),dtype=np.complex_)#define time evolution Hamiltonian
        #spin hopping
        self.H[2,1] += self.t
        self.H[1,2] += self.t
        
        #Anderson
        self.H[1,1] += self.E_up
        self.H[2,2] += self.E_down
        self.H[3,3] += self.E_up + self.E_down + self.U

    def test_Hamiltonian(self):
        print("Testing the Hamiltonian")
        #test that the Hamiltonian is correctly set up
        Ham = Hamiltonian(E_up=self.E_up, E_down=self.E_down, t=self.t, U=self.U)
        self.assertTrue(np.array_equal(self.H, Ham))
        

class TestOperatorToKernel(unittest.TestCase):
    def setUp(self):
        #set up parameters
        self.t= 0.2
        self.E_up = .3
        self.E_down = 0.1
        self.U = 0.23

        self.delta_tau = 0.1

        #spin hopping gates
        self.Ham_hopping = Hamiltonian(t=self.t)
        self.U_evol_hopping = expm(- self.Ham_hopping * self.delta_tau)

        self.gate_hopping = np.zeros((4,4),dtype=np.complex_)
        self.gate_hopping[0,0] = 1
        self.gate_hopping[2,1] = np.sinh(self.t * self.delta_tau) * (1.j)
        self.gate_hopping[1,2] = -np.sinh(self.t * self.delta_tau) * (1.j)
        self.gate_hopping[0,3] = - np.cosh(self.t * self.delta_tau) 
        self.gate_hopping[3,0] =  - np.cosh(self.t * self.delta_tau)
        self.gate_hopping[3,3] = 1

        #Anderson gates
        self.Ham_Anderson = Hamiltonian(E_up=self.E_up, E_down=self.E_down, U=self.U)
        self.U_evol_Anderson = expm(- self.Ham_Anderson * self.delta_tau)

        self.gate_Anderson = np.zeros((4,4),dtype=np.complex_)
        self.gate_Anderson[0,0] = 1
        self.gate_Anderson[0,3] = - np.exp(- self.E_up * self.delta_tau)
        self.gate_Anderson[3,0] = - np.exp(- self.E_down * self.delta_tau) 
        self.gate_Anderson[3,3] = np.exp(-1. *(self.E_up + self.E_down + self.U) * self.delta_tau)

    def test_operator_to_kernel(self):
        print("Testing the operator-to-kernel function")
        #"dual" evolution operator
        gate_hopping = operator_to_kernel(self.U_evol_hopping)
        gate_Anderson = operator_to_kernel(self.U_evol_Anderson)

        self.assertTrue(np.allclose(gate_hopping, self.gate_hopping))
        self.assertTrue(np.allclose(gate_Anderson, self.gate_Anderson))


if __name__ == "__main__":
        unittest.main()