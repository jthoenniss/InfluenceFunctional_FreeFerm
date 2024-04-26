import unittest
import numpy as np
import os,sys
parent_dir = os.path.join(os.path.dirname(__file__),"../..")
#append parent directory to path
sys.path.append(parent_dir)
from src.shared_modules.many_body_operator import many_body_operator, annihilation_ops


class TestAnnihilationsOps(unittest.TestCase):
    """
    Test the function to generate the annihilation operators.
    """

    def setUp(self) -> None:
        
        #generate the annihilation operators for 2
        self.annihilation_ops_twosite = annihilation_ops(n_ferms = 2)

        #generate the annihilation operators for 5 sites
        self.annihilation_ops = annihilation_ops(n_ferms = 5)
        

    def test_annihilation_ops(self):

        #check the annihilation opeators for 2 sites explicitly:

        c_1 = np.zeros((4,4),dtype=np.complex_) 
        c_1[0,2] = 1
        c_1[1,3] = 1
        
        c_2 = np.zeros((4,4),dtype=np.complex_) 
        c_2[0,1] = 1
        c_2[2,3] = -1

        self.assertTrue(np.allclose(self.annihilation_ops_twosite[0],c_1), f"The first annihilation operator is not computed correctly,\n{c_1},\n{self.annihilation_ops_twosite[0]}\n{self.annihilation_ops_twosite[1]}\n{c_2}")
        self.assertTrue(np.allclose(self.annihilation_ops_twosite[1],c_2), f"The second annihilation operator is not computed correctly,\n{c_2},\n{self.annihilation_ops_twosite[1]}")
        
        #check internal consistency by comparing complicated many-body operators to operator generated by combination of annhilation operators
        #check 1:
        MB_op = many_body_operator(output_ferms=[1,0,0,1,0], input_ferms=[0,1,1,0,1]) 
        MB_op_check = self.annihilation_ops[0].T @ self.annihilation_ops[3].T @ self.annihilation_ops[1] @ self.annihilation_ops[2] @ self.annihilation_ops[4]

        #check 2:
        MB_op = many_body_operator(output_ferms=[1,0,1,1,0], input_ferms=[1,1,1,0,1]) 
        MB_op_check = self.annihilation_ops[0].T @ self.annihilation_ops[2].T @ self.annihilation_ops[3].T @ self.annihilation_ops[0] @ self.annihilation_ops[1] @ self.annihilation_ops[2] @ self.annihilation_ops[4]


        self.assertTrue(np.allclose(MB_op,MB_op_check), "The many-body operator is not computed correctly")
if __name__ == "__main__":
        unittest.main()