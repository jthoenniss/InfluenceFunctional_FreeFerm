import unittest
import numpy as np
import os,sys
parent_dir = os.path.join(os.path.dirname(__file__),"../..")
#append parent directory to path
sys.path.append(parent_dir)
from src.real_time.compute_impurity_gate.interleave_gate import map_interleave, interleave, dict_interleave
from scipy.linalg import expm


class TestMapInterleave(unittest.TestCase):
    """
    Check that the interleaving of two strings works correctly:
    """
    
    def setUp(self):
        #string 1: consider string of length 4
        self.idx_len1 = 4
        self.idx_int1 = 10 #corresponds to binary string ['1', '0', '1', '0']
        self.idx_int1_reordered = 12#corresponds to reordered binary string ['1', '1', '0', '0']
        self.string1_reordered_sign = 1 # no sign change in the above reordering

        #string 2: consider string of length 8
        self.idx_len2 = 8
        self.idx_int2 = 170 #corresponds to binary string ['1', '0', '1', '0', '1', '0', '1', '0']
        self.idx_int2_reordered = 204#corresponds to reordered binary string ['1', '1', '0', '0', '1', '1', '0', '0']
        self.string2_reordered_sign = -1 # sign change in the above reordering

    def test_map_interleave(self):
         
        print("Test map_interleave")
        #test string 1:
        idx_int1_reordered, string1_reordered_sign = map_interleave(idx_int = self.idx_int1 , bin_length = self.idx_len1)
        self.assertEqual(idx_int1_reordered, self.idx_int1_reordered, msg="The reordering of the string 1 failed.")
        self.assertEqual(string1_reordered_sign, self.string1_reordered_sign, msg="The string associated with string 1 failed.")

        #test string 2:
        idx_int2_reordered, string2_reordered_sign = map_interleave(idx_int = self.idx_int2 , bin_length = self.idx_len2)
        self.assertEqual(idx_int2_reordered, self.idx_int2_reordered, msg="The reordering of the string 2 failed.")
        self.assertEqual(string2_reordered_sign, self.string2_reordered_sign, msg="The string associated with string 2 failed.")

        

class TestDictInterleave(unittest.TestCase):
     
    """
    Check that the dictionary of the interleaving of two strings works correctly
    """

    def setUp(self) -> None:
        
        #consider strings of length 8
        self.idx_len = 8

        #consider the dictionary for the interleaving of two strings
        self.map = dict_interleave(bin_length = self.idx_len)

    def test_dict_interleave(self):
        print("Test dict_interleave")
        #for a range of indices, check that the dictionary is correctly set up

        #Test 1: 
        #check original index idx_int = 12, corresponding to binary string ['0', '0', '0', '0', '1', '1', '0', '0']
        idx_int = 12
        #reordered string ['0', '1', '0', '1', '0', '0', '0', '0'] corresponds to idx_int_reordered = 80, no sign change
        idx_int_reordered = 80
        sign = 1
        self.assertEqual(self.map["idx"][idx_int], idx_int_reordered, msg="The reordering of string 1 failed.")
        self.assertEqual(self.map["sign"][idx_int], sign, msg="The sign associated with string 1 failed.")

        #Test 2:
        #check original index idx_int = 170, corresponding to binary string ['1', '0', '1', '0', '1', '0', '1', '0']
        idx_int = 170
        #reordered string ['1', '1', '0', '0', '1', '1', '0', '0'] corresponds to idx_int_reordered = 204, sign change
        idx_int_reordered = 204
        sign = -1
        self.assertEqual(self.map["idx"][idx_int], idx_int_reordered, msg="The reordering of string 2 failed.")
        self.assertEqual(self.map["sign"][idx_int], sign, msg="The sign associated with string 2 failed.")


class TestInterleave(unittest.TestCase):
     
    """
    Test that matrix from interleaving of two gates is correctly set up
    """

    def setUp(self) -> None:
        
        #define two gates for forward and backward branch, respectively
        self.gate_fw = np.random.rand(4,4)
        self.gate_bw = np.random.rand(4,4)

        #generate map that specifies the order of the rows and columns in the interleaved gate, as well as the signs
        self.map = dict_interleave(bin_length = self.gate_fw.shape[0])

        self.gate_interleaved = interleave(forward_gate = self.gate_fw, backward_gate = self.gate_bw, mapping = self.map)

    def interleaved_gate_manual(D_plus, D_minus):
        """
        Function that takes two gates, one from forward, one from backward branch, and spits out the interleaved gate (16x16)
        Parameters:
        D_plus: 4x4 np.array, gate from forward branch
        D_minus: 4x4 np.array, gate from backward branch

        Returns:
        gate: 16x16 np.array, interleaved gate
        """
        
        gate = np.zeros((16,16),dtype=np.complex_)
        #specify the order of the rows in the interleaved gate:
        rows = [0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15]

        #write products of the two gates into the interleaved gate
        for i in range (4):
            for j in range (4):
                gate[rows[i + 4 * j],0] = D_plus[j,0] * D_minus[i,0]
                gate[rows[i + 4 * j],1] = D_plus[j,0] * D_minus[i,1]
                gate[rows[i + 4 * j],2] = D_plus[j,1] * D_minus[i,0]
                gate[rows[i + 4 * j],3] = D_plus[j,1] * D_minus[i,1]
                gate[rows[i + 4 * j],4] = D_plus[j,0] * D_minus[i,2]
                gate[rows[i + 4 * j],5] = D_plus[j,0] * D_minus[i,3]
                gate[rows[i + 4 * j],6] = D_plus[j,1] * D_minus[i,2]
                gate[rows[i + 4 * j],7] = D_plus[j,1] * D_minus[i,3]
                gate[rows[i + 4 * j],8] = D_plus[j,2] * D_minus[i,0]
                gate[rows[i + 4 * j],9] = D_plus[j,2] * D_minus[i,1]
                gate[rows[i + 4 * j],10] = D_plus[j,3] * D_minus[i,0]
                gate[rows[i + 4 * j],11] = D_plus[j,3] * D_minus[i,1]
                gate[rows[i + 4 * j],12] = D_plus[j,2] * D_minus[i,2]
                gate[rows[i + 4 * j],13] = D_plus[j,2] * D_minus[i,3]
                gate[rows[i + 4 * j],14] = D_plus[j,3] * D_minus[i,2]
                gate[rows[i + 4 * j],15] = D_plus[j,3] * D_minus[i,3]

        #chage the sign for all variabels that ...
        sign_changes = np.identity(16)
        sign_changes[6,6] *= -1
        sign_changes[7,7] *= -1
        sign_changes[14,14] *= -1
        sign_changes[15,15] *= -1
        
        #apply the sign changes
        gate = sign_changes @ gate @ sign_changes

        return gate

    def test_interleave(self):
        
        print("Test interleave")
        #interleave manually and check that results coincide
        gate_interleaved_manual = TestInterleave.interleaved_gate_manual(self.gate_fw, self.gate_bw)

        self.assertTrue(np.allclose(self.gate_interleaved, gate_interleaved_manual), msg="The interleaved gate is not correctly set up.")


 
if __name__ == "__main__":
        unittest.main()