import unittest
import numpy as np
import os,sys
parent_dir = os.path.join(os.path.dirname(__file__),"../..")
#append parent directory to path
sys.path.append(parent_dir)
from src.shared_modules.Keldysh_contour import position_to_Keldysh_idx, Keldysh_idx_to_position


class TestPositionToKeldyshIdx(unittest.TestCase):

    def setUp(self) -> None:
        
        self.nbr_time_steps = 10
        self.indices_expected = np.arange(2*self.nbr_time_steps+1)

    def test_position_to_Keldysh_idx(self):

        print("Test position_to_Keldysh_idx")
        
        indices = []
        #forward branch
        for time_point in range(11):
            indices.append(position_to_Keldysh_idx(time_point, 'f', self.nbr_time_steps))
    
        #backward branch
        for time_point in range(10,0,-1):
            indices.append(position_to_Keldysh_idx(time_point, 'b', self.nbr_time_steps))

        self.assertTrue(np.allclose(indices, self.indices_expected), msg="The Keldysh indices are not as expected.")
    

    def test_Keldysh_idx_to_position(self):
         
        print("Test Keldysh_idx_to_position")

        time_points = []
        branches = []
        for Keldysh_idx in range(2*self.nbr_time_steps+2):
            time_point, branch = Keldysh_idx_to_position(Keldysh_idx, self.nbr_time_steps)
            time_points.append(time_point)
            branches.append(branch)

        self.assertTrue(np.allclose(time_points[:len(time_points)//2], np.arange(self.nbr_time_steps+1)), msg=f"The first half of time points are not as expected: {time_points[:len(time_points)//2]}")
        self.assertTrue(np.allclose(time_points[len(time_points)//2:], np.arange(self.nbr_time_steps,-1,-1)), msg=f"The second half of time points are not as expected: {time_points[len(time_points)//2:]}")

        #check that the branches are as expected
        self.assertTrue(np.array_equal(branches[:len(branches)//2], ['f']*(self.nbr_time_steps+1)), msg=f"The first half of branches are not as expected: {branches[:len(branches)//2]}")
        self.assertTrue(np.array_equal(branches[len(branches)//2:], ['b']*(self.nbr_time_steps+1)), msg=f"The second half of branches are not as expected:{branches[len(branches)//2:]}")

if __name__ == "__main__":
        unittest.main()
