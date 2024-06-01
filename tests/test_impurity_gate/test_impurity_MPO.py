import unittest
import numpy as np
import os,sys
parent_dir = os.path.join(os.path.dirname(__file__),"../..")
#append parent directory to path
sys.path.append(parent_dir)
from src.real_time.compute_impurity_gate.interleave_gate import interleave, dict_interleave
from src.real_time.single_orbital.generate_MPO import impurity_MPO, adjust_string, operator_at_Keldysh_idx, gate, gate_initial
from src.shared_modules.Keldysh_contour import position_to_Keldysh_idx
from src.shared_modules.many_body_operator import annihilation_ops
from src.shared_modules.dual_kernel import operator_to_kernel
from scipy.linalg import expm

class TestAdjustString(unittest.TestCase):
        
        def setUp(self) -> None:
            #generate array with random booleans
            self.nbr_time_steps = 10
            self.string = np.random.choice([True,False], size=2*self.nbr_time_steps+2)

        def test_adjust_string(self):
            print("Test adjust_string")
            #compute a Keldysh-index from a time-point and branch
            Keldysh_idx = position_to_Keldysh_idx(7, 'f', nbr_time_steps=self.nbr_time_steps)
            print("Test adjust_string")
            string_adjusted = adjust_string(string = self.string, Keldysh_idx=Keldysh_idx, parity=-1)

            #swap back the entries and see if the string self.string is recovered
            for time in range(7):
                #forward branch
                Keldysh_idx = position_to_Keldysh_idx(time, 'f', nbr_time_steps=self.nbr_time_steps)
                string_adjusted[Keldysh_idx] = not string_adjusted[Keldysh_idx]
                #backward branch
                Keldysh_idx = position_to_Keldysh_idx(time, 'b', nbr_time_steps=self.nbr_time_steps)
                string_adjusted[Keldysh_idx] = not string_adjusted[Keldysh_idx]

            self.assertTrue(np.array_equal(self.string, string_adjusted), msg="The string is not as expected.")
            
   

class TestOperatorAtKeldyshIndex(unittest.TestCase):
      
    def setUp(self) -> None:
        
        self.U_evol = np.random.rand(4,4) + 1.j * np.random.rand(4,4)
        self.nbr_time_steps = 10
        self.operator_a = np.random.rand(4,4)
        self.Keldysh_idx_a = position_to_Keldysh_idx(5, 'f', nbr_time_steps=self.nbr_time_steps)
        self.operator_b = np.random.rand(4,4)
        self.Keldysh_idx_b = position_to_Keldysh_idx(7, 'b', nbr_time_steps=self.nbr_time_steps)

    def test_operator_at_Keldysh_idx(self):
         
        print("Test operator_at_Keldysh_idx")

        #compute the operator at a few indices and compare to manual result:
        operator_test = lambda test_idx : operator_at_Keldysh_idx(Keldysh_idx = test_idx, U_evol = self.U_evol, operator_a=self.operator_a, Keldysh_idx_a=self.Keldysh_idx_a, operator_b=self.operator_b, Keldysh_idx_b=self.Keldysh_idx_b, nbr_time_steps=self.nbr_time_steps)
        
        #check 1:
        test_idx = position_to_Keldysh_idx(3, 'f', nbr_time_steps=self.nbr_time_steps)
        operator_out = operator_test(test_idx)
        #compute the expected result
        operator_expected = self.U_evol
        #compare the results
        self.assertTrue(np.allclose(operator_out, operator_expected), msg="The operator is not as expected.")

        #check 2:
        test_idx = position_to_Keldysh_idx(3, 'b', nbr_time_steps=self.nbr_time_steps)
        operator_out = operator_test(test_idx)
        #compute the expected result
        operator_expected = self.U_evol.T.conj()
        #compare the results
        self.assertTrue(np.allclose(operator_out, operator_expected), msg="The operator is not as expected.")

        #check 3:
        test_idx = position_to_Keldysh_idx(5, 'f', nbr_time_steps=self.nbr_time_steps)
        operator_out = operator_test(test_idx)
        #compute the expected result
        operator_expected = self.operator_a
        #compare the results
        self.assertTrue(np.allclose(operator_out, operator_expected), msg="The operator is not as expected.")

        #check 4:
        test_idx = position_to_Keldysh_idx(7, 'b', nbr_time_steps=self.nbr_time_steps)
        operator_out = operator_test(test_idx)
        #compute the expected result
        operator_expected = self.operator_b
        #compare the results
        self.assertTrue(np.allclose(operator_out, operator_expected), msg="The operator is not as expected.")


        #check 5: Insert to operators at the same position
        #compute the operator at a few indices and compare to manual result:
        Keldysh_idx = 13
        operator_test = lambda test_idx : operator_at_Keldysh_idx(Keldysh_idx = test_idx, U_evol = self.U_evol, operator_a=self.operator_a, Keldysh_idx_a=Keldysh_idx, operator_b=self.operator_b, Keldysh_idx_b=Keldysh_idx, nbr_time_steps=self.nbr_time_steps)

        test_idx = Keldysh_idx
        operator_out = operator_test(test_idx)
        #compute the expected result
        operator_expected = self.operator_b @ self.operator_a 
        #compare the results
        self.assertTrue(np.allclose(operator_out, operator_expected), msg="The operator is not as expected.")



class TestImpurityMPO(unittest.TestCase):
     
    def setUp(self) -> None:
              
        self.nbr_time_steps = 5

        self.U_evol = np.random.rand(4,4) + 1.j * np.random.rand(4,4)

        #initial density matrix
        initial_density_matrix = np.random.rand(4,4) 
        #make hermitian
        self.initial_density_matrix = initial_density_matrix + initial_density_matrix.T.conj()


        #operators
        
        c = annihilation_ops(n_ferms=2)
        self.c_down = c[0] #annihilation operator for spin down
        self.c_up = c[1] #annihilation operator for spin up

        #Keldysh indices of operators
        self.Keldysh_idx_a = position_to_Keldysh_idx(3, 'f', self.nbr_time_steps)
        self.Keldysh_idx_b = position_to_Keldysh_idx(1, 'b', self.nbr_time_steps)

        self.operator_a = self.c_down
        self.operator_b = self.c_up.T

    def test_impurity_MPO(self):
        print("Test impurity_MPO")

        #compute the MPO
        MPO = impurity_MPO(self.U_evol, self.initial_density_matrix, self.nbr_time_steps,self.operator_a, self.Keldysh_idx_a, self.operator_b, self.Keldysh_idx_b)
        MPO_boundary_condition = MPO["boundary_condition"]
        MPO_init_state = MPO["init_state"]
        MPO_gates = MPO["gates"]
        global_sign = MPO["global_sign"]

        #check the size of the MPO
        self.assertTrue(MPO_boundary_condition.shape == (4,4), msg="The boundary condition MPO has the wrong shape.")
        self.assertTrue(MPO_init_state.shape == (4,4), msg="The initial state MPO has the wrong shape.")
        self.assertTrue(np.all(MPO_gates[0].shape == (4**2,4**2)), msg="The gate MPO has the wrong shape.")
        

        #compute the expected MPO by hand and compare
        #compute gates by hand as benchmark:
        #string must be included in the gates 2 (f + b), 1(b)
        MPO_bulk1 = interleave(operator_to_kernel(self.U_evol), operator_to_kernel(self.c_up.T, branch = 'b', string=True), mapping=dict_interleave(bin_length = 4))
        MPO_bulk2 = interleave(operator_to_kernel(self.U_evol, string=True), operator_to_kernel(self.U_evol.T.conj(), branch = 'b', string=True), mapping=dict_interleave(bin_length = 4))
        MPO_bulk3 = interleave(operator_to_kernel(self.c_down), operator_to_kernel(self.U_evol.T.conj(), branch = 'b'), mapping=dict_interleave(bin_length = 4))

        #compare the first three gates
        self.assertTrue(np.allclose(MPO_gates[0], MPO_bulk1), msg="The first gate is not as expected.")
        self.assertTrue(np.allclose(MPO_gates[1], MPO_bulk2), msg="The second gate is not as expected.")
        self.assertTrue(np.allclose(MPO_gates[2], MPO_bulk3), msg="The third gate is not as expected.")

        #check the boundary condition
        boundary_expected = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
        self.assertTrue(np.allclose(MPO_boundary_condition, boundary_expected), msg="The boundary condition is not as expected.")

        #check the initial state
        initial_state_expected = operator_to_kernel(self.initial_density_matrix, branch='b')
        self.assertTrue(np.allclose(MPO_init_state, initial_state_expected), msg="The initial state is not as expected.")

        #check the global sign (for these operator insertions, the global sign must be 1)
        self.assertTrue(global_sign == 1, msg="The global sign is not as expected.")


        #check for insertions at the same position 

        #insert both in initial state:
        #Keldysh indices of operators
        Keldysh_idx_a = position_to_Keldysh_idx(0, 'f', self.nbr_time_steps)
        Keldysh_idx_b = position_to_Keldysh_idx(0, 'b', self.nbr_time_steps)
        MPO = impurity_MPO(self.U_evol, self.initial_density_matrix, self.nbr_time_steps, self.operator_a, Keldysh_idx_a, self.operator_b, Keldysh_idx_b)
        MPO_boundary_condition = MPO["boundary_condition"]
        MPO_init_state = MPO["init_state"]
        #check the initial state
        initial_state_expected = operator_to_kernel(self.operator_a @ self.initial_density_matrix @ self.operator_b, branch='b')
        self.assertTrue(np.allclose(MPO_init_state, initial_state_expected), msg="The initial state is not as expected.")

        #insert both in the first gate
        #Keldysh indices of operators
        # on different branches
        Keldysh_idx_a = position_to_Keldysh_idx(1, 'f', self.nbr_time_steps)
        Keldysh_idx_b = position_to_Keldysh_idx(1, 'b', self.nbr_time_steps)
        MPO = impurity_MPO(self.U_evol, self.initial_density_matrix, self.nbr_time_steps, self.operator_a, Keldysh_idx_a, self.operator_b, Keldysh_idx_b)
        MPO_boundary_condition = MPO["boundary_condition"]
        MPO_init_state = MPO["init_state"]
        MPO_gates = MPO["gates"]
        #string must be included in 1(f) 
        MPO_bulk1 = interleave(operator_to_kernel(self.operator_a, string=True), operator_to_kernel(self.operator_b, branch = 'b'), mapping=dict_interleave(bin_length = 4))
        self.assertTrue(np.allclose(MPO_gates[0], MPO_bulk1), msg="The first gate is not as expected for equal insertion on different branches.")

        # on same branch
        Keldysh_idx_a = position_to_Keldysh_idx(1, 'b', self.nbr_time_steps)
        Keldysh_idx_b = position_to_Keldysh_idx(1, 'b', self.nbr_time_steps)
        MPO = impurity_MPO(self.U_evol, self.initial_density_matrix, self.nbr_time_steps, self.operator_a, Keldysh_idx_a, self.operator_b, Keldysh_idx_b)
        MPO_boundary_condition = MPO["boundary_condition"]
        MPO_init_state = MPO["init_state"]
        MPO_gates = MPO["gates"]
        #string must be included in 1(f) 
        MPO_bulk1 = interleave(operator_to_kernel(self.U_evol), operator_to_kernel(self.operator_b @ self.operator_a, branch = 'b'), mapping=dict_interleave(bin_length = 4))
        self.assertTrue(np.allclose(MPO_gates[0], MPO_bulk1), msg="The first gate is not as expected for equal insertion on same branch.")

        #insert both in boundary condition
        #Keldysh indices of operators
        Keldysh_idx_a = position_to_Keldysh_idx(self.nbr_time_steps, 'f', self.nbr_time_steps)
        Keldysh_idx_b = position_to_Keldysh_idx(self.nbr_time_steps, 'b', self.nbr_time_steps)
        MPO = impurity_MPO(self.U_evol, self.initial_density_matrix, self.nbr_time_steps, self.operator_a, Keldysh_idx_a, self.operator_b, Keldysh_idx_b)
        MPO_boundary_condition = MPO["boundary_condition"]
        MPO_init_state = MPO["init_state"]
        MPO_gates = MPO["gates"]
        #boundary condition by hand:
        boundary_expected = operator_to_kernel(self.operator_b @ self.operator_a, boundary=True)
        self.assertTrue(np.allclose(MPO_boundary_condition, boundary_expected), msg="The boundary condition is not as expected for operators inserted.")





class TestGate(unittest.TestCase):
     
          
    def setUp(self) -> None:
              
        self.nbr_time_steps = 5

        self.delta_t = 0.1

        #operators
        
        c = annihilation_ops(n_ferms=2)
        self.c_down = c[0] #annihilation operator for spin down
        self.c_up = c[1] #annihilation operator for spin up

        #set up many-body Hamiltonian with arbitrarily chosen parameters, including also spin-hopping, interaction and superconudcting terms
        self.Ham = 0.3 * self.c_up.T @ self.c_up + 0.1 * self.c_down.T @ self.c_down + 0.23 * (self.c_up.T @ self.c_up) @ (self.c_down.T @ self.c_down) + 0.23 * self.c_up.T @ self.c_down.T + 0.86 * self.c_down.T @ self.c_up + 0.1 * self.c_up.T @ self.c_down + 0.46 * self.c_down.T @ self.c_up
        #make hermitian
        self.Ham = self.Ham + self.Ham.T.conj()
        #evolution operator
        self.U_evol = expm(-1.j * self.delta_t * self.Ham)

    def test_gate(self):
        print("Test impurity_MPO")

        #generate MPO_gates 
        string = False
        gate_0, string = gate(Ham=self.Ham, delta_t=self.delta_t, op_bw = self.c_up.T, string_in=string)
        gate_1, string = gate(Ham=self.Ham, delta_t=self.delta_t, string_in =string)
        gate_2, string = gate(Ham=self.Ham, delta_t=self.delta_t, op_fw = self.c_down, string_in=string)
        MPO_gates = [gate_0, gate_1, gate_2]

        #compute the expected MPO by hand and compare
        #compute gates by hand as benchmark:
        #string must be included in the gates 2 (f + b), 1(b)
        MPO_bulk1 = interleave(operator_to_kernel(self.U_evol), operator_to_kernel(self.c_up.T @ self.U_evol.T.conj(), branch = 'b', string=True), mapping=dict_interleave(bin_length = 4))
        MPO_bulk2 = interleave(operator_to_kernel(self.U_evol, string=True), operator_to_kernel(self.U_evol.T.conj(), branch = 'b', string=True), mapping=dict_interleave(bin_length = 4))
        MPO_bulk3 = interleave(operator_to_kernel(self.c_down @ self.U_evol), operator_to_kernel(self.U_evol.T.conj(), branch = 'b'), mapping=dict_interleave(bin_length = 4))

        #compare the first three gates
        self.assertTrue(np.allclose(MPO_gates[0], MPO_bulk1), msg="The first gate is not as expected.")
        self.assertTrue(np.allclose(MPO_gates[1], MPO_bulk2), msg="The second gate is not as expected.")
        self.assertTrue(np.allclose(MPO_gates[2], MPO_bulk3), msg="The third gate is not as expected.")


    def test_gate_boundary_condition(self):
        print("Test boundary condition of impurity_MPO")
        
        #check the boundary condition
        boundary_expected = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
        boundary_computed, _ = gate(Ham=self.Ham, delta_t=self.delta_t, string_in=False, boundary = True)#second argument is outgoing string which is always "False"
        self.assertTrue(np.allclose(boundary_computed, boundary_expected), msg=f"The boundary condition is not as expected. Result: \n {boundary_computed} \n Expected: \n {boundary_expected}")

        #now insert odd operator. This should throw an error if the ingoing string is not set to True
        self.assertRaises(AssertionError, gate, Ham=self.Ham, delta_t=self.delta_t, op_fw = self.c_down, string_in=False, boundary = True)
        
        #now, set ingoing string to True. This should throw an error if the gate is even
        self.assertRaises(AssertionError, gate, Ham=self.Ham, delta_t=self.delta_t, string_in=True, boundary = True)

        #now, set the gate to odd and string_in = True. This should pass.
        _, _ = gate(Ham=self.Ham, delta_t=self.delta_t, op_fw = self.c_down, string_in=True, boundary = True)

    

class TestInitialState(unittest.TestCase):

    def setUp(self):
        self.c = annihilation_ops(n_ferms=2)
        
        #define density matrix
        density_matrix = 0.1 * self.c[0].T @ self.c[0] + 0.3 * self.c[1].T @ self.c[0]
        #make hermitian:
        self.density_matrix = density_matrix + density_matrix.T.conj()



    def test_initial_state(self):
        print("Test initial state of impurity_MPO")
        
        #check the initial state
        initial_state_even = operator_to_kernel(self.density_matrix, branch='b')
        initial_state_odd = operator_to_kernel(self.density_matrix @ self.c[0], branch='b', string = True) #if initial state is odd, the outgoing string muste be 'True'
        dual_density_matrix, string_out = gate_initial(self.density_matrix)
        dual_density_matrix_odd, string_out_odd = gate_initial(self.density_matrix @ self.c[0])

        #compare the results
        self.assertTrue(np.allclose(dual_density_matrix, initial_state_even), msg="The initial state is not as expected.")
        self.assertTrue(np.allclose(dual_density_matrix_odd, initial_state_odd), msg="The initial state is not as expected.")

        #check the string
        self.assertTrue(string_out == False, msg="The string is not as expected.")
        self.assertTrue(string_out_odd == True, msg="The string is not as expected.")


     



if __name__ == "__main__":
        unittest.main()