import numpy as np

#_________Define fermionic cration and annihilation operators in the many-body basis
# Note: The convention for the basis of a many-body operator A is: A = v^\dagger @ a @ v, with: v = [<0|, <0|c_up , <0|c_down, <0|c_up c_down].

c_up_dag = np.zeros((4,4),dtype=np.complex_) 
c_up_dag[1,0] = 1
c_up_dag[3,2] = -1

c_down_dag = np.zeros((4,4),dtype=np.complex_) 
c_down_dag[2,0] = 1
c_down_dag[3,1] = 1

c_up = np.zeros((4,4),dtype=np.complex_) 
c_up[0,1] = 1
c_up[2,3] = -1

c_down = np.zeros((4,4),dtype=np.complex_) 
c_down[0,2] = 1
c_down[1,3] = 1