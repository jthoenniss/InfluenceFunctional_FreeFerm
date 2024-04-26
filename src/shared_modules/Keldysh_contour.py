#convert indices on the Keldysh contour

#import Tuple from typing
from typing import Tuple

def position_to_Keldysh_idx(time_point: int, branch: str, nbr_time_steps: int) -> int:
    """
    Converts a given time point along with the branch index to a Keldysh index which is defined along the unfolded Keldysh contour.
    Parameters:
    - time_point (int): The time point on the Keldysh contour. Index 0 refers to the initial state, index 1 to the first impurity gate, etc.
    - branch (str): The branch index. 'f' refers to the forward branch and 'b' to the backward branch.
    - nbr_time_steps (int): The total number of time steps in the influence functional 

    Returns:
    - int: The Keldysh index corresponding to the given time point and branch index. 
        The indices are ordered as follows: 0 : initial state, 1 : (first fw-gate), ..., nbr_time_steps: (last fw-gate,i.e. final time), nbr_time_steps+1: (last bw-gate, i.e. final time), ..., 2*nbr_time_steps = (first bw-gate.
    """

    assert isinstance(time_point, int) and time_point >= 0, "The time point must be a non-negative integer."
    assert time_point <= nbr_time_steps, "The time point must be less than or equal to the total number of time steps."
    assert branch in ['f', 'b'], "The branch index must be either 'f' for the forward branch or 'b' for the backward branch."
    assert isinstance(nbr_time_steps, int) and nbr_time_steps > 0, "The number of time steps must be a positive integer."

    if branch == 'f':
        return time_point
    else:
        return 2*nbr_time_steps - time_point + 1
    

def Keldysh_idx_to_position(Keldysh_idx: int, nbr_time_steps: int) -> Tuple[int, str]:
    """
    Inverse of the function position_to_Keldysh_idx. Converts a given Keldysh index to a time point and branch index.
    Parameters:
    - Keldysh_idx (int): The Keldysh index corresponding to a time point and branch index.
    - nbr_time_steps (int): The total number of time steps in the influence functional.

    Returns:
    - Tuple[int, str]: A tuple containing the time point and branch index corresponding to the given Keldysh index.
    """

    assert isinstance(Keldysh_idx, int) and Keldysh_idx >= 0, "The Keldysh index must be a non-negative integer."
    assert Keldysh_idx < 2*nbr_time_steps + 2, "The Keldysh index must be less than or equal to twice the total number of time steps + 1."
    assert isinstance(nbr_time_steps, int) and nbr_time_steps > 0, "The number of time steps must be a positive integer."

    if Keldysh_idx <= nbr_time_steps:
        return Keldysh_idx, 'f'
    else:
        return 2*nbr_time_steps - Keldysh_idx + 1, 'b'
    

if __name__ == '__main__':

    nbr_time_steps = 10

    print("forward branch")
    for time_point in range(11):
        #forward branch
        print(f"Time point: {time_point}, Keldysh index: {position_to_Keldysh_idx(time_point, 'f', nbr_time_steps)}")
    
    print("backward branch")
    for time_point in range(10,-1,-1):
        #backward branch
        print(f"Time point: {time_point}, Keldysh index: {position_to_Keldysh_idx(time_point, 'b', nbr_time_steps)}")

    #test inverse function
    for Keldysh_idx in range(2*nbr_time_steps+2):
        time_point, branch = Keldysh_idx_to_position(Keldysh_idx, nbr_time_steps)
        print(f"Keldysh index: {Keldysh_idx}, Time point: {time_point}, Branch: {branch}")