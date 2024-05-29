#Module to print the comparison of the Grassmann and many-body propagators and density matrices to the terminal.

import numpy as np


def print_propagators(GM_upup, GM_downdown, MB_upup, MB_downdown):
    num_time_steps = len(GM_upup)
    print(f"{'':>10} "
          f"{'G_upup(t)':^60} "
          f"{'G_downdown(t)':^57}  "
          f"{'Coincidence':^30}")
    print(f"{'Time Step':>10} | "
          f"{'Grassmann':^25} <-> {'many-body':^25} | "
          f"{'Grassmann':^25} <-> {'many-body':^25} || "
          f"{'up-up':^15} | {'down-down':^15}")
    print("-" * 165)  # Adjust the line length based on the new width for better alignment

    for t in range(num_time_steps):
        GM_upup_fmt = format_complex(GM_upup[t])
        MB_upup_fmt = format_complex(MB_upup[t])
        GM_downdown_fmt = format_complex(GM_downdown[t])
        MB_downdown_fmt = format_complex(MB_downdown[t])

        coincidence_upup = check_coincidence(GM_upup[t], MB_upup[t])
        coincidence_downdown = check_coincidence(GM_downdown[t], MB_downdown[t])

        print(f"{t:>10} | {GM_upup_fmt:^25} <-> {MB_upup_fmt:^25} | "
              f"{GM_downdown_fmt:^25} <-> {MB_downdown_fmt:^25} || "
              f"{coincidence_upup:^15} | {coincidence_downdown:^15}")

    print("\n")

def format_complex(c):
    """Ensures complex numbers are formatted uniformly."""
    return f"({c.real:.6f}{'+' if c.imag >= 0 else '-'}{abs(c.imag):.6f}j)"

def check_coincidence(a, b):
    """Checks for equivalence of two complex numbers, returns 'Match' or 'Differ'."""
    return "Match" if np.allclose(a, b, atol=1e-6) else "Differ"


def format_number(n):
    """ Format a number or complex number to ensure clear presentation. """
    if np.isclose(n, 0, atol=1e-6):
        return " 0.         "  # Uniform zero presentation
    elif np.isclose(n.imag, 0, atol=1e-6):  # If imaginary part is zero
        return f"{n.real:+11.6f} "  # Format real part only
    else:
        return f"{n.real:+11.6f}{n.imag:+11.6f}j".replace('+-', '-')  # Format complex numbers

def print_matrices(GM_matrices, MB_matrices):
    t = 0
    for GM_DM, MB_DM in zip(GM_matrices, MB_matrices):

        #normalize both density matrices by their trace
        GM_DM /= np.trace(GM_DM)
        MB_DM /= np.trace(MB_DM)


        if np.allclose(GM_DM, MB_DM):
            print(f"Time Step {t} -- Coincidence: Match_________________________\n")
        else:
            print(f"Time Step {t} -- Coincidence: DIFFER________________________\n")
    
        #print them one below the other. Arrange their real and imaginary parts in an easily readable way
        print("Density Matrix from Grassmann path integral (real part):")
        print(np.real(GM_DM))
        print("Density Matrix from many-body overlap (real part):")
        print(np.real(MB_DM))
        print("\n")
        t += 1