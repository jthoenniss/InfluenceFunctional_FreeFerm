import numpy as np
from scipy import linalg

#____________Define non-interacting Green's functions inside the environment_____________________________________________
def g_lesser(omega, beta, tau,tau_p):
    return np.real(np.exp(-omega * (tau - tau_p)) * 1. / (1.+ np.exp(beta * omega)))
def g_greater(omega, beta, tau,tau_p):
    return - np.real(np.exp(-omega * (tau -tau_p)) * 1. / (1.+ np.exp(-beta * omega)))
def spec_dens(gamma,energy):
    e_c = 10.*gamma 
    nu = 10./gamma
    #return  2 * gamma /((1+np.exp(nu*(energy - e_c))) * (1+np.exp(-nu*(energy + e_c)))) #this gives a flat band with smooth edges
    return gamma  #this gives a flat band with sharp edges

