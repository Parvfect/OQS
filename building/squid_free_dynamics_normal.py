

""" Time Dependent solver for the SQUID model """

import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *

n = 3
l=1
C=1
je = 0
hbar = 1

adag = create_annihilation_operator(n) # Annihilation operator
a = create_creation_operator(n) # Creation operator

phi = (0.2*(adag + a))/(l) # Flux operator (analogous to position operator)
Q = 0.5 * (1j)* (adag - a)/(C) # Momentum operator
cphi = je*create_cos_phi(n, phi, 1, 1)

H = np.dot(Q,Q) + np.dot(phi, phi) + cphi
init = make_initial_density_matrix(n)

def handler(x):
    return (-1j/hbar)* (np.dot(H, x) - np.dot(x, H))  


if __name__ == "__main__":
    # Setting simulation parameters
    t_i = 0
    t_f = 100
    nsteps = 20000
    h = (t_f-t_i)/nsteps
    t = np.zeros((nsteps+1, n,n), dtype=complex)
    t[0] = init

    t = solver(t, handler, h)

    # Plotting
    plot_density_matrix_elements(t)
    #plot_trace_purity(t)