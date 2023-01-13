

""" Time Dependent solver for the SQUID model """

import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *

n = 3
l=3e-10
C=5e-15
je = 9.9e-22
hbar = 1e-34
m = 9.1e-31
w = 8.16e11


adag = create_annihilation_operator(n) # Annihilation operator
a = create_creation_operator(n) # Creation operator

phi = (np.sqrt((hbar)/(2*m*w))*((adag + a))*1e-13)/(2*l) # Flux operator (analogous to position operator)
Q = (np.sqrt((hbar*m*w)/(2)) * (1j)* (adag - a))/(2*C) # Momentum operator
cphi = create_cos_phi(n, phi, 1, 1)
print(cphi)
#print(phi)
#print(Q)

H = np.dot(Q,Q) + np.dot(phi, phi) + je*cphi
H = H* 1e24
print(H)
H = np.array(H)
init = make_initial_density_matrix(n)


def handler(x):
    return (-1j)* (np.dot(H, x) - np.dot(x, H))  


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