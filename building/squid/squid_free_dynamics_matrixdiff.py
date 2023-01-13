
# Convert to matrix diff form

""" Time Dependent solver for the SQUID model """

import numpy as np
from helper_functions import *

n = 40
adag = create_annihilation_operator(n) # Annihilation operator
a = create_creation_operator(n) # Creation operator

phi = 0.2 * (adag + a) # Flux operator (analogous to position operator)
Q = 0.5 * (1j)* (adag - a) # Momentum operator
cphi = 0.5*create_cos_phi(n, phi, 1, 1)

H = np.dot(Q,Q) + np.dot(phi, phi) + cphi
H = np.array(H)
init = make_initial_density_matrix(n)

# Setting simulation parameters
t_i = 0
t_f = 100
nsteps = 20000
h = (t_f-t_i)/nsteps
t = np.zeros((nsteps+1, n,n), dtype=complex)
t[0] = init

def LinEm():
    return -(1j) * (np.kron(np.identity(n), H) - np.kron(H.T, np.identity(n))) 

L = LinEm()

def handler(x):
    return np.dot(L, x.flatten("F")).reshape(n,n).T

if __name__ == '__main__':
    t = solver(t, handler, h)

    # Plotting
    plot_density_matrix_elements(t)
    #plot_trace_purity(t)