

# Needs fixing

import numpy as np
from helper_functions import *


n = 14 # Hilbert Space Dimension
gamma = 0.3 # Damping Rate

# Annihilation and Creation Operators
adag = create_annihilation_operator(n)
a = create_creation_operator(n)

# Hamiltonian
q = (adag + a)/2
p = 1j*(adag - a)/2
H = (np.dot(p,p) + np.dot(q,q)) + gamma/2 * get_anti_commutator(q,p)
H = np.array(H)

# Initial Density Matrix
w = 2e13
hbar = 1e-34
kb = 1.38e-23
T = 300

L = q + (1j)*p
Ldag = np.conjugate(L).T

def LinEm(x):
    hamiltonian_part = (-1j)* (np.dot(H, x) - np.dot(x, H))
    second_kernel = - 1j * gamma * get_commutator(q, get_anti_commutator(p, x))
    third_kernel = - gamma * T * get_commutator(q, get_commutator(q,x))

    return hamiltonian_part + second_kernel# + third_kernel  

if __name__ == "__main__":
    init = make_initial_density_matrix(n)
    t_i = 0
    t_f = 1000
    nsteps = 10000

    h = (t_f-t_i)/nsteps
    solRK = np.zeros((nsteps+1,n, n),dtype=complex)
    solRK[0]=init

    # Solving
    solRK = solver(solRK, LinEm, h)

    # Visualising
    plot_density_matrix_elements(solRK, title="QHO Thermal Bath with {}states".format(n))
    #plot_trace_purity(solRK, title="QHO Thermal Bath with {}states".format(n))

    plot_steady_state_td(solRK, title="Calderia Leggett with {}states".format(n))
