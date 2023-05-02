
# Needs fixing

import numpy as np
from helper_functions import *


n = 10# Hilbert Space Dimension
gamma = 1e-2# Damping Rate

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
T = 100

L = gamma**(0.5)*(q + np.sqrt(1/T)*(1j)*p)
Ldag = np.conjugate(L).T

def LinEm(x):
    hamiltonian_part = (-1j)* (np.dot(H, x) - np.dot(x, H))
    lindblad_part_1 = get_commutator(L, np.dot(x, Ldag))
    lindblad_part_2 = get_commutator(np.dot(L, x), Ldag)
    
    return hamiltonian_part + 0.5*(lindblad_part_1 + lindblad_part_2)    


if __name__ == "__main__":
    init = make_initial_density_matrix(n)
    t_i = 0
    t_f = 2000
    h = 0.01
    nsteps = int((t_f-t_i)/h)

    h = (t_f-t_i)/nsteps
    solRK = np.zeros((nsteps+1,n, n),dtype=complex)
    solRK[0]=init

    # Solving
    solRK = solver(solRK, LinEm, h)

    # Visualising
    plot_density_matrix_elements(solRK, title="CL with {} states temperature - {}K".format(n, T))
    #plot_trace_purity(solRK, title="QHO Thermal Bath with {}states".format(n))
    plot_diagonal_density_matrix_elements(solRK, title="CL with {} states temperature - {}K".format(n, T))
    #plot_offdiagonal_density_matrix_elements(solRK, title="CL with {} states temperature - {}K".format(n, T))

    plot_steady_state_td(solRK, title="Calderia Leggett with {}states".format(n))
    print("Steady state purity is {}".format(np.trace(np.dot(solRK[-1], solRK[-1]))))
