
import numpy as np
from helper_functions import *

n = 14 # Hilbert Space Dimension
gamma = 0.05 # Damping Rate

# Annihilation and Creation Operators
adag = create_annihilation_operator(n)
a = create_creation_operator(n)

# Hamiltonian
q = (adag + a)/2
p = 1j*(adag - a)/2
H = (np.dot(p,p) + np.dot(q,q)) + gamma/2 * get_anti_commutator(q,p)

# Initial Density Matrix
w = 2e13
hbar = 1e-34
kb = 1.38e-23
T = 1

def LinEm(x):
    hamiltonian_part = (-1j)* get_commutator(H, x)
    second_kernel = - 1j * gamma * get_commutator(q, get_anti_commutator(p, x))
    third_kernel = - gamma * T * get_commutator(q, get_commutator(q,x))
    added_term = - gamma /T * get_commutator(p, get_commutator(p,x))
    what = -1j*gamma* get_commutator(p, get_anti_commutator(q,x))

    #return hamiltonian_part + 0.01*second_kernel + 0.01*third_kernel + 10*added_term
    return hamiltonian_part + 2 *added_term

if __name__ == "__main__":
    init = make_initial_density_matrix(n)
    t_i = 0
    t_f = 200
    h = 0.01
    nsteps = int((t_f-t_i)/h)
    solRK = np.zeros((nsteps+1,n, n),dtype=complex)
    solRK[0]=init

    # Solving
    solRK = solver(solRK, LinEm, h)

    # Visualising
    plot_density_matrix_elements(solRK)
    #plot_trace_purity(solRK)
    #plot_diagonal_density_matrix_elements(solRK)
    #plot_offdiagonal_density_matrix_elements(solRK)
    #plot_steady_state_td_2d(solRK)
    #plot_steady_state_td_3d(solRK)

   # plot_steady_state_td(solRK, title="Calderia Leggett with {}states".format(n))
    