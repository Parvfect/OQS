
# Needs fixing
# Add cosphi and effective hammy term and we have SQUID model
# And then we can parameterize

import numpy as np
from helper_functions import *


n = 14 # Hilbert Space Dimension
gamma = 0.05 # Damping Rate

# Annihilation and Creation Operators
Ldag = create_annihilation_operator(n)
a = create_creation_operator(n)

# Hamiltonian
q = (Ldag + a)/2
p = 1j*(Ldag - a)/2
cphi = create_cos_phi(q, 1, 0.2, 1)
H = (np.dot(p,p) + np.dot(q,q) - cphi) + gamma/2 * get_anti_commutator(q,p)
H = np.array(H)

# Initial Density Matrix
rho_0 = [[0.5,0.5],[0.5,0.5]]
w = 2e13
hbar = 1e-34
kb = 1.38e-23
T = 300

#print(1/(np.exp((hbar*w)/(kb*T))-1))
nth = 1/(np.exp((hbar*w)/(kb*T))-1) #2
sz = np.array([[1,0],[0,-1]])
wo = 10
epsilon = 1
L = gamma**(0.5) * (q - (1j - epsilon/2) * p)
Ldag = np.conjugate(L).T
#H = (wo/2) * sz 

# Encoding Equation


def LinEm(x):
    hamiltonian_part = (-1j)* (np.dot(H, x) - np.dot(x, H))
    lindblad_part_1 = get_commutator(L, np.dot(x, Ldag))
    lindblad_part_2 = get_commutator(np.dot(L, x), Ldag)
    #lindblad_part_3 = get_commutator(Ldag, np.dot(x, L))
    #lindblad_part_4 = get_commutator(np.dot(Ldag, x), L)
    
    return hamiltonian_part + 0.5*gamma*(lindblad_part_1 + lindblad_part_2)    

if __name__ == "__main__":
    init = make_initial_density_matrix(n)
    t_i = 0
    t_f = 2000
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
