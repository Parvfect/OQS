
# Needs fixing

import numpy as np
from helper_functions import *


n = 14 # Hilbert Space Dimension
gamma = 0.05# Damping Rate

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
T = 1

L = q + (1/T)*(1j)*p
Ldag = np.conjugate(L).T

def LinEm(x):
    hamiltonian_part = (-1j)* (np.dot(H, x) - np.dot(x, H))
    lindblad_part_1 = get_commutator(L, np.dot(x, Ldag))
    lindblad_part_2 = get_commutator(np.dot(L, x), Ldag)
    
    return hamiltonian_part + 0.5*gamma*(lindblad_part_1 + lindblad_part_2)    

"""
def LinEm(x, L, Ldag):
    
    hamiltonian_part = (-1j)* (np.dot(H, x) - np.dot(x, H))
    lindblad_part_1 = get_commutator(L, np.dot(x, Ldag))
    lindblad_part_2 = get_commutator(np.dot(L, x), Ldag)

    return hamiltonian_part + 0.5*gamma*(lindblad_part_1 + lindblad_part_2)    

def temp_analysis():
    T = np.linspace(0, 500, 1)
    steady_state_purity = np.zeros(len(T))
    for i in range(len(T)):
        L = q + (1/T[i])*(1j)*p
        Ldag = np.conjugate(L).T
        init = make_initial_density_matrix(n)
        t_i = 0
        t_f = 1000
        nsteps = 10000
        solRK = np.zeros((nsteps+1,n, n),dtype=complex)
        solRK[0]=init

        # Solving
        solRK = solver(solRK, LinEm, h)
        steady_state_purity[i] = np.trace(np.dot(solRK[-1], solRK[-1]))
    plt.plot(T, steady_state_purity)
    plt.show()
"""

if __name__ == "__main__":
    init = make_initial_density_matrix(n)
    t_i = 0
    t_f = 800
    nsteps = 10000

    h = (t_f-t_i)/nsteps
    solRK = np.zeros((nsteps+1,n, n),dtype=complex)
    solRK[0]=init

    # Solving
    solRK = solver(solRK, LinEm, h)

    # Visualising
    plot_density_matrix_elements(solRK, title="CL with {} states temperature - {}K".format(n, T))
    #plot_trace_purity(solRK, title="QHO Thermal Bath with {}states".format(n))

    plot_steady_state_td(solRK, title="Calderia Leggett with {}states".format(n))
    print("Steady state purity is {}".format(np.trace(np.dot(solRK[-1], solRK[-1]))))
