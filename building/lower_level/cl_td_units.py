

from helper_functions import *
import numpy as np


hbar = 1.0545718e-34
kb = 1.38064852e-23
w = 2e13
T = 300
m = 1e-31
gamma = 0.05
n = 2

# Annihilation and Creation Operators
adag = create_annihilation_operator(n)
a = create_creation_operator(n)

# Position and Momentum Operators
q = np.sqrt(hbar/(2*m*w))*(adag + a)/2
p = np.sqrt(hbar*m*w/2)*1j*(adag - a)/2

# Hamiltonian\
#print(np.dot(p,p)/(2*m))
#print(0.5*m*w*w*np.dot(q,q))
H = np.dot(p,p)/(2*m) + 0.5*m*w*w*np.dot(q,q) + gamma/2
#print(H)

def LinEm(x):
    hamiltonian_part = (-1j/hbar)* (np.dot(H, x) - np.dot(x, H))
    #print(x)
    #lindblad_part_1 = get_commutator(L, np.dot(x, Ldag))
    #lindblad_part_2 = get_commutator(np.dot(L, x), Ldag)
    
    return hamiltonian_part# + 0.5*gamma*(lindblad_part_1 + lindblad_part_2)    

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

    #plot_steady_state_td(solRK, title="Calderia Leggett with {}states".format(n))
