

import numpy as np
from helper_functions import *
from numpy.linalg import multi_dot

n = 14

# Annihilation and Creation Operators
adag = create_annihilation_operator(n)
a = create_creation_operator(n)

# Hamiltonian
q = (adag + a)/2
p = 1j*(adag - a)/2
H = (np.dot(p,p) + np.dot(q,q))
H = np.array(H)

# Parameters
w = 2e13
hbar = 1e-34
kb = 1.38e-23
T = 300
gamma = 0.01
nth = 1/(np.exp((hbar*w)/(kb*T))-1) #2
L  = adag + a
Ldag = np.conjugate(L).T

# Encoding Equation
def LinEm(x):
    res = -1j*(np.dot(H,x)-np.dot(x,H))
    res += gamma*(nth+1)*(multi_dot([Ldag,x,L])-0.5*multi_dot([L,Ldag,x])-0.5*multi_dot([x,L,Ldag]))
    res += gamma*nth*(multi_dot([L,x,Ldag])-0.5*multi_dot([Ldag,L,x])-0.5*multi_dot([x,Ldag,L]))    
    return res

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
    plot_trace_purity(solRK)
    plot_diagonal_density_matrix_elements(solRK)
    plot_offdiagonal_density_matrix_elements(solRK)
    plot_steady_state_td(solRK)
    wigner_plot_steady_state(solRK)
