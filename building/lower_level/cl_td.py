
import numpy as np
from helper_functions import *


n = 15 # Hilbert Space Dimension
gamma = 0.001 # Damping Rate

# Annihilation and Creation Operators
Ldag = create_annihilation_operator(n)
a = create_creation_operator(n)

# Hamiltonian
q = (Ldag + a)/2
p = 1j*(Ldag - a)/2
H = (np.dot(p,p) + np.dot(q,q)) + gamma/2 * get_anti_commutator(q,p)
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
L = q + p
Ldag = np.conjugate(L).T
#H = (wo/2) * sz 

# Encoding Equation
def LinEm(x):
    res = -1j*(np.dot(H,x)-np.dot(x,H))
    res += gamma*(nth+1)*(multi_dot([Ldag,x,L])-0.5*multi_dot([L,Ldag,x])-0.5*multi_dot([x,L,Ldag]))
    #res += gamma*nth*(multi_dot([L,x,Ldag])-0.5*multi_dot([Ldag,L,x])-0.5*multi_dot([x,Ldag,L]))    
    return res

if __name__ == "__main__":
    init = make_initial_density_matrix(n)
    t_i = 0
    t_f = 100
    nsteps = 800

    h = (t_f-t_i)/nsteps
    solRK = np.zeros((nsteps+1,n, n),dtype=complex)
    solRK[0]=init

    # Solving
    solRK = solver(solRK, LinEm, h)

    # Visualising
    plot_density_matrix_elements(solRK, title="QHO Thermal Bath with {}states".format(n))
    plot_trace_purity(solRK, title="QHO Thermal Bath with {}states".format(n))

    plot_steady_state_td(solRK, title="Calderia Leggett with {}states".format(n))
