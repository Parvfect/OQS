
# Needs fixing
# Add cosphi and effective hammy term and we have SQUID model
# And then we can parameterize

import numpy as np
from helper_functions import *


n = 14 # Hilbert Space Dimension
gamma = 0.5 # Damping Rate

# Annihilation and Creation Operators
adag = create_annihilation_operator(n)
a = create_creation_operator(n)

# Hamiltonian
q = (adag + a)/2
p = 1j*(adag - a)/2
cphi = cosphi_taylor(q-0.5, 10)
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
L = gamma**(0.5) * (np.dot(a,a))
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
    run_simulation(n, LinEm, t_i=0, t_f=300, h=1e-2)