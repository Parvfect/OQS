
from helper_functions import *

# Hilbert Space Dimensions
n = 14

# Constants
pi = np.pi
C = 5e-15
l = 3e-10
je = 9.99e-22
hbar = 1e-34
w = 8.16e11
e = 1.6e-19
phi_o = hbar/(2*e)
phi_x = 0.5* phi_o
mu = je/hbar
alpha = np.sqrt((4 * pi*pi * hbar)/(phi_o*phi_o*C))
muomega = mu/w # 
cutoff = 20 * w
epsilon = w/cutoff # Cutoff frequency
gamma = 0.05 # Damping Rate

# Operators
a = create_annihilation_operator(n) # Annihilation operator
adag = create_creation_operator(n) # Creation operator

momentum_constant = np.sqrt((hbar*C*w))
position_constant = np.sqrt((hbar)/(C*w))

X = np.sqrt((C*w)/hbar) * position_constant * create_position_operator(n) 
P = np.sqrt((1)/(C*w*hbar)) * momentum_constant * create_momentum_operator(n)
cphi = muomega * create_cos_phi(position_constant*create_position_operator(n), phi_o, phi_x, alpha)
#cphi = cosphi_taylor(create_position_operator(n), 40)

H =  (np.dot(X, X) + np.dot(P, P) ) + (hbar*gamma/2)*get_anti_commutator(X, P) 
L = gamma**(0.5) * (X  + (1j - epsilon/2) * P)
Ldag = L.conj().T

def hal(n, gamma=0.05, C=5e-15, l=3e-10, w=8.16e11, flux_ratio = 0.5, cutoff_bound=20):
        
    momentum_constant = np.sqrt((hbar*C*w))
    position_constant = np.sqrt((hbar)/(C*w))

    X = np.sqrt((C*w)/hbar) * position_constant * create_position_operator(n) 
    P = np.sqrt((1)/(C*w*hbar)) * momentum_constant * create_momentum_operator(n)
    cphi = muomega * create_cos_phi(position_constant*create_position_operator(n), phi_o, phi_x, alpha)
    #cphi = cosphi_taylor(create_position_operator(n), 40)

    H =  (np.dot(X, X) + np.dot(P, P) ) + (hbar*gamma/2)*get_anti_commutator(X, P) 
    L = gamma**(0.5) * (X  + (1j - epsilon/2) * P)

    return H, L, gamma

def handler(x):
    hamiltonian_part = (-1j)* (np.dot(H, x) - np.dot(x, H))
    lindblad_part_1 = get_commutator(L, np.dot(x, Ldag))
    lindblad_part_2 = get_commutator(np.dot(L, x), Ldag)
    return hamiltonian_part + 0.5*(lindblad_part_1 + lindblad_part_2)

if __name__ == "__main__":

    # Setting simulation parameters
    H, L, gamma = hal(n)
    run_simulation(n, H, L, gamma)
    #run_normal_simulation(n, handler)
      