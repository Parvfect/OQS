

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
adag = create_annihilation_operator(n) # Annihilation operator
a = create_creation_operator(n) # Creation operator

Q = (np.sqrt((hbar*C*w)/(2)) * (1j)* (adag - a)) # Momentum operator
phi = (np.sqrt((hbar)/(2*C*w))*((adag + a))) # Flux operator (analogous to position operator)

# Dimensionless position and momentum operators
X = np.sqrt((C*w)/hbar) * phi
P = np.sqrt((1)/(C*w*hbar)) * Q
cphi = muomega * (create_cos_phi(X, phi_o, phi_x, alpha))
#cphi = cosphi_taylor(X-0.3, n)
H =  (np.dot(X, X) + np.dot(P, P) - cphi) + (hbar*gamma/2)*get_anti_commutator(X, P)
L = gamma**(0.5) * (X + 0.01*(1j - epsilon/2) * P)
#L = a - adag
Ldag = L.conj().T


def handler(x):
    hamiltonian_part = (-1j)* (np.dot(H, x) - np.dot(x, H))
    lindblad_part_1 = get_commutator(L, np.dot(x, Ldag))
    lindblad_part_2 = get_commutator(np.dot(L, x), Ldag)
    return hamiltonian_part + 0.5*(lindblad_part_1 + lindblad_part_2)


if __name__ == "__main__":

    # Setting simulation parameters
    run_simulation(n, handler, t_i=0, t_f=200, h=1e-2)