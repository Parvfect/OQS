
from helper_functions import *
 
n = 1

adag = create_annihilation_operator(n) # Annihilation operator
a = create_creation_operator(n) # Creation operator

phi_o = 1 
phi_x = 0.5
Q =  (1j)* (adag - a) # Momentum operator
phi = (adag + a) # Flux operator (analogous to position operator)
cphi = create_cos_phi(phi, phi_o, phi_x)
gamma = 0.05


H =  (np.dot(phi, phi) + np.dot(Q, Q) )#- cphi )# + (gamma/2)*get_commutator(phi, Q)


def potential(x):
    return x**2 - np.cos((2*pi/phi_o)*(x +  phi_x))

if __name__ == "__main__":
    x = np.linspace(-2, 2, 1000)
    plt.plot(x, potential(x))
    plt.show()