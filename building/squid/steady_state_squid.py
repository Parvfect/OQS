
# Maybe try the normal one first aloha
from helper_functions import *

# Hilbert Space Dimensions
n = 5

# Constants
pi = np.pi
C = 5e-15
l = 3e-10
je = 9e-22
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
cphi = muomega * create_cos_phi(X, phi_o, phi_x, alpha)

H =  (np.dot(X, X) + np.dot(P, P) - cphi) + (hbar*gamma/2)*get_commutator(X, P)
L = gamma**(0.5) * (X + (1j - epsilon/2) * P)
Ldag = L.conj().T


def first_order_equation():
    """ First order equation for steady state """
    hamiltonian_part = -(1j) * (np.kron(H, np.identity(n)) - np.kron(np.identity(n), H)) 
    lindblad_part_1 = np.kron(Ldag, L) 
    lindblad_part_2 = -0.5*(np.kron(np.identity(n), np.dot(Ldag, L)) + np.kron(np.dot(Ldag, L), np.identity(n)))
    return hamiltonian_part + lindblad_part_1 + lindblad_part_2

L = first_order_equation()

if __name__ == "__main__":
    sol = null_space(L)
    print("Solution obtained for n = ", n)
    sol = sol.reshape(n,n)
    print(sol)
    sol = np.array(sol).astype(np.float64)
    diag = np.diag(sol)
    diag = diag[::-1]
    """
    trange = np.arange(n)
    plt.plot(diag,trange)
    plt.title("Probability Distribution of the Steady State Density Matrix")
    plt.xlabel("Diagonal Elements")
    plt.ylabel("Probability")
    plt.show()
    """
    #plt.imshow(sol, interpolation='nearest')
    #print(type(sol))
    plt.imshow(np.array(sol).astype(np.float64))
    plt.title("Steady State Density Matrix, n = " + str(n))
    plt.show()
    trace = np.trace(sol)
    purity = np.trace(np.dot(sol, sol))
    print("Trace: ", trace)
    print("Purity: ", purity)