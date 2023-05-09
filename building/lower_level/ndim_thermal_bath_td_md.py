
import numpy as np
from helper_functions import *


n = 24# Hilbert Space Dimension

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
gamma = 0.001
#print(1/(np.exp((hbar*w)/(kb*T))-1))
nth = 1/(np.exp((hbar*w)/(kb*T))-1) #2
sz = np.array([[1,0],[0,-1]])
wo = 10
L = q + 1j*p
Ldag = np.conj(L.T)
#H = (wo/2) * sz 

# Encoding Equation
def LinEm():
    hamiltonian_part = -(1j) * (np.kron(H, np.identity(n)) - np.kron(np.identity(n), H)) 
    lindblad_part_1 = np.kron(Ldag, L) 
    lindblad_part_2 = -0.5*(np.kron(np.identity(n), np.dot(Ldag, L)) + np.kron(np.dot(Ldag, L), np.identity(n)))
    return hamiltonian_part + lindblad_part_1 + lindblad_part_2

L = LinEm()

def handler(rho):
    return (np.dot(L, rho.flatten("F"))).reshape(n,n).T

if __name__ == "__main__":
    steady_state = steady_state_solver(L)
    print(steady_state[0])
    #plt.imshow(steady_state[0])
    #plt.show()