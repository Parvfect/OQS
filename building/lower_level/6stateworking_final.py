
import numpy as np
from helper_functions import *


n = 6 # Hilbert Space Dimension

# Annihilation and Creation Operators
adag = create_annihilation_operator(n)
a = create_creation_operator(n)

# Hamiltonian
q = (adag + a)/2
p = 1j*(adag - a)/2
H = 15*(np.dot(p,p) + np.dot(q,q))
H = np.array(H)

# Initial Density Matrix
rho_0 = [[0.5,0.5],[0.5,0.5]]
w = 2e13
hbar = 1e-34
kb = 1.38e-23
T = 300
gamma = 0.1
#print(1/(np.exp((hbar*w)/(kb*T))-1))
nth = 1/(np.exp((hbar*w)/(kb*T))-1) #2
sz = np.array([[1,0],[0,-1]])
wo = 10
#H = (wo/2) * sz 

# Encoding Equation
def LinEm():
    res = -1j* (np.kron(np.eye(n), H) - np.kron(H.T, np.eye(n)))
    res += gamma* (nth+1) * (np.kron(a.T, adag) - 0.5*(np.kron(np.eye(n), np.dot(a, adag)) + np.kron(np.dot(a, adag).T, np.eye(n))))
    res += gamma* (nth) * (np.kron(adag.T, a) - 0.5*(np.kron(np.eye(n), np.dot(adag, a)) + np.kron(np.dot(adag, a).T, np.eye(n))))
    return res

L = LinEm()

def handler(rho):
    return (np.dot(L, rho.flatten("F"))).reshape(n,n).T

if __name__ == "__main__":
    # Simulating
    init = make_initial_density_matrix(n)
    t_i = 0
    t_f = 30
    nsteps = 750

    h = (t_f-t_i)/nsteps


    solRK = np.zeros((nsteps+1,n,n),dtype=complex)
    solRK[0]=init

    solver(solRK, handler, h)

    # Visualising
    plot_density_matrix_elements(solRK, title="QHO Thermal Bath with {}states".format(n))
    plot_trace_purity(solRK, title="QHO Thermal Bath with {}states".format(n))
