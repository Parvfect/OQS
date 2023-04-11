

# Imports 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from helper_functions import *


# Defining Operators
dim = 4
sp = create_creation_operator(dim)
sm = create_annihilation_operator(dim)
wo = 10
q = sp + sm
p = 1j*(sp - sm)
H = q*wo/2 + 0.5*p**2
gamma = 0.1
nth = 4

# Encoding Equation
def LinEm():
    res = -1j* (np.kron(np.eye(dim), H) - np.kron(H.T, np.eye(dim)))
    res += gamma* (nth+1) * (np.kron(sp.T, sm) - 0.5*(np.kron(np.eye(dim), np.dot(sp, sm)) + np.kron(np.dot(sp, sm).T, np.eye(dim))))
    res += gamma* (nth) * (np.kron(sm.T, sp) - 0.5*(np.kron(np.eye(dim), np.dot(sm, sp)) + np.kron(np.dot(sm, sp).T, np.eye(dim))))
    return res

L = LinEm()

def handler(rho):
    return (np.dot(L, rho.flatten("F"))).reshape(4,4).T

if __name__ == "__main__":
    # Simulating
    init = make_initial_density_matrix(4)

    t_i = 0
    t_f = 30
    nsteps = 750

    h = (t_f-t_i)/nsteps


    solRK = np.zeros((nsteps+1,dim, dim),dtype=complex)
    solRK[0]=init

    solver(solRK, handler, h)

    # Visualising
    plot_density_matrix_elements(solRK)
    plot_trace_purity(solRK)