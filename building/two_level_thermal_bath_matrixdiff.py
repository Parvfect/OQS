
# Imports 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from helper_functions import *


# Defining Operators
sz = np.array([[1,0],[0,-1]])
sp = create_creation_operator(2)
sm = create_annihilation_operator(2)
wo = 10
H = (wo/2) * sz 
gamma = 0.1
nth = 2

# Encoding Equation
def LinEm():
    res = -1j* (np.kron(np.eye(2), H) - np.kron(H.T, np.eye(2)))
    res += gamma* (nth+1) * (np.kron(sp.T, sm) - 0.5*(np.kron(np.eye(2), np.dot(sp, sm)) + np.kron(np.dot(sp, sm).T, np.eye(2))))
    res += gamma* (nth) * (np.kron(sm.T, sp) - 0.5*(np.kron(np.eye(2), np.dot(sm, sp)) + np.kron(np.dot(sm, sp).T, np.eye(2))))
    return res

L = LinEm()

def handler(rho):
    return (np.dot(L, rho.flatten("F"))).reshape(2,2).T

if __name__ == "__main__":
    # Simulating
    init = make_initial_density_matrix(2)

    t_i = 0
    t_f = 30
    nsteps = 750

    h = (t_f-t_i)/nsteps


    solRK = np.zeros((nsteps+1,2,2),dtype=complex)
    solRK[0]=init

    solver(solRK, handler, h)

    # Visualising
    plot_density_matrix_elements(solRK)
    plot_trace_purity(solRK)