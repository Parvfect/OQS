
# Imports 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from helper_functions import *


# Defining Operators
sz = np.array([[1,0],[0,-1]])
wo = 10
H = (wo/2) * sz 

# Equation
def LinEm():
    res = -1j* (np.kron(np.eye(2), H) - np.kron(H.T, np.eye(2)))
    return res

L = LinEm()

def handler(rho):
    return np.dot(L, rho.flatten("F")).reshape(2,2).T

if __name__ == '__main__':
    # Simulating
    init = np.array([[0.5,0.5],[0.5,0.5]], dtype=complex)

    t_i = 0
    t_f = 20
    nsteps = 500

    h = (t_f-t_i)/nsteps

    solRK = np.zeros((nsteps+1,2,2),dtype=complex)
    solRK[0]=init
    solRK = solver(solRK, handler, h)
    # Visualising
    plot_density_matrix_elements(solRK)

