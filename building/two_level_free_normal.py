

# Imports 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from helper_functions import *

# Defining Operators

sz = np.array([[1,0],[0,-1]])

# Equation
def LinEm(x, omega0=10, nth=2, gamma = 0.1):
    res = -1j*(omega0/2)*(np.dot(sz,x)-np.dot(x,sz))
    return res

if __name__ == "__main__":
    init = np.array([[0.5,0.5],[0.5,0.5]], dtype=complex)

    t_i = 0
    t_f = 20
    nsteps = 500

    h = (t_f-t_i)/nsteps
    solRK = np.zeros((nsteps+1,2,2),dtype=complex)
    solRK[0]=init
    solRK = solver(solRK, LinEm, h)
    plot_density_matrix_elements(solRK)


