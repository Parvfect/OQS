

# Imports 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from helper_functions import *


# Defining Operators

sz = np.array([[1,0],[0,-1]])
sp = np.array([[0,1],[0,0]])
sm = np.array([[0,0],[1,0]])

# Equation
def LinEm(x, omega0=10, nth=2, gamma = 0.1):
    res = -1j*(omega0/2)*(np.dot(sz,x)-np.dot(x,sz))
    res += gamma*(nth+1)*(multi_dot([sm,x,sp])-0.5*multi_dot([sp,sm,x])-0.5*multi_dot([x,sp,sm]))
    res +=     gamma*nth*(multi_dot([sp,x,sm])-0.5*multi_dot([sm,sp,x])-0.5*multi_dot([x,sm,sp]))    
    return res


if __name__ == "__main__":
    init = np.array([[0.5,0.5],[0.5,0.5]], dtype=complex)

    t_i = 0
    t_f = 20
    nsteps = 500

    h = (t_f-t_i)/nsteps
    solRK = np.zeros((nsteps+1,2,2),dtype=complex)
    solRK[0]=init

    # Solving
    solRK = solver(solRK, LinEm, h)

    # Visualising
    plot_density_matrix_elements(solRK)
    plot_trace_purity(solRK)