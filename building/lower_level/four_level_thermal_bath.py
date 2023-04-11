


# Imports 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from helper_functions import *


# Defining Operators
dim = 2
sp = create_creation_operator(dim)
sm = create_annihilation_operator(dim)
wo = 10
q = sp + sm
p = 1j*(sp - sm)
H = q**2 + p**2
gamma = 0.1
nth = 4


# Equation
def LinEm(x, omega0=10, nth=2, gamma = 0.1):
    res = -1j*(omega0/2)*(np.dot(H,x)-np.dot(x,H))
    res += gamma*(nth+1)*(multi_dot([sm,x,sp])-0.5*multi_dot([sp,sm,x])-0.5*multi_dot([x,sp,sm]))
    res +=     gamma*nth*(multi_dot([sp,x,sm])-0.5*multi_dot([sm,sp,x])-0.5*multi_dot([x,sm,sp]))    
    return res


if __name__ == "__main__":
    init = make_initial_density_matrix(dim)

    t_i = 0
    t_f = 200
    nsteps = 1000

    h = (t_f-t_i)/nsteps
    solRK = np.zeros((nsteps+1,dim,dim),dtype=complex)
    solRK[0]=init

    # Solving
    solRK = solver(solRK, LinEm, h)

    # Visualising
    plot_density_matrix_elements(solRK)
    plot_trace_purity(solRK)