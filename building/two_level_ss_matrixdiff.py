
# Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
from helper_functions import *


# Defining Operators

sz = np.array([[1,0],[0,-1]])
sp = np.array([[0,1],[0,0]])
sm = np.array([[0,0],[1,0]])
wo = 10
H = (wo/2) * sz 
gamma = 0.4
nth = 2

# Equation
def LinEm():
    res = -1j* (np.kron(np.eye(2), H) - np.kron(H.T, np.eye(2)))
    res += gamma* (nth+1) * (np.kron(sp.T, sm) - 0.5*(np.kron(np.eye(2), np.dot(sp, sm)) + np.kron(np.dot(sp, sm).T, np.eye(2))))
    res += gamma* (nth) * (np.kron(sm.T, sp) - 0.5*(np.kron(np.eye(2), np.dot(sm, sp)) + np.kron(np.dot(sm, sp).T, np.eye(2))))
    return res

L = LinEm()

if __name__ == '__main__':
    print(steady_state_solver(L))