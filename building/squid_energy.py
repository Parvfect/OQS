
from helper_functions import *

import numpy as np
import matplotlib.pyplot as plt

n = 100

adag = create_annihilation_operator(n) # Annihilation operator
a = create_creation_operator(n) # Creation operator

phi = (adag + a)/2
Q = (1j)* (adag - a)/2
cphi = create_cos_phi(n, phi, 1, 0.5)

H = np.dot(Q, Q) + np.dot(phi, phi) + cphi

#energy_eigenvalues = np.linalg.eigvalsh(H)
"""
plt.plot(energy_eigenvalues)
plt.title("Energy Eigenvalues for SQUID system")
plt.show()
"""

""" Form of potential energy for SQUID system """

def V(x):
    return np.dot((phi - x), (phi - x)) - cphi

x = np.linspace(-1, 1, 1000)
y = [V(i) for i in x]

plt.plot(x, y[:,0,0])
plt.title("Potential Energy for SQUID system")
plt.show()
