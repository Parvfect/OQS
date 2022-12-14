


""" Time Dependent solver for the SQUID model 
wihtout matrix diff form, might need to iron out the constants firsts
work with this and then extend to matrix diff and steady state solutions
"""

import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *

n = 3
adag = create_annihilation_operator(n) # Annihilation operator
a = create_creation_operator(n) # Creation operator

phi = 0.2 * (adag + a) # Flux operator (analogous to position operator)
Q = 0.5 * (1j)* (adag - a) # Momentum operator
cphi = 0.5*create_cos_phi(n, phi, 1, 1)
L =  phi + (1j - 0.2) * Q # Lindblad operator
Ldag = np.conj(L.T) # Conj transpose of Lindblad

L = np.array(L)

H = np.dot(Q,Q) + np.dot(phi, phi) + cphi
H = np.array(H)
init = make_initial_density_matrix(n)
# Setting simulation parameters
t_i = 0
t_f = 500
nsteps = 20000
h = (t_f-t_i)/nsteps
t = np.zeros((nsteps+1, n,n), dtype=complex)
t[0] = init

def first_order_equation():
    """ First order equation for steady state """
    hamiltonian_part = -1j* (np.dot(H, x) - np.dot(x, H)) 
    lindblad_part = 0.5* (get_commutator(L, np.dot(x, Ldag)) + get_commutator(np.dot(L,x), Ldag))
    
    return hamiltonian_part + lindblad_part

def handler(x):
  return -1j* (np.dot(H, x) - np.dot(x, H))

def RK4step(x, h):
    k1 = handler(x)
    k2 = handler(x+h*k1/2)
    k3 = handler(x+h*k2/2)
    k4 = handler(x+h*k3)
    return x+(h/6)*(k1+2*k2+2*k3+k4)

if __name__ == "__main__":
    t = solver(t, handler, h)
    plot_density_matrix_elements(t)
    plot_trace_purity(t)