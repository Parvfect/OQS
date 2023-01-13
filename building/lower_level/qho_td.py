
from helper_functions import *
import numpy as np

n = 4

adag = create_annihilation_operator(n) # Annihilation operator
a = create_creation_operator(n) # Creation operator

hbar = 1e-34
m = 9.1e-31
w = 1e11
l = 1
C = 1
je = 9e-22
gamma = 0.005 * w # Relaxation Rate
cutoff = 20
e = 1.6e-19
phi_o = hbar/2*e
phi_x = 0.5*phi_o
gamma = 0.1

q = (adag + a)
p =  (1j)* (adag - a)
cphi = create_cos_phi(n, q, 1, 1)

Hp = np.dot(p,p)
Hq = np.dot(q, q)
Hcphi = cphi

L = np.sqrt(2*gamma)*(adag)
Ldag = np.matrix(L).H

H = Hp + Hq# + Hcphi

"""
X = np.sqrt(C*wo/hbar) * phi # Position operator dimensionless
P = np.sqrt(1/(C*wo*hbar)) * Q # Momentum operator dimensionless

Hc = (hbar*gamma/2) * (np.dot(X, P) + np.dot(P, X)) # Counter term to Hamiltonian due to casting in Lindblad form
H += Hc
"""
init = make_initial_density_matrix(n)

def handler(x):
    H_part = (-1j)* get_commutator(H, x)
    L_part =  0.5 * (get_commutator(L, np.dot(x, Ldag)) + get_commutator(np.dot(L, x), Ldag))
    return H_part + L_part


if __name__ == "__main__":
    # Setting simulation parameters
    t_i = 0
    t_f = 20
    nsteps = 2000
    h = (t_f-t_i)/nsteps
    t = np.zeros((nsteps+1, n,n), dtype=complex)
    t[0] = init
    t = solver(t, handler, h)

    # Plotting
    plot_density_matrix_elements(t)
    plot_trace_purity(t)