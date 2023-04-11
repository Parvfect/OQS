
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt

n = 6


a = create_annihilation_operator(n)
adag = create_creation_operator(n)
gamma = 0.2
hbar = 1e-34/2/np.pi
m = 1e-27
w = 1e12


q = (adag + a)/2 # np.sqrt(hbar/m*w)*
p = (1j)* (adag - a)/2 #  np.sqrt(hbar*m*w/2)*
#cphi = create_cos_phi(n, q, 1, 1)

Hp = np.dot(p,p)
Hq = np.dot(q, q)
#Hcphi = cphi

L = np.sqrt(2*gamma)*(a) 
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
    L_part =  np.dot(L, np.dot(x, Ldag)) -0.5*(get_anti_commutator(np.dot(Ldag, L), x))
    return H_part + L_part


if __name__ == "__main__":
    # Setting simulation parameters
    t_i = 0
    t_f = 2000
    nsteps = 2000
    h = (t_f-t_i)/nsteps
    t = np.zeros((nsteps+1, n,n), dtype=complex)
    t[0] = init
    t = solver(t, handler, h)

    # Plotting
    plot_density_matrix_elements(t, 0)
    plot_trace_purity(t)

    # Looking at purity

    counter = 100
    for i in t:
        #print(np.trace(i))
        print(np.trace(np.dot(i,i)))
        #print(np.trace(i))
        #print(np.dot(i,i))
        #print(np.trace(np.dot(i,i)))
        counter -= 1
        if counter == 0:
            break

    """
    trace = get_trace(t)
    purity = get_purity(t)
    plt.plot(purity, label = r'$\mathrm{Tr}[\rho^2]$')
    plt.show()
    """