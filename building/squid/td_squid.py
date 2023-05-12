

from helper_functions import *

# Hilbert Space Dimensions
pi = np.pi
je = 9.99e-22
hbar = 1e-34
e = 1.6e-19
phi_o = hbar/(2*e)
mu = je/hbar


def hal(n, gamma=0.05, C=5e-15, l=3e-10, w=8.16e11, flux_ratio = 0.5, cutoff_bound=20):

    phi_x = flux_ratio* phi_o
    alpha = np.sqrt((4 * pi*pi * hbar)/(phi_o*phi_o*C))
    muomega = mu/w  
    cutoff = cutoff_bound * w
    epsilon = w/cutoff # Cutoff frequency


    X = np.sqrt((C*w)/hbar) * np.sqrt((hbar)/(C*w)) * create_position_operator(n) 
    P = np.sqrt((1)/(C*w*hbar)) * np.sqrt((hbar*C*w)) * create_momentum_operator(n)
    cphi = cosphi_taylor((X+0.5), 20)
    H =  (np.dot(X, X) + np.dot(P, P) - cphi) + (hbar*gamma/2)*get_anti_commutator(X, P)
    #L = gamma**(0.5) * (X + 0.001*(1j - epsilon/2) * P)
    a, adag = create_annihilation_operator(n), create_creation_operator(n)    
    L = gamma**(0.5)* (a + adag)

    return H, L, gamma

if __name__ == "__main__":
    n = 14
    H, L, gamma = hal(n)
    run_simulation(n, H, L, gamma, t_f=500, h=1e-2, title="SQUID")