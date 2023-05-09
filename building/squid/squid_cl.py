
# Needs fixing
# Add cosphi and effective hammy term and we have SQUID model
# And then we can parameterize

import numpy as np
from helper_functions import *

def hal(n, gamma=0.05, flux=0):
    q, p = create_position_operator(n), create_momentum_operator(n)
    cphi = cosphi_taylor(q, 20)
    
    H = (np.dot(p,p) + np.dot(q,q) ) + gamma/2 * get_anti_commutator(q,p)
     
    a = create_annihilation_operator(n)
    adag = create_creation_operator(n)
    L = q + 0.01*(1j)* p
    
    return H, L, gamma

if __name__ == "__main__":
    
    n = 14
    H, L, gamma = hal(n, flux=0.5)
    run_simulation(n, H, L, gamma, t_i=0, t_f=200, h=1e-2)
    