
# Needs fixing
# Add cosphi and effective hammy term and we have SQUID model
# And then we can parameterize

import numpy as np
from helper_functions import *


n = 14 # Hilbert Space Dimension
#H = (wo/2) * sz 

def squid_cl(n, gamma=0.005, flux=0):
    a = create_annihilation_operator(n)
    q = create_position_operator(n)
    p = create_momentum_operator(n)
    cphi = cosphi_taylor(q+flux, 0)
    
    H = (np.dot(p,p) + np.dot(q,q) - cphi) + gamma/2 * get_anti_commutator(q,p)

    L = (np.dot(a,a))
    return H, L, gamma


def flux_study():
    fluxes = np.arange(-10,10,1)
    ssps = []
    for flux in fluxes:
        H, L, gamma = squid_cl(14, flux=flux)
        t = run_simulation(n, H, L, gamma, t_i=0, t_f=100, h=1e-2)
        ssps.append(get_purity_simple(t[-1]))
    plt.plot(fluxes, ssps)
    plt.show()

if __name__ == "__main__":
    
    #flux_study()
    H, L, gamma = squid_cl(14, flux=0.5)
    run_simulation(n, H, L, gamma, t_i=0, t_f=400, h=1e-2)
    