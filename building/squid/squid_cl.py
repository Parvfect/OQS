
# Needs fixing
# Add cosphi and effective hammy term and we have SQUID model
# And then we can parameterize

import numpy as np
from helper_functions import *

def hal(n, gamma=0.05, flux=0):
    q, p = create_position_operator(n), create_momentum_operator(n)
    cphi = cosphi_taylor(q+flux, 20)
    
    H = (np.dot(p,p) + np.dot(q,q) -(cosphi_taylor(q+flux, 20))) + gamma/2 * get_anti_commutator(q,p)
     
    a = create_annihilation_operator(n)
    adag = create_creation_operator(n)
    L = 0.01*a 
    
    return H, L, gamma

def flux_study(n):
    fluxes = np.arange(-2, 2, 0.1)
    ssps = []
    sspus = []

    for i in tqdm(fluxes):
        H, L, gamma = hal(n, flux=i)
        solRK = run_simulation(n, H, L, gamma, t_i=0, t_f=100, h=1e-3)
        sspu = measure_pureness_state(solRK[-1])
        ssp = np.trace(np.dot(solRK[-1], solRK[-1]))
        ssps.append(ssp)
        sspus.append(sspu)
    
    plt.plot(fluxes, ssps)
    plt.title("Steady State Purity vs Flux for the SQUID Model")
    plt.xlabel("Flux")
    plt.ylabel("Steady State Purity")
    plt.grid()
    plt.show()

    plt.plot(fluxes, sspus)
    plt.title("Steady State Pureness vs Flux for the SQUID Model")
    plt.xlabel("Flux")
    plt.ylabel("Steady State Purity")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    
    #flux_study(14)
    
    n = 10
    H, L, gamma = hal(n, flux=0)
    run_simulation(n, H, L, gamma, t_i=0, t_f=5000, h=1e-1)
    