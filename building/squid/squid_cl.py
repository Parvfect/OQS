
# Faster Matrix multiplication is the need of the hour
# I need to create a situation where the steady state purity does not settle at 1/n for n being the 
# number of states in the system. I have to do this via utilization of the cosphi - which is indeed 
# the nonlinearity in the system.

import numpy as np
from helper_functions import *

def hal(n, gamma=0.05, flux=0, alpha=1.2, delta=1):
    q, p = create_position_operator(n), create_momentum_operator(n)
    cphi = cosphi_taylor(alpha*q+flux, 20)
    
    H = (np.dot(p,p) + np.dot(q,q) -delta*(cphi)) + gamma/2 * get_anti_commutator(q,p)
     
    a = create_annihilation_operator(n)
    adag = create_creation_operator(n)
    L = 0.1*a
    L = np.zeros((n,n))
    
    return H, L, gamma


def flux_study(n=2):
    flux = np.arange(0,0.5, 0.05)
    ssps = []
    for i in tqdm(flux):
        n = 4
        H, L, gamma = hal(14, gamma=0.9, flux=i)
        solRK = run_simulation(14, H, L, gamma, t_i=0, t_f=1000, h=1e-3, plotting=False)
        linear_entropy = get_linear_entropy(solRK)
        plt.plot(linear_entropy, label=f"{i}")
        ssp = np.trace(np.dot(solRK[-1], solRK[-1]))
        ssps.append(ssp)

    plt.title("Linear Entropy vs Time for the SQUID model")
    plt.xlabel("Time")
    plt.ylabel("Linear Entropy")
    plt.legend()
    plt.grid()
    plt.show()


    plt.plot(flux, ssps)
    plt.title("Steady State Purity vs Flux")
    plt.xlabel("Flux")
    plt.ylabel("Steady State Purity")
    plt.grid()
    plt.show()
    return
        

if __name__ == "__main__":
    
    #flux_study()

    H, L, gamma = hal(40, gamma=0.9, flux=0)
    run_simulation(40, H, L, gamma, t_i=0, t_f=1000, h=1e-2, title="SQUID dynamics 40 states")
    