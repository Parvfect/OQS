

from helper_functions import *
import numpy as np

def low_temperature_hal(n, gamma = 0.05, w=3, cutoff=200):
    q = create_position_operator(n)
    p = create_momentum_operator(n)
    H = (np.dot(p,p) + np.dot(q,q) + gamma/2 * get_anti_commutator(q,p))
    L = np.sqrt(w)*q + np.sqrt(1/w)*(1j - w/cutoff)*p

    return H, L, gamma

def high_temperature_hal(n, gamma=0.05, T=10):
    q = create_position_operator(n)
    p = create_momentum_operator(n)
    H = (np.dot(p,p) + np.dot(q,q) + gamma/2 * get_anti_commutator(q,p))
    L = q + (1/T)*(1j)*p
    return H, L, gamma

def temperature_study(n):

    T = np.arange(1, 50, 2)
    ssps = []

    for i in tqdm(T):
        H, L, gamma = high_temperature_hal(n, T=i)
        solRK = run_simulation(n, H, L, gamma)
        ssp = [np.trace(np.dot(solRK[-1], solRK[-1]))]
        ssps.append(ssp)

    plt.plot(T, ssps)
    plt.title("Steady State Purity vs Temperature for the High Temperature Caldeira-Leggett Model")
    plt.xlabel("Temperature")
    plt.ylabel("Steady State Purity")
    plt.grid()
    plt.show()

def frequency_analysis(n):
    w = np.arange(1, 50, 2)
    ssps = []

    for i in tqdm(w):
        H, L, gamma = low_temperature_hal(n, w=i)
        solRK = run_simulation(n, H, L, gamma)
        ssp = [np.trace(np.dot(solRK[-1], solRK[-1]))]
        ssps.append(ssp)

    plt.plot(w, ssps)
    plt.title("Steady State Purity vs frequency for the High Temperature Caldeira-Leggett Model")
    plt.xlabel("Frequency")
    plt.ylabel("Steady State Purity")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=10, title="CL model")
    
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=11, title="CL model")
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=12, title="CL model")
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=13, title="CL model")
    
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=14, title="CL model")
    
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=15, title="CL model")
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=16, title="CL model")
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=17, title="CL model")
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=18, title="CL model")
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=19, title="CL model")
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=20, title="CL model")
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=21, title="CL model")
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=22, title="CL model")
    n = 20
    H, L, gamma = high_temperature_hal(n, T = 1000)
    solRK = run_simulation(n, H, L, gamma, t_f=23, title="CL model")
    