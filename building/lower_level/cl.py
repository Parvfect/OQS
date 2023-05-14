

from helper_functions import *
import numpy as np
from matrixeqs import System

n = 14


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

def first_order_equation(H, L, n, Ldag):
    hamiltonian_part = -(1j) * (np.kron(H, np.identity(n)) - np.kron(np.identity(n), H)) 
    lindblad_part_1 = np.kron(Ldag, L) 
    lindblad_part_2 = -0.5*(np.kron(np.identity(n), np.dot(Ldag, L)) + np.kron(np.dot(Ldag, L), np.identity(n)))
    return hamiltonian_part + 0.5*gamma*(lindblad_part_1 + lindblad_part_2)

def compare_entropy(factor = 1e-3):

    a = create_annihilation_operator(n)
    adag = create_creation_operator(n)
    Ls = [a, np.dot(a,a), a+adag, a-adag, np.dot(a,a)-np.dot(adag,adag)]
    Ls = 0.001*np.array(Ls)
    labels = ["a", "a^2", "a+a^dag", "a-a^dag", "a^2-a^dag^2"]
    ssps = []

    for i in range(len(Ls)):
        L = Ls[i]
        solRK = run_simulation(n, H, L, gamma, t_f=5000, h=0.1, plotting=False)
        linear_entropy = get_linear_entropy(solRK)
        plt.plot(linear_entropy, label=labels[i])
        ssps.append([np.trace(np.dot(solRK[-1], solRK[-1]))])
    
    plt.title("Linear Entropy vs Time for the Caldeira-Leggett Model")
    plt.xlabel("Time")
    plt.ylabel("Linear Entropy")
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(labels, ssps, "o")
    plt.title("Steady State Purity vs L for the Caldeira-Leggett Model")
    plt.xlabel("L")
    plt.ylabel("Steady State Purity")
    plt.grid()
    plt.show()

    return

if __name__ == "__main__":
    n = 14
    q = create_position_operator(n)
    p = create_momentum_operator(n)
    gamma = 0.5
    H = (np.dot(p,p) + np.dot(q,q) + gamma/2 * get_anti_commutator(q,p))
    a = create_annihilation_operator(n)
    adag = create_creation_operator(n)
    L = 1e-3*(np.dot(a,a))
     
    
    solRK1 = run_simulation(n, H, L, gamma, t_f=5000, h=0.1, plotting=True)
    compare_entropy()