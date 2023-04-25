
# Needs fixing

import numpy as np
from helper_functions import *
from tqdm import tqdm


n = 14 # Hilbert Space Dimension
gamma = 0.05# Damping Rate

# Annihilation and Creation Operators
adag = create_annihilation_operator(n)
a = create_creation_operator(n)

# Hamiltonian
q = (adag + a)/2
p = 1j*(adag - a)/2
H = (np.dot(p,p) + np.dot(q,q)) + gamma/2 * get_anti_commutator(q,p)
H = np.array(H)

# Initial Density Matrix
w = 2e13
hbar = 1e-34
kb = 1.38e-23
T = 1000

L = q + (1/T)*(1j)*p
Ldag = np.conjugate(L).T



def LinEm(x, L, Ldag):
    hamiltonian_part = (-1j)* (np.dot(H, x) - np.dot(x, H))
    lindblad_part_1 = get_commutator(L, np.dot(x, Ldag))
    lindblad_part_2 = get_commutator(np.dot(L, x), Ldag)
    
    return hamiltonian_part + 0.5*gamma*(lindblad_part_1 + lindblad_part_2)    

def RK4step(x, h, f, L, Ldag):
    """ Runge Kutta step, x is the current state, h is the step size, f is the function to be integrated """
    k1 = f(x, L, Ldag)
    k2 = f(x+h*k1/2, L, Ldag)
    k3 = f(x+h*k2/2, L, Ldag)
    k4 = f(x+h*k3, L, Ldag)

    return x+(h/6)*(k1+2*k2+2*k3+k4)

def solver(sol_arr, f, h, L, Ldag):

    for i in range(1, sol_arr.shape[0]):
        sol_arr[i] = RK4step(sol_arr[i-1], h, f, L, Ldag)

    return sol_arr


def temp_analysis():
    T = np.arange(1, 400, 5)
    steady_state_purity = np.zeros(len(T))
    steady_state_trace = np.zeros(len(T))
    for i in tqdm(range(len(T))):
        L = q + (1/T[i])*(1j)*p
        Ldag = np.conjugate(L).T
        init = make_initial_density_matrix(n)
        t_i = 0
        t_f = 500
        nsteps = 10000
        h = (t_f-t_i)/nsteps
        solRK = np.zeros((nsteps+1,n, n),dtype=complex)
        solRK[0]=init

        # Solving
        solRK = solver(solRK, LinEm, h, L, Ldag)
        #plot_density_matrix_elements(solRK)
        steady_state_purity[i] = np.trace(np.dot(solRK[-1], solRK[-1]))
        steady_state_trace[i] = np.trace(solRK[-1])
        
    plt.plot(T, steady_state_purity)
    plt.title("Steady State Purity vs Temperature for Calderia Leggett Model")
    plt.xlabel("Temperature")
    plt.ylabel("Steady State Purity")
    plt.show()

    plt.plot(T, steady_state_trace)
    plt.title("Steady State Trace vs Temperature for Calderia Leggett Model")
    plt.xlabel("Temperature")
    plt.ylabel("Steady State Trace")
    plt.show()

if __name__ == "__main__":
    temp_analysis()