
import numpy as np
from helper_functions import *

def hal(n, gamma=0.05):
    n = 10 # Hilbert Space Dimension
    # Hamiltonian
    q = create_position_operator(n)
    p = create_momentum_operator(n)
    H = (np.dot(p,p) + np.dot(q,q)) + gamma/2 * get_anti_commutator(q,p)

    # Initial Density Matrix
    T = 1
    L = q + 1/T*(1j)*p
    return H, L, gamma


def LinEm(x):
    hamiltonian_part = (-1j)* (np.dot(H, x) - np.dot(x, H))
    lindblad_part_1 = get_commutator(L, np.dot(x, Ldag))
    lindblad_part_2 = get_commutator(np.dot(L, x), Ldag)
    
    return hamiltonian_part + 0.5*gamma*(lindblad_part_1 + lindblad_part_2)    


def validate_steady_state():
    times = np.arange(4, 800, 50)
    lrho_sum = []

    for t in tqdm(times):
        t_i = 0
        t_f = t
        h = 1e-2
        nsteps = int((t_f-t_i)/h)
        sol = np.zeros((nsteps+1, n,n), dtype=complex)
        sol[0] = make_initial_density_matrix(n)
        
        sol = solver(sol, LinEm, h)
        lrho_sum.append(np.sum(np.dot(np.kron(L, np.eye(n)), sol[-1].reshape(n*n, 1))))

    plt.plot(times, lrho_sum)
    plt.show()

if __name__ == "__main__":
    
    H, L, gamma = hal(10)
    t = run_simulation(10, H, L, gamma)