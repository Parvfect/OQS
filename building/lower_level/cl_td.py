
import numpy as np
from helper_functions import *

n = 14 # Hilbert Space Dimension
gamma = 0.05 # Damping Rate

# Annihilation and Creation Operators
a = create_annihilation_operator(n)
adag = create_creation_operator(n)

# Hamiltonian
q = create_position_operator(n)
p = create_momentum_operator(n)
H = (np.dot(p,p) + np.dot(q,q)) + gamma/2 * get_anti_commutator(q,p)
H = np.array(H)

# Initial Density Matrix
w = 2e13
hbar = 1e-34
kb = 1.38e-23
T = 1000

L = np.sqrt(T)*q + np.sqrt(1/T)*(1j)*p
Ldag = np.conjugate(L).T

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
    
    init = make_initial_density_matrix(n)
    t_i = 0
    t_f = 600
    h = 0.01
    nsteps = int((t_f-t_i)/h)

    h = (t_f-t_i)/nsteps
    solRK = np.zeros((nsteps+1,n, n),dtype=complex)
    solRK[0]=init

    # Solving
    solRK = solver(solRK, LinEm, h)

    # Visualising
    plot_density_matrix_elements(solRK, title="CL Model at {} Temperature".format("Low"))
    plot_trace_purity(solRK, title="QHO Thermal Bath with {}states".format(n))
    #plot_diagonal_density_matrix_elements(solRK, title="CL Model at {} Temperature".format("Low"))
    #plot_offdiagonal_density_matrix_elements(solRK, title="CL Model at {} Temperature".format("Low"))
    plot_steady_state_td_2d(solRK, title="CL Model at {} Temperature".format("Low"))
    plot_steady_state_td_3d(solRK)
    