

"""
convert to matrix diff form and see what the steady state looks like
Dynamics are not too bad if one is assuming that the initial moment of coupling
is within the simulation. Very interesting about the trace and purity though
"""

# Get this working and second order is trivial you just add a lindblad as a higher order contribution

from helper_functions import *


# Hilbert Space Dimensions
n = 14

# Constants
pi = np.pi
C = 5e-15
l = 3e-10
je = 9.99e-22
hbar = 1e-34
w = 8.16e11
e = 1.6e-19
phi_o = hbar/(2*e)
phi_x = 0.5* phi_o
mu = je/hbar
alpha = np.sqrt((4 * pi*pi * hbar)/(phi_o*phi_o*C))
muomega = mu/w # 
cutoff = 20 * w
epsilon = w/cutoff # Cutoff frequency
gamma = 0.05 # Damping Rate

# Operators
adag = create_annihilation_operator(n) # Annihilation operator
a = create_creation_operator(n) # Creation operator

Q = (np.sqrt((hbar*C*w)/(2)) * (1j)* (adag - a)) # Momentum operator
phi = (np.sqrt((hbar)/(2*C*w))*((adag + a))) # Flux operator (analogous to position operator)

# Dimensionless position and momentum operators
X = np.sqrt((C*w)/hbar) * phi
P = np.sqrt((1)/(C*w*hbar)) * Q
cphi = muomega * (create_cos_phi(X, phi_o, phi_x, alpha))

H =  (np.dot(X, X) + np.dot(P, P) - cphi) + (hbar*gamma/2)*get_anti_commutator(X, P)
L = gamma**(0.5) * (X + 0.01*(1j - epsilon/2) * P)
#L = a - adag
Ldag = L.conj().T


def handler(x):
    hamiltonian_part = (-1j)* (np.dot(H, x) - np.dot(x, H))
    lindblad_part_1 = get_commutator(L, np.dot(x, Ldag))
    lindblad_part_2 = get_commutator(np.dot(L, x), Ldag)
    return hamiltonian_part + 0.5*(lindblad_part_1 + lindblad_part_2)

def first_order_equation():
    """ First order equation for steady state """
    hamiltonian_part = -(1j) * (np.kron(H, np.identity(n)) - np.kron(np.identity(n), H)) 
    lindblad_part_1 = np.kron(Ldag, L) 
    lindblad_part_2 = -0.5*(np.kron(np.identity(n), np.dot(Ldag, L)) + np.kron(np.dot(Ldag, L), np.identity(n)))
    return hamiltonian_part + lindblad_part_1 + lindblad_part_2

def validate_steady_state():
    times = np.arange(4, 1000, 50)
    lrho_sum = []

    for t in tqdm(times):
        t_i = 0
        t_f = t
        h = 1e-2
        nsteps = int((t_f-t_i)/h)
        sol = np.zeros((nsteps+1, n,n), dtype=complex)
        sol[0] = make_initial_density_matrix(n)
        
        sol = solver(sol, handler, h)
        lrho_sum.append(np.sum(np.dot(first_order_equation(), sol[-1].reshape(n*n, 1))))

    plt.plot(times, lrho_sum)
    plt.title("Steady State Validation, sum(Lrho) vs Length of Simulation")
    plt.ylabel("sum(Lrho) = dp/dt")
    plt.xlabel("Length of Simulation")
    plt.show()

if __name__ == "__main__":

    validate_steady_state()
    """
    # Setting simulation parameters
    t_i = 0
    t_f = 5000
    h = 1e-2
    nsteps = int((t_f-t_i)/h)
    t = np.zeros((nsteps+1, n,n), dtype=complex)
    t[0] = make_initial_density_matrix(n)
    
    t = solver(t, handler, h)

    # Plotting
    plot_density_matrix_elements(t)
    plot_trace_purity(t)
    plot_diagonal_density_matrix_elements(t)
    plot_offdiagonal_density_matrix_elements(t)
    plot_steady_state_td_2d(t)
    plot_steady_state_td_3d(t)

    steady_state = t[-1]
    print(np.dot(np.kron(L, np.eye(n)), steady_state.reshape(n*n, 1)))
    """