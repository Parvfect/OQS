

from helper_functions import *


# Hilbert Space Dimensions
n = 14

# Constants
pi = np.pi
C = 5e-15
l = 3e-10
je = 9e-22
hbar = 1e-34
w = 8.16e11
e = 1.6e-19
phi_o = hbar/(2*e)
phi_x = 0.5* phi_o
flux_ratio = 0.5
mu = je/hbar
alpha = np.sqrt((4 * pi*pi * hbar)/(phi_o*phi_o*C))
muomega = mu/w # 
cutoff = 20 * w
epsilon = w/cutoff # Cutoff frequency
gamma = 0.05 # Damping Rate

Ic = 2*pi*hbar*mu/phi_o
beta = 2*pi * l*Ic//phi_o
sinphi_const = np.sqrt(beta* w/mu)

# Operators
adag = create_annihilation_operator(n) # Annihilation operator
a = create_creation_operator(n) # Creation operator

Q = (np.sqrt((hbar*C*w)/(2)) * (1j)* (adag - a)) # Momentum operator
phi = (np.sqrt((hbar)/(2*C*w))*((adag + a))) # Flux operator (analogous to position operator)

# Dimensionless position and momentum operators
X = np.sqrt((C*w)/hbar) * phi
P = np.sqrt((1)/(C*w*hbar)) * Q
cphi = cosphi_taylor(X, 20)

X = create_position_operator(n)
P = create_momentum_operator(n)
cphi = cosphi_taylor(X, 20)

H =  (np.dot(X, X) + np.dot(P, P) - cphi) + (gamma/2)*get_anti_commutator(X, P) + gamma* np.sqrt(beta*epsilon*mu/w) * sinphi_taylor(X, 20) 

L_1 = gamma**(0.5) * (X + (1j) * P)  

L_2 = gamma**(0.5) * (X + 50*(1j - epsilon/2)* sinphi_taylor(X, 20))

L_1dag = L_1.conj().T
L_2dag = L_2.conj().T

def handler(x):
    hamiltonian_part = (-1j)* (np.dot(H, x) - np.dot(x, H))
    lindblad_part_1 = get_commutator(L_1, np.dot(x, L_1dag))
    lindblad_part_2 = get_commutator(np.dot(L_1, x), L_1dag)  
    lindblad_part_3 = get_commutator(L_2, np.dot(x, L_2dag))
    lindblad_part_4 = get_commutator(np.dot(L_1, x), L_2dag)
  
    return hamiltonian_part + 0.5*(lindblad_part_1 + lindblad_part_2 + lindblad_part_3 + lindblad_part_4)

if __name__ == "__main__":

    # Setting simulation parameters
    t_i = 0
    t_f = 200
    h = 1e-2
    nsteps = int((t_f - t_i)/h)
    
    t = np.zeros((nsteps+1, n,n), dtype=complex)
    t[0] = make_initial_density_matrix(n)
    
    t = solver(t, handler, h)

    # Plotting
    plot_density_matrix_elements(t, title=f"{n} state SQUID TD simulation")
    
    plot_trace_purity(t)

    # Plotting the steady state
    plot_steady_state_td_2d(t, title=f"{n} state SQUID TD simulation steady state")
    
    print("Steady state purity = {}".format(np.abs(np.trace(np.dot(t[-1], t[-1])))))
