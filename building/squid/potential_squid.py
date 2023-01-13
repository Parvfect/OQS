
# So we are trying to get an idea of what the potential well looks like

from helper_functions import *


# Hilbert Space Dimensions
n = 80

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
mu = je/hbar
alpha = (4 * pi*pi * hbar)/(phi_o*phi_o*C*w)
muomega = mu/w

# Operators
adag = create_annihilation_operator(n) # Annihilation operator
a = create_creation_operator(n) # Creation operator

Q = (np.sqrt((hbar*C*w)/(2)) * (1j)* (adag - a)) # Momentum operator
phi = (np.sqrt((hbar)/(2*C*w))*((adag + a))) # Flux operator (analogous to position operator)

# Dimensionless position and momentum operators
X = np.sqrt((C*w)/hbar) * phi
P = np.sqrt((1)/(C*w*hbar)) * Q
cphi = muomega * create_cos_phi(X, phi_o, phi_x, alpha)

H =  (np.dot(X, X) + np.dot(P, P) - cphi)

def handler(x):
    return (-1j)* (np.dot(H, x) - np.dot(x, H))

# Get eigenvalues of the Hamiltonian
eigenvalues, eigenvectors = np.linalg.eig(H)

plt.plot(eigenvalues)
plt.show()  
"""
if __name__ == "__main__":

    # Setting simulation parameters
    t_i = 0
    t_f = 100
    nsteps = 20000
    h = (t_f-t_i)/nsteps
    t = np.zeros((nsteps+1, n,n), dtype=complex)
    t[0] = make_initial_density_matrix(n)

    t = solver(t, handler, h)

    # Plotting
    plot_density_matrix_elements(t)
    #plot_trace_purity(t)
"""
