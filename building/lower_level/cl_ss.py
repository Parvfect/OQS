
# Maybe try the normal one first aloha
# The CL SS from matlab seems to give purity 1 as well, hmm


from helper_functions import *

# Hilbert Space Dimensions
n = 14
gamma = 0.5 # Damping Rate

# Operators
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
T = 1

L = 0.1*a 
Ldag = np.conjugate(L).T

def first_order_equation():
    """ First order equation for steady state """
    hamiltonian_part = -(1j) * (np.kron(H, np.identity(n)) - np.kron(np.identity(n), H)) 
    lindblad_part_1 = np.kron(Ldag, L) 
    lindblad_part_2 = -0.5*(np.kron(np.identity(n), np.dot(Ldag, L)) + np.kron(np.dot(Ldag, L), np.identity(n)))
    return hamiltonian_part + 0.5*gamma*(lindblad_part_1 + lindblad_part_2)

L = first_order_equation()

if __name__ == "__main__":
    
    sol = null_space(L)
    #sol = steady_state_solver(L)
    print("Solution obtained for n = ", n)
    sol = sol.reshape(n,n)
    print(sol)
    sol = np.array(sol).astype(np.float64)
    diag = np.diag(sol)
    diag = diag[::-1]
    
    #plt.imshow(sol, interpolation='nearest')
    #print(type(sol))
    plt.imshow(np.array(sol).astype(np.float64))
    plt.title("Steady State Density Matrix, n = " + str(n))
    plt.show()
    trace = np.trace(sol)
    purity = np.trace(np.dot(sol, sol))
    print("Trace: ", trace)
    print("Purity: ", purity)
    