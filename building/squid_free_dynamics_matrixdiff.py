
# Convert to matrix diff form

""" Time Dependent solver for the SQUID model """

import numpy as np
import matplotlib.pyplot as plt

""" Operator creation functions """
def create_annihilation_operator(n):
    """ Create an annihilation operator for the n-th mode """
    return np.matrix(np.diag(np.sqrt(np.arange(1, n)), -1), dtype=complex)

def create_creation_operator(n):
    """ Create a creation operator for the n-th mode """
    return np.matrix(np.diag(np.sqrt(np.arange(1, n)), 1), dtype=complex)

def get_function_of_operator(f, op):
    """ Return the function of the operator using Sylvester's formula """
    
    # Get eigenvalues of operator
    eigs = np.linalg.eig(op)[0]

    # Get Frobenius covariants of operators
    covs = []

    for eig in eigs:
        cov = np.ones((len(eigs), len(eigs)), dtype=complex)
        remaining = [i for i in eigs if i != eig]
        for i in remaining:
            cov *= (op - np.identity(len(eigs))*i)/(eig - i)
        covs.append(cov)

    result = np.zeros((len(eigs), len(eigs)), dtype=complex)
    for i in range(0, len(eigs)):
        result += f(eigs[i])*covs[i]
    
    return result

def create_cos_phi(n, phi, phi_o, phi_x):
    """ Create a cos(phi) operator for the n-th mode """
    cos_phi_1 = (2*3.14/phi_o)*phi - 2*3.14*(phi_x/phi_o)*np.identity(n)
    return get_function_of_operator(lambda x: np.cos(x), phi)

def make_initial_density_matrix(n):
  return np.ones((n,n), dtype=complex)/n


n = 3
adag = create_annihilation_operator(n) # Annihilation operator
a = create_creation_operator(n) # Creation operator

phi = 0.2 * (adag + a) # Flux operator (analogous to position operator)
Q = 0.5 * (1j)* (adag - a) # Momentum operator
cphi = 0.5*create_cos_phi(n, phi, 1, 1)

H = np.dot(Q,Q) + np.dot(phi, phi) + cphi
init = make_initial_density_matrix(n)
# Setting simulation parameters
t_i = 0
t_f = 100
nsteps = 20000
h = (t_f-t_i)/nsteps
t = np.zeros((nsteps+1, n,n), dtype=complex)
t[0] = init

def LinEm():
    return -(1j) * (np.kron(H.T, np.identity(n)) - np.kron(np.identity(n), H)) 

L = LinEm()

def handler(x):
    return np.dot(LinEm(), x)

def RK4step(x, h):
    k1 = handler(x)
    #k2 = handler(x+h*k1/2)
    #k3 = handler(x+h*k2/2)
    #k4 = handler(x+h*k3)
    return x*k1#+(h/6)*(k1+2*k2+2*k3+k4)

for i in range(1,nsteps):
    rho = t[i-1]
    rho = np.ravel(rho, order="F")
   # print(rho.shape)
    t[i] = RK4step(rho, h).reshape(n,n).T

trange = 0.1*np.linspace(t_i,t_f,nsteps+1)
plt.plot(trange, t[:, 0,1], )

trace = [np.real(i[0,0] + i[1,1]+ i[2,2]) for i in t]
#plt.plot(trace)
#plt.plot(trange,np.real(t[:,1,1]), label = r'$\rho_{11}$')


#plt.plot(trange,np.real(t[:,1,1]+solRK[:,0,0]), label = r'$\mathrm{Tr}[\rho]$')
plt.plot(trange,np.real(t[:,0,1]), label = r'$\mathrm{Re}[\rho_{01}]$')
#plt.plot(trange,np.imag(t[:,0,1]), label = r'$\mathrm{Im}[\rho_{01}]$')
plt.title("SQUID free dynamics")
plt.legend()
plt.show()