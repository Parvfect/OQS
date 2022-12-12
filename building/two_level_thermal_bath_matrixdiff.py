



# Imports 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot


# Defining Operators

sz = np.array([[1,0],[0,-1]])
sp = np.array([[0,1],[0,0]])
sm = np.array([[0,0],[1,0]])
wo = 10
H = (wo/2) * sz 
gamma = 0.1
nth = 2

# Equation
def LinEm():
    res = -1j* (np.kron(np.eye(2), H) - np.kron(H.T, np.eye(2)))
    res += gamma* (nth+1) * (np.kron(sp.T, sm) - 0.5*(np.kron(np.eye(2), np.dot(sp, sm)) + np.kron(np.dot(sp, sm).T, np.eye(2))))
    res += gamma* (nth) * (np.kron(sm.T, sp) - 0.5*(np.kron(np.eye(2), np.dot(sm, sp)) + np.kron(np.dot(sm, sp).T, np.eye(2))))
    return res

L = LinEm()

def handler(rho):
    return np.dot(L, rho)

# Integration Method
def RK4step(x, h):
    k1 = handler(x)
    k2 = handler(x+h*k1/2)
    k3 = handler(x+h*k2/2)
    k4 = handler(x+h*k3)
    return x+(h/6)*(k1+2*k2+2*k3+k4)

# Simulating

init = np.array([[0.5,0.5],[0.5,0.5]], dtype=complex)

t_i = 0
t_f = 30
nsteps = 750

h = (t_f-t_i)/nsteps


solRK = np.zeros((nsteps+1,2,2),dtype=complex)
solRK[0]=init

for step in range(1,solRK.shape[0]):

    solRK[step] = RK4step(solRK[step-1].flatten("F"),h).reshape(2,2).T


# Visualising

import matplotlib
matplotlib.rcParams.update({'font.size': 16,'font.family':'serif'})
fig, ax = plt.subplots(figsize=(12, 9))

trange = 0.1*np.linspace(t_i,t_f,nsteps+1)
#plt.plot(trange,np.real(solRK[:,0,0]), label = r'$\rho_{11}$')
#plt.plot(trange,np.real(solRK[:,1,1]), label = r'$\rho_{22}$')
plt.plot(trange,np.real(solRK[:,1,1]+solRK[:,0,0]), label = r'$\mathrm{Tr}[\rho]$')
#plt.plot(trange,np.real(solRK[:,0,1]), label = r'$\mathrm{Re}[\rho_{12}]$')
#plt.plot(trange,np.imag(solRK[:,0,1]), label = r'$\mathrm{Im}[\rho_{12}]$')


purity = [np.real(np.dot(rho,rho)[1,1] + np.dot(rho, rho)[0,0]) for rho in solRK]
plt.plot(trange,purity, label = r'$\mathrm{Tr}[\rho^2]$')

plt.xlabel('$\gamma t$')
#plt.ylim(-0.5, 1.1)
plt.legend(loc="lower right", numpoints=1,frameon=True)
plt.title("Variation of Trace and Purity for Two level atom coupled to Thermal Bath")

plt.show()

