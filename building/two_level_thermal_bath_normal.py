

# Imports 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot


# Defining Operators

sz = np.array([[1,0],[0,-1]])
sp = np.array([[0,1],[0,0]])
sm = np.array([[0,0],[1,0]])

# Equation
def LinEm(x, omega0=10, nth=2, gamma = 0.1):
    res = -1j*(omega0/2)*(np.dot(sz,x)-np.dot(x,sz))
    res += gamma*(nth+1)*(multi_dot([sm,x,sp])-0.5*multi_dot([sp,sm,x])-0.5*multi_dot([x,sp,sm]))
    res +=     gamma*nth*(multi_dot([sp,x,sm])-0.5*multi_dot([sm,sp,x])-0.5*multi_dot([x,sm,sp]))    
    return res


# Integration Method
def RK4step(x, h):
    k1 = LinEm(x)
    k2 = LinEm(x+h*k1/2)
    k3 = LinEm(x+h*k2/2)
    k4 = LinEm(x+h*k3)
    return x+(h/6)*(k1+2*k2+2*k3+k4)

# Simulating

init = np.array([[0.5,0.5],[0.5,0.5]], dtype=complex)

t_i = 0
t_f = 20
nsteps = 500

h = (t_f-t_i)/nsteps

def RK4step(x, h):
    k1 = LinEm(x)
    k2 = LinEm(x+h*k1/2)
    k3 = LinEm(x+h*k2/2)
    k4 = LinEm(x+h*k3)
    return x+(h/6)*(k1+2*k2+2*k3+k4)

solRK = np.zeros((nsteps+1,2,2),dtype=complex)
solRK[0]=init

for step in range(1,solRK.shape[0]):

    solRK[step] = RK4step(solRK[step-1],h)


# Visualising

import matplotlib
matplotlib.rcParams.update({'font.size': 16,'font.family':'serif'})
fig, ax = plt.subplots(figsize=(12, 9))

trange = 0.1*np.linspace(t_i,t_f,nsteps+1)
#plt.plot(trange,np.real(solRK[:,0,0]), label = r'$\rho_{11}$')
plt.plot(trange,np.real(solRK[:,1,1]), label = r'$\rho_{22}$')
plt.plot(trange,np.real(solRK[:,1,1]+solRK[:,0,0]), label = r'$\mathrm{Tr}[\rho]$')
plt.plot(trange,np.real(solRK[:,0,1]), label = r'$\mathrm{Re}[\rho_{12}]$')
plt.plot(trange,np.imag(solRK[:,0,1]), label = r'$\mathrm{Im}[\rho_{12}]$')

plt.xlabel('$\gamma t$')
plt.ylim(-0.5, 1.1)
plt.legend(loc = "best",numpoints=1,frameon=True)
plt.title("Two Level System with Thermal Bath")

plt.show()

