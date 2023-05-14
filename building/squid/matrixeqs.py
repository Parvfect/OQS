# Collection of the matrix differential equations for Lrho

# Gamma is in the Lindblad not the LinEm

import numpy as np

def lindblad(L, H):
    n = len(H) # Dimension of the Hilbert Space
    hamiltonian_part = -(1j) * (np.kron(H, np.identity(n)) - np.kron(np.identity(n), H)) 
    lindblad_part_1 = np.kron(Ldag, L) 
    lindblad_part_2 = -0.5*(np.kron(np.identity(n), np.dot(Ldag, L)) + np.kron(np.dot(Ldag, L), np.identity(n)))
    return hamiltonian_part + lindblad_part_1 + lindblad_part_2

def get_commutator(a,b):
    return np.matmul(a,b) - np.matmul(b,a)


class System:
    def __init__(self, H, L, gamma):
        self.H = H
        self.L = L
        self.Ldag = np.conjugate(L).T
        self.gamma = gamma
        self.n = len(H)
    def LinEm(self, x):
        hamiltonian_part = (-1j)* (np.dot(self.H, x) - np.dot(x, self.H))
        lindblad_part_1 = get_commutator(self.L, np.dot(x, self.Ldag))
        lindblad_part_2 = get_commutator(np.dot(self.L, x), self.Ldag)
        return hamiltonian_part + 0.5*self.gamma*(lindblad_part_1 + lindblad_part_2)    



