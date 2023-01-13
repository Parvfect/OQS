
import matplotlib.pyplot as plt
import numpy as np


def qho_potential(x):
    return 0.5 * x**2

def squid_potential(x):
    return 0.5 * x**2 - 3 * np.cos(x+3.14)


t = np.linspace(-10, 10, 1000)
plt.plot(t, qho_potential(t), label="QHO")
plt.plot(t, squid_potential(t), label="Squid")
plt.legend()    
plt.xlabel("x")
plt.ylabel("V(x)")
plt.title("Form of Potential for QHO and Squid")
plt.show()