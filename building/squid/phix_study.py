
from td_alternate import run_simulation
from helper_functions import plot_trace_purity, get_purity
import matplotlib.pyplot as plt


arr = [get_purity(run_simulation(i)) for i in range(1, 5)]

for i in range(len(arr)):
    plt.plot(arr[i], label = r'${}$'.format(i+1))

plt.legend()
plt.title("Studying purity change for different external flux")
plt.show()