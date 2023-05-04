
from helper_functions import *
# should we make each handler class a function that subclasses a base class? 

def run_simulation(n, handler, t_i=0, t_f=1000, h=1e-2):
    # Setting simulation parameters
    
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

