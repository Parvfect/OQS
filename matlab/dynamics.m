

% Hilbert Space Dimensions
n = 10;

% Constants
pi = 3.14;
C = 5e-15;
l = 3e-10;
je = 9e-22;
hbar = 1e-34;
w = 8.16e11;
e = 1.6e-19;
phi_o = hbar/(2*e);
phi_x = 0.5* phi_o;
mu = je/hbar;
alpha = sqrt((4 * pi*pi * hbar)/(phi_o*phi_o*C));
muomega = mu/w ;
cutoff = 20 * w;
epsilon = w/cutoff ;% Cutoff frequency;
gamma = 0.05; % Damping Rate;


adag = diag(ones(n,1), -1) ;% Annihilation Operator;
a = adag.' ;% Creation Operator;

Q = (sqrt((hbar*C*w)/(2)) * (1j)* (adag - a)); % Momentum operator;
phi = (sqrt((hbar)/(2*C*w))*((adag + a))); % Flux operator (analogous to position operator);

% Dimensionless position and momentum operators
X = sqrt((C*w)/hbar) * phi;
P = sqrt((1)/(C*w*hbar)) * Q;
%cphi = muomega * create_cos_phi(X, phi_o, phi_x, alpha);

% Defining Hamiltonian and Lindbladian
H =  (dot(X, X) + dot(P, P)) + (hbar*gamma/2)*get_commutator(X, P);
L = gamma^(0.5) * (X + (1j - epsilon/2) * P);
Ldag = conj(L).';


% Solving for dynamics
t_i = 0;
t_f = 500;
nsteps = 20000;
h = (t_f-t_i)/nsteps;

t = zeros(nsteps+1, n,n);
t(1,:,:) = make_initial_density_matrix(n);
%t = solver(t, @handler, h, H, L, Ldag);
%plot_density_matrix_elements(t);
%plot_trace_purity(t);


s = ones(3,3,3);
s(1,:,:) = s(1,:,:) + s(1,:,:);
s


function result =  RK4step(x, h, f, H, L, Ldag)
    % Runge Kutta step, x is the current state, h is the step size, f is the function to be integrated
    k1 = f(x, H, L, Ldag);
    k2 = f(x+h*k1/2, H, L, Ldag);
    k3 = f(x+h*k2/2, H, L, Ldag);
    k4 = f(x+h*k3, H, L, Ldag);
    result = x+(h/6)*(k1+2*k2+2*k3+k4);
end


function lindblad_part =  LinEm()
    lindblad_part = get_commutator(L);
end

function y =  handler(x, H, L, Ldag)
    hamiltonian_part = (-1j)* (dot(H, x) - dot(x, H));
    lindblad_part_1 = get_commutator(L, dot(x, Ldag));
    lindblad_part_2 = get_commutator(dot(L, x), Ldag);
    y = hamiltonian_part + 0.5*(lindblad_part_1 + lindblad_part_2);
end

function result = get_commutator(A,B)
    result = dot(A,B) - dot(B,A);
end

function density_matrix = make_initial_density_matrix(n)
    density_matrix =  ones(n,n)/n;
end


function sol_arr =  solver(sol_arr, f, h, H, L, Ldag)

    for i = 1:length(sol_arr(2,:,:))
        sol_arr(i) = RK4step(sol_arr(i-1,:,:), h, f, H, L, Ldag);
    end
end

function plot_density_matrix_elements()
    % To be implemented
end

function plot_trace_purity()
    % to be implemented
end
