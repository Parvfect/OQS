
% Hilbert Space Dimensions
n = 6;

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

a = create_annihilation_operator(n);
adag = create_creation_operator(n);


Q =(1j)* (adag - a); % Momentum operator;
phi = (adag + a); % Flux operator (analogous to position operator);


% Dimensionless position and momentum operators
X = sqrt((C*w)/hbar) * phi;
P = sqrt((1)/(C*w*hbar)) * Q;
%cphi = muomega * create_cos_phi(X, phi_o, phi_x, alpha);

% Defining Hamiltonian and Lindbladian
H =  (Q.*Q + phi.*phi + (gamma/2)*get_commutator(Q, phi)) - cosphi_taylor(phi, n);
L = gamma^0.5*(phi + 0.1*(1j) * Q);
Ldag = conj(L).';


Lsup = first_order(H, L, Ldag, n);
disp(Lsup)

rho  = steady_soln(Lsup);
rho = rho(:,1);
rho = reshape(rho, n, n);
image(abs(rho), 'CDataMapping','scaled')
colorbar
steady_state_trace = trace(rho)
steady_state_purity = trace(rho.*rho)

function result = exponential_series(x, n)
    t=0.0;
    for i =1:n
        t += x^i/factorial(i);
    end
    result = t;
end 

function result = cosphi_taylor(phi, n)

    result =  (exponential_series(1j*phi, n) + exponential_series(-1j*phi, n))/2;
end

function result = first_order(H, L, Ldag, n)
    hamiltonian_part = -(1j) * (kron(H, eye(n)) - kron(eye(n), H));
    lindblad_part_1 = kron(Ldag, L);
    lindblad_part_2 = -0.5*(kron(eye(n), (Ldag.*L)) + kron((Ldag.*L), eye(n)));
    result =  hamiltonian_part + (lindblad_part_1 + lindblad_part_2);
end

function result = steady_soln(L)
    result = null(L);
end

function result = get_commutator(A,B)
    result = A.*B - B.*A;
end

function result = create_annihilation_operator(n)
    result = zeros(n,n);
    for i = 1:(n-1)
        result(i,i+1) = sqrt(i);
    end
end

function result = create_creation_operator(n)
    result = create_annihilation_operator(n).';
end
