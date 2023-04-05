
% Trying to convert a matrice to nullspace to make sure I have a function
% that does the steady state for me

% Generate random sequence, right perfect rand creates a matrice
a = rand(20);

% Get the nullspace
t = null(a);

adag = diag(ones(n,1), -1)

function result = first_order(H, L, Ldag)
    hamiltonian_part = -(1j) * (kron(H, eye(n)) - kron(eye(n), H));
    lindblad_part_1 = np.kron(Ldag, L);
    lindblad_part_2 = -0.5*(np.kron(eye(n), np.dot(Ldag, L)) + np.kron(np.dot(Ldag, L), eye(n)));
    result =  hamiltonian_part + lindblad_part_1 + lindblad_part_2;
end

