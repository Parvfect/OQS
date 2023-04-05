
% Got to fix the trace function and then we are golden

n = 4;
a = create_annihilation_operator(n);
adag = a.';

q = (adag + a)/2;
p = (1j)*(adag - a)/2;

H = q*q + p*p;
L = 0.2 * a;
Ldag = conj(L).';

init = make_initial_density_matrix(n);
size(init)
t_i = 0;
t_f = 100;
nsteps = 2000;
h = (t_f - t_i)/nsteps;
t = zeros(n, n, nsteps+1);
t(:,:,1) = init;
t = solver(t, @handler, h, H, L, Ldag);

plot_density_matrix_elements(t)
plot_trace_purity(t)

function result = get_commutator(A,B)
    result = A.*B - B.*A;
end


function trace = get_global_trace(rho)
    for i = 1:5
        disp(rho(:,:,i))
    end
end

function y = handler(x, H, L, Ldag)
    H_part = (-1j)* get_commutator(H, x);
    L_part =  0.5 * (get_commutator(L, (x.*Ldag)) + get_commutator((L.*x), Ldag));
    y = H_part + L_part;
end

function result =  RK4step(x, h, f, H, L, Ldag)
    % Runge Kutta step, x is the current state, h is the step size, f is the function to be integrated
    k1 = f(x, H, L, Ldag);
    k2 = f(x+h*k1/2, H, L, Ldag);
    k3 = f(x+h*k2/2, H, L, Ldag);
    k4 = f(x+h*k3, H, L, Ldag);
    result = x+(h/6)*(k1+2*k2+2*k3+k4);
end

function density_matrix = make_initial_density_matrix(n)
    density_matrix =  ones(n,n)/n;
end


function sol_arr =  solver(sol_arr, f, h, H, L, Ldag)

    for i = 2:length(sol_arr(:,:,2))
        sol_arr(:,:,i) = RK4step(sol_arr(:,:, i-1), h, f, H, L, Ldag);
    end
end

function plot_density_matrix_elements(rho)
    figure
    plot(rho(1, 1,:), rho(2, 3, :), get_global_trace(rho)) 
    legend({'r11','r23','tr(p)'},'Location','southwest')
end

function plot_trace_purity(rho)
    figure
    plot(get_global_trace(rho), get_global_trace(rho.*rho))
    legend({'tr(p)','tr(p2)'})
end

function annihilate = create_annihilation_operator(n)
    annihilate = zeros(n,n);
    for i =1:n-1
        annihilate(i,i+1) = sqrt(i); 
    end
end
