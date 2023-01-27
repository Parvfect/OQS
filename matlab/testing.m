
% Calling functions - https://uk.mathworks.com/matlabcentral/answers/328959-how-to-call-functions-from-another-m-file

function result = get_function_of_operator(f, op)
    """ Return the function of the operator using Sylvester's formula """
    
    % Get eigenvalues of operator
    eigs = eig(op);

    % Get Frobenius covariants of operators
    covs = [];

    for i = 1:length(eigs)+1
        eig = eigs[i]
        cov = ones((length(eigs), length(eigs)), dtype=complex)
        remaining = [i for i in eigs if i != eig]
        for i in remaining:
            cov *= (op - np.identity(len(eigs))*i)/(eig - i)
        covs.append(cov)

    result = np.zeros((len(eigs), len(eigs)), dtype=complex)
    for i in range(0, len(eigs)):
        result += f(eigs[i])*covs[i]
    end
end

def create_cos_phi(phi, phi_o, phi_x, alpha): 
    """
    Create a cos(phi) operator for the n-th mode     
    """
    cos_const = np.cos((2*pi)*(phi_x/phi_o))
    sin_const = np.sin((2*pi)*(phi_x/phi_o))
    cos_phi = get_function_of_operator(lambda x: np.cos(x), alpha*phi)
    sin_phi = get_function_of_operator(lambda x: np.sin(x), alpha*phi)
    return cos_const*cos_phi - sin_const*sin_phi
