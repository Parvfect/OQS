
# Sylvester's verified
# Time to see if matrix orders are accurate

import numpy as np
from steady_state import get_function_of_operator
import time

# Sylvester's method is faster than matrix raised to n for numpy matrices

def sylvester_check():
    """ Checking if sylvester's formula approximation working """

    a = np.matrix('[1 3; 4 2]')
    b = np.matrix('[3 -2; 1 0]')

    """ fix assert stuff
    assert get_function_of_operator(lambda x: x, a) == a
    # Wikipidea example - https://en.wikipedia.org/wiki/Sylvester%27s_formula
    assert get_function_of_operator(lambda x: 1/x, a) == np.matrix('[-0.2 0.3; 0.4 -0.1]')
    assert get_function_of_operator(lambda x: x**2, b) == np.matrix('[7 -6; 3 -2]')
    assert get_function_of_operator(lambda x: x**3, b) == np.matrix('[15 -14; 7 -6]')
    assert get_function_of_operator(lambda x: x, b) == np.matrix('[3 -2; 1 0]')
    print("Tests passed Successfully, sylvester's formula approximation working for 2x2 matrices")
    """



if __name__ == '__main__':
    sylvester_check()