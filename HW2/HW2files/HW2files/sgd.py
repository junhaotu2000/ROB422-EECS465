#---------------------------------------------------
# sgd --- Junhao TU/10.3#
#---------------------------------------------------

import numpy as np
import random

def sgd(x0, maxi, grad_f, t, iteration_limit):
    """
    Perform Stochastic Gradient Descent.
    
    Parameters:
    - f: function to be minimized
    - x0: current point
    - maxi: the number of functions
    - grad_f: gradient of the function
    - t: fixed step size
    - iteration_limit: the max number of iterations
    
    Returns:
    - value: list of x values from each iteration
    """
    x = x0
    value = [x]

    for _ in range(iteration_limit):
        i = random.randint(0, maxi-1)
        delta_xi = -grad_f(x, i)

        x = x + t * delta_xi
        value.append(x)
    return value


















