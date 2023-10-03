#---------------------------------------------------
# gradientdescent --- Junhao TU/10.3#
#---------------------------------------------------
from backtracking import bls

def grad_descent(f, x0, grad_f, alpha = 0.1, beta = 0.6):
    """
    Perform gradient descent.
    
    Parameters:
    - f: function to be minimized
    - x: current point
    - grad_f: gradient of the function
    - alpha, beta: control parameters
    
    Returns:
    - history: list of x values from each iteration
    """
    x = x0
    history = [x]

    while abs(grad_f(x)) > 0.0001:
        delta_x = -grad_f(x)
        t = bls(f, x, grad_f, delta_x, alpha, beta)
        x = x + t * delta_x
        history.append(x)
        
    return history 
