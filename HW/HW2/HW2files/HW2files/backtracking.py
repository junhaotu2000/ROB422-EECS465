#---------------------------------------------------
# backtracking line search (BLS) --- Junhao TU/10.3#
#---------------------------------------------------

def bls(f, x, grad_f, delta_x, alpha = 0.1, beta = 0.6):
    """
    Perform backtracking line search.
    
    Parameters:
    - f: function to be minimized
    - x: current point
    - grad_f: gradient of the function
    - delta_x: descent direction
    - alpha, beta: control parameters
    
    Returns:
    - t: step size
    """
    t = 1.0
    while f(x + t * delta_x) > (f(x) + alpha * t * grad_f(x) * delta_x):
        t = beta * t
    return t

