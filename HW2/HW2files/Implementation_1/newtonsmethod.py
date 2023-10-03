#---------------------------------------------------
# newtonsmethod--- Junhao TU/10.3#
#---------------------------------------------------
from backtracking import bls

def newton_method(f, x0, grad_f, hessian_f, alpha = 0.1, beta = 0.6):
    """
    Perform Newton's method.
    
    Parameters:
    - f: function to be minimized
    - x: current point
    - grad_f: gradient of the function
    - hessian_f: second derivative of the function
    - alpha, beta: control parameters
    
    Returns:
    - history: list of x values from each iteration
    """
    x = x0
    history = [x]

    while True:
        delta_x = -grad_f(x)/hessian_f(x)
        lambda_square = grad_f(x) * (grad_f(x) / hessian_f(x))
        if (lambda_square/2) <= 0.0001:
            break
        
        t = bls(f, x, grad_f, delta_x, alpha, beta)
        x = x + t * delta_x
        history.append(x)

    return history
