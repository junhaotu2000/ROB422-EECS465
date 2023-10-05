#---------------------------------------------------
# newtonsmethod--- Junhao TU/10.3#
#---------------------------------------------------
from backtracking import bls

def newton_method(f, x, grad_f, hessian_f):
    """
    Perform Newton's method.
    
    Parameters:
    - f: function to be minimized
    - x: current point
    - grad_f: gradient of the function
    - hessian_f: second derivative of the function
    
    Returns:
    - value: list of x values from each iteration
    """
    value = [x]

    while True:
        delta_x = -grad_f(x)/hessian_f(x)
        lambda_square = grad_f(x) * (grad_f(x) / hessian_f(x))
        if (lambda_square/2) <= 0.0001:
            break
        
        t = bls(f, x, grad_f, delta_x)
        x = x + t * delta_x
        value.append(x)

    return value
