#---------------------------------------------------
# plot_descents --- Junhao TU/10.3#
#---------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from gradientdescent import grad_descent
from newtonsmethod import Newton_method

def f(x):
    return np.exp(0.5*x + 1) + np.exp(-0.5*x - 0.5) + 5*x

xvals = np.arange(-10, 10, 0.01)
yvals = f(xvals)


x0 = 5
x_grad_descent = grad_descent(f, x0)
y_grad_descent = f(x_grad_descent)

x_Newton_method = Newton_method(f, x0)
y_Newton_method = f(x_Newton_method)


# Plot 1: Objective function and sequences of points
plt.figure(figsize=(10, 6))
plt.plot(xvals, yvals, 'k', label='Objective function')
plt.plot(x_grad_descent, y_grad_descent, 'ro-', label='Gradient Descent')
plt.plot(x_Newton_method, y_Newton_method, 'mo-', label='Newton Method')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Objective function and sequences of points')
plt.grid(True)
plt.show()

# Plot 2: f(x(i)) vs. i
plt.figure(figsize=(10, 6))
plt.plot(y_grad_descent, 'r-', label='Gradient Descent')
plt.plot(y_Newton_method, 'm-', label='Newton Method')
plt.xlabel('Iteration i')
plt.ylabel('f(x(i))')
plt.legend()
plt.title('f(x(i)) vs. i for Gradient Descent and Newton Method')
plt.grid(True)
plt.show()


