#---------------------------------------------------
# plot_descents --- Junhao TU/10.3#
#---------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from gradientdescent import grad_descent
from newtonsmethod import newton_method

# Define f, grad_f, hess_f
def f(x): 
    return np.exp(0.5*x + 1) + np.exp(-0.5*x - 0.5) + 5*x
def grad_f(x):
    return 0.5*np.exp(0.5*x + 1) - 0.5*np.exp(-0.5*x - 0.5) + 5
def hess_f(x):
    return 0.25*np.exp(0.5*x + 1) + 0.25*np.exp(-0.5*x - 0.5)

# Determine data points for each method and given function
x0 = 5
gd_history_x = grad_descent(f, x0, grad_f)
gd_history_y = f(np.transpose(gd_history_x))

nm_history_x = newton_method(f, x0, grad_f, hess_f)
nm_history_y = f(np.transpose(nm_history_x))

x_vals = np.linspace(-10, 10, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 5))
# Plot 1: Objective function and sequences
plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, 'k', label="Objective Function")
plt.plot(gd_history_x, gd_history_y, 'r*-', label="Gradient Descent")
plt.plot(nm_history_x, nm_history_y, 'm*-', label="Newton's Method")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# Plot 2: f(x(i)) vs i
plt.subplot(1, 2, 2)
plt.plot(gd_history_y, 'r', label="Gradient Descent")
plt.plot(nm_history_y, 'm', label="Newton's Method")
plt.xlabel('Iteration i')
plt.ylabel('f(x(i))')
plt.legend()

plt.tight_layout()
plt.show()