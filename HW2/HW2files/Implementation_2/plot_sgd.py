#---------------------------------------------------
# plot_sgd --- Junhao TU/10.3#
#---------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sgd import sgd

# Directly import fi, fiprime and fsum from SGDtest may lead to cross import error
# Therefore, they are defined here again 
maxi = 10000 
def fi(x,i):
    coef1 = 0.01 + (0.5-0.01)*i/maxi
    coef2 = 1 + (6-1)*i/maxi
    return (np.exp(coef1*x + 0.1) + np.exp(-coef1*x - 0.5) - coef2*x)/(maxi/100)

def fiprime(x,i):
    coef1 = 0.01 + (0.5-0.01)*i/maxi
    coef2 = 1 + (6-1)*i/maxi
    return (coef1*np.exp(coef1*x + 0.1) -coef1*np.exp(-coef1*x - 0.5) - coef2)/(maxi/100)

def fsum(x):
    sum = 0
    for i in range(0,maxi):
       sum = sum + fi(x,i)
    return sum

sgd_history_x = sgd(-5, maxi, fiprime, 1, 1000)
sgd_history_y = [fsum(x) for x in sgd_history_x]

plt.close('all')
plt.figure(figsize=(7, 5))
plt.plot(range(len(sgd_history_x)), sgd_history_y, 'k-', label="fsum(x(i)) vs i")
plt.xlabel('Iteration i')
plt.ylabel('fsum(x(i))')
plt.legend()

plt.tight_layout()
plt.show()

