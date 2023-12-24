#---------------------------
# HW1_q7 --- Junhao TU/9.14#
#---------------------------
import numpy as np
import math

a = np.array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]])
b = np.array([[math.sqrt(2)/2, 0, math.sqrt(2)/2],[0, 1, 0],[-math.sqrt(2)/2, 0, math.sqrt(2)/2]])
c = np.array([[1/2, -math.sqrt(3)/2, 0],[math.sqrt(3)/2, 1/2, 0],[0, 0, 1]])
d = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

e = np.dot(a,b)
f = np.dot(e,c)
g = np.dot(f,d)

print(a)
print(b)
print(c)
print(d)

print(g)