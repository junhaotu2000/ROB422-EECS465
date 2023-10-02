#---------------------------
# HW1_q5 --- Junhao TU/9.14#
#---------------------------
import numpy as np

# System a
# For system of Ax = b, it define A and b matrix sperately
A1 = np.array([[0, 0, -1],[4, 1, 1],[-2, 2, 1]])
b1 = np.array([3, 1, 1])
# Using numpy built-in function to solve linear algera eqaution
try: 
    x1 = np.linalg.solve(A1, b1)
    print("Solution for system a:", x1)
except np.linalg.LinAlgError:# np.linalg.LinAlgError is error output of numpy for linear algbera
    print("System a has no solution or infintely many solutions")

# System b
A2 = np.array([[0, -2, 6],[-4, -2, -2],[2, 1, 1]])
b2 = np.array([1, -2, 0])
try: 
    x2 = np.linalg.solve(A2, b2)
    print("Solution for system b:", x2)
except np.linalg.LinAlgError:
    print("System b has no solution or infintely many solutions")

# System a
A3 = np.array([[2, -2],[-4, 3]])
b3 = np.array([3, -2])
try: 
    x3 = np.linalg.solve(A3, b3)
    print("Solution for system c:", x3)
except np.linalg.LinAlgError:
    print("System c has no solution or infintely many solutions")
