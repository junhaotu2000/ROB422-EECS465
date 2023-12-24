#---------------------------
# HW1_q5 --- Junhao TU/9.14#
#---------------------------
import numpy as np

#Define matrices A and B
A = np.array([[1, 2],[3, -1]])
B = np.array([[-2, -2],[4, -3]])

#a. A + 2B
result_a = A + 2*B
print("a. A+2B:\n", result_a)

#b. AB and BA
result_b1 = np.dot(A, B)
result_b2 = np.dot(B, A)
print("b. AB:\n", result_b1)
print("b. BA:\n", result_b2)

#c. A', transpose of A
result_c = np.transpose(A)
print("c. A':\n", result_c)

#d. B^2
result_d = np.dot(B, B)
print("d. B^2:\n", result_d)

#e. A'B' and (AB)'
result_e1 = np.dot(np.transpose(A), np.transpose(B))
result_e2 = np.transpose(np.dot(A,B))
print("e. A'B':\n", result_e1)
print("e. (AB)':\n", result_e2)

#f. det(A), determinant of A
result_f = np.linalg.det(A)
print("f. det(A):\n", result_f)

#g. B^(-1), inverse of B
result_g = np.linalg.inv(B)
print("g. B^(-1):\n", result_g)