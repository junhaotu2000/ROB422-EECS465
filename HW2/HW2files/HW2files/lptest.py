#---------------------------------------------------
# lptest --- Junhao TU/10.3#
#---------------------------------------------------

import cvxpy as cp
import numpy as np

def solve_lp_with_cvxpy():
    # Define the variable
    x = cp.Variable(2)

    # Define the constraints
    A = np.array([[0.7071, 0.7071], 
                  [-0.7071, 0.7071],
                  [0.7071, -0.7071],
                  [-0.7071, -0.7071]])
    b = np.array([1.5, 1.5, 1, 1])

    constraints = [A @ x <= b]

    # Define the objective
    c = np.array([2, 1])
    objective = cp.Minimize(c.T @ x)

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Print the result
    print("The optimal point:", x.value)

if __name__ == "__main__": #用于测试
    solve_lp_with_cvxpy()
