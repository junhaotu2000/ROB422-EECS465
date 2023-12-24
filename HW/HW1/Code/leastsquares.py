#---------------------------
# leastsquares --- Junhao TU/9.15#
#---------------------------
import numpy as np
import matplotlib.pyplot as plt

# Given data
data = data = np.loadtxt('/home/rob502/Rob-422/HW1/calibration.txt') # You need to change to your dirct
commanded_positions = data[:, 0]
measured_positions = data[:, 1]

# Set up the design matrix A and the output vector b
A = np.vstack([commanded_positions, np.ones_like(commanded_positions)]).T
b = measured_positions

# Compute the pseudo-inverse of A and use it to compute the parameters of the line
params = np.linalg.pinv(A).dot(b)
slope, intercept = params

# Compute the sum of squared errors
errors = measured_positions - (slope * commanded_positions + intercept)
sse = np.sum(errors**2)

print(f"Parameters of the line: Slope = {slope:.4f}, Intercept = {intercept:.4f}")
print(f"Sum of squared errors: {sse:.4f}")

# Plot the data and the fitted line
plt.scatter(commanded_positions, measured_positions, color='blue', marker='x', label='Data')
plt.plot(commanded_positions, slope * commanded_positions + intercept, color='red', label=f'Fitted Line: y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('Commanded Position')
plt.ylabel('Measured Position')
plt.legend()
plt.title('Least-Squares Fit of a Line to the Data')
plt.savefig('least_squares_plot.png')
plt.show()

