#---------------------------
# pwleastsquares --- Junhao TU/9.15#
#---------------------------
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('/home/rob502/Rob-422/HW1/calibration.txt') # You need to change to your dirct
commanded_positions = data[:, 0]
measured_positions = data[:, 1]

# Set up the design matrix A and the output vector b
num_data = len(commanded_positions)
A = np.zeros((num_data, 6))
b = measured_positions

for i, x in enumerate(commanded_positions):
    if x < -0.5:
        A[i, :2] = [x, 1]
    elif x <= 0.5:
        A[i, 2:4] = [x, 1]
    else:
        A[i, 4:] = [x, 1]

# Solve for the parameters
params = np.linalg.lstsq(A, b, rcond=None)[0]
m1, c1, m2, c2, m3, c3 = params

# Compute the fitted values
fitted_positions = A.dot(params)

# Compute the sum of squared errors
errors = measured_positions - fitted_positions
sse = np.sum(errors**2)

# Predict the measured position for a command of 0.68
command = 0.68
if command < -0.5:
    prediction = m1 * command + c1
elif command <= 0.5:
    prediction = m2 * command + c2
else:
    prediction = m3 * command + c3

print(f"Parameters: m1 = {m1:.4f}, c1 = {c1:.4f}, m2 = {m2:.4f}, c2 = {c2:.4f}, m3 = {m3:.4f}, c3 = {c3:.4f}")
print(f"Sum of squared errors: {sse:.4f}")
print(f"Prediction for command 0.68: {prediction:.4f}")

# Plot the data and the fitted lines
plt.scatter(commanded_positions, measured_positions, color='blue', marker='x', label='Data')
plt.plot(commanded_positions, fitted_positions, color='red', label='Piece-wise Linear Fit')
plt.xlabel('Commanded Position')
plt.ylabel('Measured Position')
plt.legend()
plt.title('Piece-wise Linear Least-Squares Fit')
plt.savefig('pw_least_squares_plot.png')
plt.show()


