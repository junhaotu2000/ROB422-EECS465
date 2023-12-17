import numpy as np

################ PR2 robot ################
# A matrix - state matrix
A = np.eye(3)

# B matrix - control matrix
def B(theta):
    B = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return B

def B2(theta):
    B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return B

# C matrix - observation matrix
C = np.eye(3)

# G matrix - Jocobian matrix
def G(state, control_input):
    x, y, theta = state
    dx, dy, dtheta = control_input
    G = np.array([[1, 0, -dx * np.sin(theta) - dy * np.cos(theta)],
                    [0, 1,  dx * np.cos(theta) - dy * np.sin(theta)],
                    [0, 0,  1]])
    return G

# R matrix - motion noise covarience matrix (small covarience between x, y and theta)
R = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]])

# Q matrix - sensor noise covarience matrix (2D lidar sensor + IMU)
Q = np.array([[0.02, 0.001, 0], [0.001, 0.02, 0], [0, 0, 0.02]])

