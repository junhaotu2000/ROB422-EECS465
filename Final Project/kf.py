import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from utils_filter import *
import pybullet as p

class KalmanFilter:
    def __init__(self, A, B_func, C, R, Q, initial_state, initial_covariance):
        self.A = A
        self.B_func = B_func
        self.C = C
        self.R = R
        self.Q = Q
        self.state = initial_state
        self.covariance = initial_covariance
 
    def predict(self, control_input):
        theta = self.state[2]
        self.B = self.B_func(theta)
        self.state = np.dot(self.A, self.state) + np.dot(self.B, control_input)
        self.covariance = np.dot(np.dot(self.A, self.covariance), self.A.T) + self.Q

    def update(self, measurement):   
        K = np.dot(np.dot(self.covariance, self.C.T), 
                   np.linalg.inv(np.dot(np.dot(self.C, self.covariance), self.C.T) + self.R))
        self.state = self.state + np.dot(K, (measurement - np.dot(self.C, self.state)))
        self.covariance = self.covariance - np.dot(np.dot(K, self.C), self.covariance)

    def get_state(self):
        return self.state



def main_kf(path_kf, map_kf):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    
    ############### Change map here ###############
    robots, obstacles = load_env(map_kf)
    
    # change camera view
    p.resetDebugVisualizerCamera(cameraDistance = 5, cameraYaw = 0, cameraPitch = -60, cameraTargetPosition = [0, 0, 0])
    
    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))

    # Read path from recorded data 
    true_state = read_path_from_file(path_kf)
    x_true, y_true, theta_true = true_state

    ################ Kalman Filter ################
    # KF initialization
    from pr2_models import A, B, C, R, Q
    initial_state = true_state[:,0]  # x_0: x, y, θ
    initial_covariance = np.eye(3)  # sigma_0: initial covarience
    kf = KalmanFilter(A, B, C, R, Q, initial_state, initial_covariance)

    # Generate location sensor measurements
    measured_state = location_sensor_measurements(true_state, Q)
    x_measured, y_measured, theta_measured = measured_state

    # Estimate the state of a pr2 robot
    kf_states = []
    kf_states.append(initial_state)
    
    for i in range(1, true_state.shape[1]):
        # control input
        dx = x_true[i] - x_true[i-1]
        dy = y_true[i] - y_true[i-1]
        dtheta = theta_true[i] - theta_true[i-1]
        control_input = np.array([dx, dy, dtheta])

        # prediction step
        kf.predict(control_input) 

        # correction step
        kf.update(measured_state[:, i])  
        kf_states.append(kf.get_state())

    kf_states = np.array(kf_states)

    # Calculate error
    rmse = calculate_rmse(kf_states, true_state.T)
    print(f"KF RMSE: {rmse}")

    # Check collision
    collision, collision_count = check_collision_in_path(kf_states, robots, base_joints, obstacles)
    print(f"Collision count: {collision_count}")

    ################ Visualization ################
    # Set the figure size
    plt.figure(figsize=(8.5, 6))
    plt.xlim(-4,4)
    plt.ylim(-2,2)

    # Plotting the actual, measured, and KF paths
    plt.plot(x_true, y_true, 'b-', label="Ground Truth", linewidth=2) 
    plt.scatter(x_measured, y_measured, color='g', s=10, label="Sensor Data")  
    # plt.plot(kf_states[:, 0], kf_states[:, 1], 'r--', label="KF estimation", linewidth=2)  
    plt.scatter(kf_states[:, 0], kf_states[:, 1], color='r', s=10, label="KF estimation") 

    # Add arrows to show orientation at selected points
    arrow_skip = 30 # Number of points to skip between arrows
    for i in range(0, len(theta_true), arrow_skip):
        plt.arrow(x_true[i], y_true[i], 
                  0.3 * np.cos(theta_true[i]), 0.3 * np.sin(theta_true[i]), 
                  head_width=0.07, head_length=0.15, fc='blue', ec='blue')
        
    # Add arrows to show orientation for KF path
    for i in range(0, kf_states.shape[0], arrow_skip):
        plt.arrow(kf_states[i, 0], kf_states[i, 1], 
                  0.3 * np.cos(kf_states[i, 2]), 0.3 * np.sin(kf_states[i, 2]), 
                  head_width=0.07, head_length=0.15, fc='red', ec='black')

    # Marking start and end points for each path
    # plt.scatter(x_true[0], y_true[0], color='b', marker='o', s=100, label="Start (Actual)", edgecolor='black')
    # plt.scatter(x_true[-1], y_true[-1], color='b', marker='X', s=100, label="End (Actual)", edgecolor='black')
    # plt.scatter(kf_states[0, 0], kf_states[0, 1], color='r', marker='o', s=100, label="Start (EKF)", edgecolor='black')
    # plt.scatter(kf_states[-1, 0], kf_states[-1, 1], color='r', marker='X', s=100, label="End (EKF)", edgecolor='black')

    # Adding labels, title, grid, and legend
    plt.xlabel("X Position", fontsize = 16) 
    plt.ylabel("Y Position", fontsize = 16) 
    plt.title(f'Kalman Filter Path Tracking for {path_kf.replace("_", " ").replace(".txt", "").title()}', fontsize = 16) 
    plt.legend(fontsize = 16) 
    plt.grid(True) 
    # plt.show(block=False) 
    print("Close plot window(s) to continue... Note: Don't close PyBullet GUI!!!") 
    plt.show()
    # wait_for_user()
    # plt.close()
    disconnect()
    return rmse, collision_count
    

if __name__ == '__main__':
    main_kf("minor_obstacle.txt", "minor_obstacle.json")
