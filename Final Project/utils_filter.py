import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.stats as scistats
import copy

from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, wait_for_user,get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
from pr2_models import *

# -- Read data from file
def read_path_from_file_no_interpolate(file_path):
    path = []
    line_temp = []
    with open(file_path, 'r') as file:
        for line in file:
            if ']' in line:
                line_temp.append(line)
                joint_line = ''.join(line_temp).replace('[', ' ').replace(']', ' ').replace('\n', ' ').split(' ')
                joint_line = np.array([float(num) for num in joint_line if num != ''])
                path.append(joint_line)
                line_temp = []
            else:
                line_temp.append(line)
    return np.array(path) 


def read_path_from_file(file_path):
    path = []
    line_temp = []
    with open(file_path, 'r') as file:
        for line in file:
            if ']' in line:
                line_temp.append(line)
                joint_line = ''.join(line_temp).replace('[', ' ').replace(']', ' ').replace('\n', ' ').split(' ')
                joint_line = np.array([float(num) for num in joint_line if num != ''])
                path.append(joint_line)
                line_temp = []
            else:
                line_temp.append(line)
    path = np.array(path)

    x_before_interpolate = np.linspace(0, path.shape[1] - 1, path.shape[1])
    x_after_interpolate = np.linspace(0, path.shape[1], 300) # extend data point into 300
    path_temp = []
    for item in path:
        path_temp.append(np.interp(x_after_interpolate, x_before_interpolate, np.squeeze(item)))

    return np.array(path_temp) 

# -- Simulate a location sensor with Guassian noise
def location_sensor_measurements(true_state, sensor_noise_covariance):
    measured_positions = np.zeros_like(true_state)
    for i in range(true_state.shape[1]):
        x_true, y_true, theta_true = true_state[:, i]
        
        # Generate noise from multivariate normal distribution
        noise = np.random.multivariate_normal([0, 0, 0], sensor_noise_covariance)
        measured_positions[:, i] = [x_true, y_true, theta_true] + noise
    return measured_positions

# -- Calculate error in rmse
def calculate_rmse(estimated_states, true_states):
    if estimated_states.shape != true_states.shape:
        raise ValueError("The shapes of the estimated and true states must be the same.")
    
    squared_errors = (estimated_states - true_states) ** 2
    mean_squared_errors = squared_errors.mean(axis=0)
    rmse = np.sqrt(mean_squared_errors)
    return rmse

def get_action(path: np.ndarray, t, idx_not_moved) -> np.ndarray:
    '''
    Input: 
    path: path given, shape:(M, 3)
    t: current time step

    Output: 
    u_t: action the robot make for next step (configuration difference)
    '''
    moved = False
    u_t = path[t] - path[t - 1]
    if la.norm(u_t) > 1e-10:
        moved = True
    else:
        idx_not_moved.append(t)
    return u_t, moved

def get_sensor(path: np.ndarray, t, sensor_cov) -> np.ndarray: # maybe can randomly generate a config within the map and plugin
    '''
    Input: 
    path: path given, shape:(M, 3)
    t: current time step

    Output: 
    z_t: sensor reading
    '''
    measured = True # set to always true for now since always taking sensor measurement
    true_config = path[t]
    cov = sensor_cov
    
    noisey_config = np.random.multivariate_normal(true_config, cov)
    return noisey_config, measured

def warp_to_pi(angles):
    angles = np.mod(angles, 2 * np.pi)
    angles[angles > np.pi] -= 2 * np.pi
    return angles

def draw(pos, color: str, radius = 0.05) -> None:
    pos = tuple(pos)
    radius = float(radius)
    if color == 'black':
        color = (0, 0, 0, 1)
    elif color == 'blue':
        color = (0, 0, 1, 1)
    elif color == 'red':
        color = (1, 0, 0, 1)
    draw_sphere_marker(pos, radius, color)

def check_collision_in_path(path, robots, base_joints, obstacles):
    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    path = np.array(path)
    path = path.reshape(-1, 3)
    count = 0
    collision = False
    for pos in path:
        if collision_fn(pos):
            collision = True
            count += 1
    return collision, count

# Function to format list elements
def format_list(lst, width):
    return f"{str(lst):<{width}}"

# Function to format single elements
def format_item(item, width):
    return f"{item:<{width}}"
