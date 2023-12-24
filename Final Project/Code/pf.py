import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.stats as scistats
import copy
from tqdm import tqdm
import pybullet as p

from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, wait_for_user,get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
from pr2_models import *
from utils_filter import *

T_MAX = 300 # iteration for particle filter & len of interpolated path
ACTION_COV = R #np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]) #[0.05, 0.05, 0.05] #[x, y, theta]
SENSOR_COV = Q #np.array([[0.02, 0.001, 0], [0.001, 0.02, 0], [0, 0, 0.02]]) #[0.03, 0.03, 0.03] #[x, y, theta]
NUM_PARTICLES = 300 # number of particles for particle filter
X_MAX = 8 # max x length for map
Y_MAX = 4 # max y length for map
ACTION_ONLY = False
CHECK_COLLISION = False
MAP = "minor_obstacle.json" 
PATH = "minor_obstacle.txt" 
# MAP = "open_space.json"
# PATH = "open_space.txt" 
# MAP = "maze.json"
# PATH = "maze.txt" 
    


class ParticleFilter():
    def __init__(self, num_particles, x_max, y_max, collision_fn, check_collision) -> None:
        self.num_particles = num_particles
        self.particles_t0 = []
        self.particles_tminus1 = []
        self.particles_t = []
        self.samples_t = [] #[[x, y, theta, w],...]
        self.weight_t = []
        self.weight_tminus1 = []
        self.u_t = []
        self.z_t = []
        self.estimated_path = []
        self.collision_fn = collision_fn
        self.check_collision = check_collision
        self.random_init(x_max, y_max)

    def random_init(self, x_max, y_max):
        if not self.check_collision:
            self.particles_t0 = np.random.rand(3, self.num_particles)
            self.particles_t0[0] = (self.particles_t0[0] * 2 - 1) * x_max/2
            self.particles_t0[1] = (self.particles_t0[1] * 2 - 1) * y_max/2
            self.particles_t0[2] = (self.particles_t0[2] * 2 - 1) * np.pi
            self.particles_t0 = self.particles_t0.T
            self.particles_tminus1 = self.particles_t0

        else: 
            particles = []
            while len(particles) < self.num_particles:
                particle_x = np.random.uniform(-x_max/2, x_max/2, 1)
                particle_y = np.random.uniform(-y_max/2, y_max/2, 1)
                particle_theta = np.random.uniform(-np.pi, np.pi, 1)
                if not self.collision_fn([particle_x, particle_y, particle_theta]):
                    particles.append(np.array([particle_x, particle_y, particle_theta]))
            
            self.particles_t0 = np.array(particles).squeeze()
            self.particles_tminus1 = np.array(particles).squeeze()

    def action_model(self, action_cov):
        '''
        Sample x_t_m ~ p(x_t | u_t, x_tminus1)
        Update self.particle_t
        '''
        delta = 5e-5
        mean = self.u_t
        cov_x = np.sqrt(action_cov[0, 0] * np.abs(mean[0]) + delta)
        cov_y = np.sqrt(action_cov[1, 1] * np.abs(mean[1]) + delta)
        cov_theta = np.sqrt(action_cov[2, 2] * np.abs(mean[2]) + delta)
        
        actual_dxs = np.random.normal(mean[0], cov_x, self.num_particles).reshape(self.num_particles, -1)
        actual_dys = np.random.normal(mean[1], cov_y, self.num_particles).reshape(self.num_particles, -1)
        actual_dthetas = np.random.normal(mean[2], cov_theta, self.num_particles).reshape(self.num_particles, -1)
        actual_dxythetas = np.concatenate((actual_dxs, actual_dys, actual_dthetas), axis=1)
        
        particles_t = self.particles_tminus1 + actual_dxythetas
        particles_t.T[2] = warp_to_pi(particles_t.T[2])
        if self.check_collision:
            for i, particle in enumerate(particles_t):
                while self.collision_fn(particle):
                    actual_dx = np.random.normal(mean[0], cov_x, 1)
                    actual_dy = np.random.normal(mean[1], cov_y, 1)
                    actual_dtheta = np.random.normal(mean[2], cov_theta, 1)
                    actual_dxytheta = np.concatenate((actual_dx, actual_dy, actual_dtheta)).squeeze()
                    particle = self.particles_tminus1[i] + actual_dxytheta
                particles_t[i] = particle

        self.particles_t = particles_t
    
    def sensor_model(self, sensor_cov, action_only):
        '''
        w_t_m = p(z_t | x_t_m)
        S_t = S_t U (x_t_m, w_t_m)
        Update self.weight_t
        '''
        w_sum = 0
        self.weight_t = []

        cov = sensor_cov
        p = scistats.multivariate_normal(self.z_t, cov)
        self.weight_t = np.array(self.weight_t)
        for particle in self.particles_t:
            w_t_m = p.pdf(list(particle))
            self.weight_t = np.append(self.weight_t, w_t_m)
            w_sum += w_t_m
        self.weight_t /= w_sum
        
        
    def low_var_resample(self, action_only):
        '''
        Update self.samplt_t, self.particle_tminus1, self.weight_tminus1
        '''
        self.particles_tminus1 = np.zeros((self.num_particles, 3))
        self.samples_t = np.zeros((self.num_particles, 4))
        
        r = np.random.uniform(0, 1.0 / self.num_particles)
        c = self.weight_t[0]
        i = 0
        for m in range(self.num_particles):
            U = r + m * (1 / self.num_particles)
            while U > c and i < self.num_particles - 1:
                i += 1
                c += self.weight_t[i]
            self.samples_t[m,:] = copy.deepcopy(np.concatenate((self.particles_t[i, :].reshape(1,-1), self.weight_t[i].reshape(1,-1)), axis=1))
        self.samples_t[:, 3] /= self.samples_t[:, 3].sum()
        self.particles_tminus1 = copy.deepcopy(self.samples_t[:,:3])
        self.weight_tminus1 = copy.deepcopy(self.samples_t[:,3])
    
    def estimate_config(self):
        x = (self.samples_t[:, 0] * self.samples_t[:, 3]).sum()
        y = (self.samples_t[:, 1] * self.samples_t[:, 3]).sum()
        theta_cos = (np.cos(self.samples_t[:, 2]) * self.samples_t[:, 3]).sum()
        theta_sin = (np.sin(self.samples_t[:, 2]) * self.samples_t[:, 3]).sum()
        theta = np.arctan2(theta_sin, theta_cos)
        
        self.estimated_path.append(np.array([x, y, theta]))

    
def main_pf(path_pf, map_pf):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    
    ############### Change map here ###############
    robots, obstacles = load_env(map_pf)
    
    # change camera view
    p.resetDebugVisualizerCamera(cameraDistance = 5, cameraYaw = 0, cameraPitch = -60, cameraTargetPosition = [0, 0, 0])
    
    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))


    ################ read path ################
    path = []
    line_temp = []
    with open(path_pf, 'r') as file:
        for line in file:
            if ']' in line:
                line_temp.append(line)
                joint_line = ''.join(line_temp).replace('[', ' ').replace(']', ' ').replace('\n', ' ').split(' ')
                joint_line = np.array([float(num) for num in joint_line if num is not ''])
                path.append(joint_line)
                line_temp = []
            else:
                line_temp.append(line)
    path = np.array(path)
    
    # interpolate path
    x_before_interpolate = np.linspace(0, path.shape[1] - 1, path.shape[1])
    x_after_interpolate = np.linspace(0, path.shape[1], T_MAX)
    path_temp = []
    for item in path:
        path_temp.append(np.interp(x_after_interpolate, x_before_interpolate, np.squeeze(item)))
    path = np.array(path_temp).T
    
    
    ################ particle filter ################
    t = 0
    u_cache = []
    z_cache = []
    particles_cache = []
    idx_not_moved = [0]
    pf = ParticleFilter(NUM_PARTICLES, X_MAX, Y_MAX, collision_fn, CHECK_COLLISION)
    moved = False
    measured = False
    
    # while(t < T_MAX):
    for t in tqdm(range(T_MAX - 1)):
        t += 1
        
        # get control input and sensor data
        pf.u_t, moved = get_action(path, t, idx_not_moved)
        u_cache.append(pf.u_t)
        pf.z_t, measured = get_sensor(path, t, SENSOR_COV)
        z_cache.append(pf.z_t)
        
        if moved and measured:
            # reset parameter
            pf.samples_t = []
            pf.particles_t = []
            
            # apply action model
            pf.action_model(ACTION_COV)
            
            # apply sensor model
            pf.sensor_model(SENSOR_COV, ACTION_ONLY)
            
            # apply resampling
            pf.low_var_resample(ACTION_ONLY)
            
            # estimate configuration
            pf.estimate_config()
            
        if t == 0 or t%(T_MAX/10) == 0 or t == T_MAX-2:
            particles_cache.append(pf.particles_t)
            # print(f"Num of iteration: {t}/{T_MAX}")

    # calculate rmse
    path_rmse = copy.deepcopy(path)
    path_rmse = np.delete(path_rmse, idx_not_moved, 0)
    path_estimated_rmse = copy.deepcopy(pf.estimated_path)
    # path_estimated_rmse = np.delete(path_estimated_rmse, 0, 0)
    rmse = calculate_rmse(np.array(path_estimated_rmse), np.array(path_rmse))
    print(f"PF RMSE: {rmse}")
    collision, collision_count = check_collision_in_path(path, robots, base_joints, obstacles)
    print(f"Collision count: {collision_count}")

    ################ plotting ################
    plt.figure(1, figsize=(8, 6))
    plt.xlim(-4,4)
    plt.ylim(-2,2)
    plt.plot(path.T[0], path.T[1], label='Ground Truth', c='b')
    plt.scatter(np.array(z_cache).T[0], np.array(z_cache).T[1], s=10, label='Sensor Data', c='g')
    plt.scatter(np.array(pf.estimated_path).T[0], np.array(pf.estimated_path).T[1], s=10 ,label='PF Estimation', c='r')
    
    # Add arrows to show orientation at selected points
    arrow_skip = T_MAX//10 # Number of points to skip between arrows
    for i in range(0, T_MAX, arrow_skip):   
        plt.arrow(path[i, 0], path[i, 1], 
                  0.3 * np.cos(path[i, 2]), 0.3 * np.sin(path[i, 2]), 
                  head_width=0.07, head_length=0.15, fc='blue', ec='blue')
        
    # Add arrows to show orientation for KF path
    for i in range(0, T_MAX, arrow_skip):
        plt.arrow(pf.estimated_path[i][0], pf.estimated_path[i][1], 
                  0.3 * np.cos(pf.estimated_path[i][2]), 0.3 * np.sin(pf.estimated_path[i][2]), 
                  head_width=0.07, head_length=0.15, fc='red', ec='black')

    plt.legend(fontsize = 16)
    plt.xlabel('X Position', fontsize = 16)
    plt.ylabel('Y Position', fontsize = 16)
    plt.grid(True)
    plt.title(f'Particle Filter Path Tracking for {path_pf.replace("_", " ").replace(".txt", "").title()}', fontsize = 16)
    
    plt.figure(2, figsize=(8, 6))
    plt.xlim(-4,4)
    plt.ylim(-2,2)
    plt.plot(path.T[0], path.T[1], label='Ground Truth', c='b')
    plt.scatter(np.array(z_cache).T[0], np.array(z_cache).T[1], s=10, label='Sensor Data', c='g')
    plt.scatter(np.array(particles_cache).T[0], np.array(particles_cache).T[1], s=10 ,label='PF Particles example', c='r')
    plt.legend(fontsize = 16)
    plt.xlabel('X Position', fontsize = 16)
    plt.ylabel('Y Position', fontsize = 16)
    plt.grid(True)
    plt.title(f'Particle Filter Particle Examples for {path_pf.replace("_", " ").replace(".txt", "").title()}', fontsize = 16)

    # plt.figure(3, figsize=(8, 6))
    # plt.xlim(-4,4)
    # plt.ylim(-2,2)
    # plt.scatter(pf.particles_t0.T[0], pf.particles_t0.T[1], s=5, c='r')

    # plt.show(block=False)
    
    print("Close plot window(s) to continue... Note: Don't close PyBullet GUI!!!") 
    plt.show() 
    # wait_for_user()
    # plt.close()
    # plt.close()
    disconnect()
    return rmse, collision_count


if __name__ == '__main__':
    main_pf(PATH, MAP)
    
    