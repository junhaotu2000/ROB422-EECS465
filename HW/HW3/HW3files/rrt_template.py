import numpy as np
from utils import load_env, get_collision_fn_PR2, execute_trajectory,  draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, wait_if_gui, joint_from_name, get_joint_positions, set_joint_positions, get_joint_info, get_link_pose, link_from_name
import random
### YOUR IMPORTS HERE ###

class Node:
    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent

def sample_configuration(joint_limits):
    return tuple(random.uniform(joint_limits[jn][0], joint_limits[jn][1]) for jn in joint_names)

def interpolate(start, end, step_size):
    dist = np.linalg.norm(np.array(end) - np.array(start))
    if dist == 0:
        return []
    direction = (np.array(end) - np.array(start)) / dist
    num_steps = int(dist / step_size)
    return [tuple(np.array(start) + direction * step_size * i) for i in range(1, num_steps + 1)]

def nearest(tree, config):
    return min(tree, key=lambda node: np.linalg.norm(np.array(node.config) - np.array(config)))

def is_close(config1, config2, threshold=0.05):
    return np.linalg.norm(np.array(config1) - np.array(config2)) < threshold

def pathtrace(node):
    path = []
    while node is not None:
        path.append(node.config)
        node = node.parent
    path.reverse()
    return path

def rrt(start_config, goal_config, joint_limits, collision_fn, step_size=0.05, max_iterations=1000, goal_bias=0.1):
    start_node = Node(start_config)
    tree = [start_node]
    
    for _ in range(max_iterations):
        if random.random() < goal_bias:
            sample = goal_config
        else:
            sample = sample_configuration(joint_limits)

        nearest_node = nearest(tree, sample)
        path_to_sample = interpolate(nearest_node.config, sample, step_size)

        if path_to_sample and not collision_fn(path_to_sample[0]):
            new_node = Node(path_to_sample[0], nearest_node)
            tree.append(new_node)

            if not collision_fn(new_node.config):
                path_to_goal = interpolate(new_node.config, goal_config, step_size)
                if all(not collision_fn(c) for c in path_to_goal):
                    for config in path_to_goal:
                        new_node = Node(config, new_node)
                        tree.append(new_node)
                    return pathtrace(new_node)
    return []

def shortcut_smoothing(path, collision_fn, num_iterations=150):
    for _ in range(num_iterations):
        if len(path) < 2:  # Need at least two points to perform shortcut
            break
        
        i, j = sorted(random.sample(range(len(path)), 2))
        if j == i + 1:  # They are consecutive, no shortcut possible
            continue
            
        direct_path = interpolate(path[i], path[j], 0.05)
        if all(not collision_fn(config) for config in direct_path):
            path = path[:i+1] + direct_path + path[j+1:]
            
    return path

#########################

joint_names =('l_shoulder_pan_joint','l_shoulder_lift_joint','l_elbow_flex_joint','l_upper_arm_roll_joint','l_forearm_roll_joint','l_wrist_flex_joint')

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2table.json')

    # define active DoFs
    joint_names =('l_shoulder_pan_joint','l_shoulder_lift_joint','l_elbow_flex_joint','l_upper_arm_roll_joint','l_forearm_roll_joint','l_wrist_flex_joint')
    joint_idx = [joint_from_name(robots['pr2'], jn) for jn in joint_names]

    # parse active DoF joint limits
    joint_limits = {joint_names[i] : (get_joint_info(robots['pr2'], joint_idx[i]).jointLowerLimit, get_joint_info(robots['pr2'], joint_idx[i]).jointUpperLimit) for i in range(len(joint_idx))}
    collision_fn = get_collision_fn_PR2(robots['pr2'], joint_idx, list(obstacles.values()))
    start_config = tuple(get_joint_positions(robots['pr2'], joint_idx))
    goal_config = (0.5, 0.33, -1.548, 1.557, -1.32, -0.1928)
    path = []
    ### YOUR CODE HERE ###
    # RRT Planning
    print(f"\nRunning RRT algorithm: ")
    path = rrt(start_config, goal_config, joint_limits, collision_fn)
    
    # Drawing end-effector positions for each configuration in the path
    for config in path:
        set_joint_positions(robots['pr2'], joint_idx, config)
        pos, _ = get_link_pose(robots['pr2'], link_from_name(robots['pr2'], 'l_gripper_tool_frame'))
        draw_sphere_marker(pos, 0.02, (1, 0, 0, 1))
        
    if path and is_close(path[-1], goal_config):
        print("Goal reached!")
    else:
        print("Goal not reached.")
           
    # Shortcut smoothing
    print(f"\nRunning shortcut smoothing algorithm: ")
    smoothed_path = shortcut_smoothing(path, collision_fn, num_iterations=150)

    # Drawing end-effector positions for shortcut smoothing path
    for config in smoothed_path:
        set_joint_positions(robots['pr2'], joint_idx, config)
        pos, _ = get_link_pose(robots['pr2'], link_from_name(robots['pr2'], 'l_gripper_tool_frame'))
        draw_sphere_marker(pos, 0.02, (0, 0, 1, 1))  # Blue color 
    
    if smoothed_path and is_close(smoothed_path[-1], goal_config):
        print("Smoothed path reached the goal!")
    else:
        print("Smoothed path did not reach the goal.")
    ######################
    # Execute planned path
    execute_trajectory(robots['pr2'], joint_idx, smoothed_path, sleep=0.1)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()