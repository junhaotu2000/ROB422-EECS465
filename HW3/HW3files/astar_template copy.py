import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
from queue import PriorityQueue

class Node():
    def __init__(self, x_in, y_in, theta_in, parent, gn):
        self.x = x_in
        self.y = y_in
        self.theta = theta_in
        self.parent = parent
        self.gn = gn
    
    def config(self):
        return (self.x, self.y, self.theta)

def action_cost(n, m):
    cost_action = np.sqrt((n.x - m.x)**2 + (n.y - m.y)**2 + min(abs(n.theta - m.theta), 2*np.pi - abs(n.theta - m.theta))**2)
    return cost_action

def heuristic(n, goal):
    result_heuristic = np.sqrt((n.x - goal.x)**2 + (n.y - goal.y)**2 + min(abs(n.theta - goal.theta), 2*np.pi - abs(n.theta - goal.theta))**2)
    return result_heuristic

def boundingangle(theta):
    if theta > np.pi:
        theta = theta - 2 * np.pi
    elif theta < -np.pi:
        theta = theta + 2 * np.pi
    return theta

def neighbor_connected(node, mode):
    neighbor = []
    if mode == '4-connected':
      if mode == '4-connected':
         for dx, dy, dtheta in [[-0.1, 0, 0],
                 [0.2, 0, 0],
                 [0, 0.1, 0],
                 [0, -0.1, 0],
                 [0, 0, -np.pi/2],
                 [0, 0, np.pi/2]]:
            neighbor_node = Node(node.x + dx, node.y + dy, boundingangle(node.theta + dtheta), node, -1)
            neighbor.append(neighbor_node)
    
    elif mode == '8-connected':
      for dx in [-0.1, 0, 0.1]:
            for dy in [-0.1, 0, 0.1]:
                for dtheta in [-np.pi/2, 0, np.pi/2]:
                    if not(dx == 0 and dy == 0 and dtheta == 0):
                        neighbor_node = Node(node.x + dx, node.y + dy, boundingangle(node.theta + dtheta), node, -1)
                        neighbor.append(neighbor_node)

    return neighbor

def pathtrace(node):
    path = []
    while node is not None:
        path.append(node.config())
        #replace current_node with its parent to continue adding nodes into list
        node = node.parent
    path.reverse()
    return path

def is_close_enough(node1, node2):
    epsilon = 1e-6
    if abs(node1.x - node2.x) > epsilon:
        return False
    if abs(node1.y - node2.y) > epsilon:
        return False
    if abs(node1.theta - node2.theta) > epsilon:
        return False
    return True

def astar_research_algorithm(start_config, goal_config, collision_fn, mode, visualize=False):
    start_node = Node(start_config[0], start_config[1], start_config[2], None, 0)
    goal_node = Node(goal_config[0], goal_config[1], goal_config[2], None, -1)

    open_set = PriorityQueue()
    h_start = heuristic(start_node, goal_node)
    open_set.put((h_start, 0, start_node))

    explored_config = []
    count = 1

    while not open_set.empty():
        current = open_set.get()
        current_node = current[2]

        if current_node.config() not in explored_config:
            explored_config.append(current_node.config())
        else:
            continue

        if visualize:
            if collision_fn(current_node.config()):  
                draw_sphere_marker((current_node.x, current_node.y, 0.1), 0.05, (1, 0, 0, 1))  # In collision (red)
            else:
                draw_sphere_marker((current_node.x, current_node.y, 0.1), 0.05, (0, 0, 1, 1))  # Collision-free (blue)

        if collision_fn(current_node.config()):
            continue

        if is_close_enough(current_node, goal_node):
            path = pathtrace(current_node)
            # Draw the final path in black when the goal is reached
            if visualize:
                for config in path:
                    draw_sphere_marker((config[0], config[1], 0.1), 0.07, (0, 0, 0, 1))
            return path, explored_config
        
        neighbors = neighbor_connected(current_node, mode)
        for neighbor in neighbors:
            if neighbor.config() not in explored_config:
                neighbor.gn = current_node.gn + action_cost(current_node, neighbor)
                neighbor_fn_cost = neighbor.gn + heuristic(neighbor, goal_node)
                open_set.put((neighbor_fn_cost, count, neighbor))
                count += 1

    return [], explored_config  # Return empty path if goal is not reached


#########################

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('/home/aaron/ROB422/ROB422/HW3/HW3files/pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, -1.3, -np.pi/2)))

    # Example use of setting body poses
    # set_pose(obstacles['ikeatable6'], ((0, 0, 0), (1, 0, 0, 0)))

    # Example of draw 
    # draw_sphere_marker((0, 0, 1), 0.1, (1, 0, 0, 1))
    
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    #goal_config = (start_config[0]+5, start_config[1], start_config[2])
    goal_config = (2.6, -1.3, -np.pi/2)
    path = []
    start_time = time.time()
    ### YOUR CODE HERE ###
    
    for mode in ['8-connected']:
        print(f"\nRunning A* algorithm based on {mode} neighbors: ")

        # The 'visualize=True' argument ensures visualizations during the search.
        path = astar_research_algorithm(start_config, goal_config, collision_fn, mode, visualize=True)
        path_cost = sum(action_cost(Node(*path[i], -1, -1), Node(*path[i+1], -1, -1)) for i in range(len(path) - 1))
        print("Path cost for ", mode, ": ", path_cost)

    ######################
    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main() 