import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
from queue import PriorityQueue

def create_node(x_in, y_in, theta_in, parent, gn):
    """
    Create a dictionary containing node information.

    :param x_in: The x-coordinate of the node.
    :param y_in: The y-coordinate of the node.
    :param theta_in: The angle theta of the node.
    :param parent: The parent node of the current node.
    :param gn: The cost from the start node to the current node.
    :return: A dictionary containing node information.
    """
    return {'x': x_in, 'y': y_in, 'theta': theta_in, 'parent': parent, 'gn': gn}

def config(node):
    """
    Extract and return the configuration (position and orientation) of a node.

    :param node: The node in dictionary format.
    :return: A tuple of the node's (x, y, theta).
    """
    return (node['x'], node['y'], node['theta'])

def distance(n, m):
    """
    Calculate the Euclidean distance between two nodes.

    :param n: The first node.
    :param m: The second node.
    :return: The distance between the two nodes.
    """
    return np.sqrt((n['x'] - m['x'])**2 + (n['y'] - m['y'])**2 + 
                   min(abs(n['theta'] - m['theta']), 2*np.pi - abs(n['theta'] - m['theta']))**2)

def action_cost(n, m):
    """
    Calculate the cost of moving from one node to another.

    :param n: The starting node.
    :param m: The destination node.
    :return: The movement cost between the two nodes.
    """
    return distance(n, m)

def heuristic(n, goal):
    """
    Heuristic function estimating the cost from a node to the goal node.

    :param n: The current node.
    :param goal: The goal node.
    :return: The estimated cost.
    """
    return distance(n, goal)

def boundingangle(theta):
    """
    Bounds an angle within [-π, π).

    :param theta: The input angle.
    :return: The adjusted angle.
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi

def neighbor_connected(node, mode):
    """
    Generate adjacent nodes for a given node.

    :param node: The central node.
    :param mode: The connection mode ('4-connected' or '8-connected').
    :return: A list of neighbor nodes.
    """
    neighbor = []
    if mode == '4-connected':
        # Generate neighbor nodes based on 4-connected mode
        for dx, dy, dtheta in [[-0.15, 0, 0], [0.15, 0, 0], [0, 0.15, 0], [0, -0.15, 0], [0, 0, -np.pi/2], [0, 0, np.pi/2]]:
            neighbor_node = create_node(node['x'] + dx, node['y'] + dy, boundingangle(node['theta'] + dtheta), node, -1)
            neighbor.append(neighbor_node)
    
    elif mode == '8-connected':
        # Generate neighbor nodes based on 8-connected mode
        for dx in [-0.15, 0, 0.15]:
            for dy in [-0.15, 0, 0.15]:
                for dtheta in [-np.pi/2, 0, np.pi/2]:
                    if not(dx == 0 and dy == 0 and dtheta == 0):
                        neighbor_node = create_node(node['x'] + dx, node['y'] + dy, boundingangle(node['theta'] + dtheta), node, -1)
                        neighbor.append(neighbor_node)
    return neighbor

def pathtrace(node):
    """
    Trace the path from a given node back to the start node.

    :param node: The end node.
    :return: The path from the start node to the end node.
    """
    path = []
    while node:
        path.append(config(node))
        node = node['parent']
    path.reverse()
    return path

def is_close_enough(node1, node2):
    """
    Determine if two nodes are close enough.

    :param node1: The first node.
    :param node2: The second node.
    :return: True if the two nodes are close enough; otherwise, False.
    """
    epsilon = 0.1
    return all(abs(node1[key] - node2[key]) < epsilon for key in ['x', 'y', 'theta'])

def astar(start_config, goal_config, collision_fn, mode):
    """
    Implementation of the A* algorithm.

    :param start_config: Starting configuration (x, y, theta).
    :param goal_config: Goal configuration (x, y, theta).
    :param collision_fn: A function to detect collisions.
    :param mode: The connection mode ('4-connected' or '8-connected').
    :return: The path and explored configurations.
    """
    start_node = create_node(start_config[0], start_config[1], start_config[2], None, 0)
    goal_node = create_node(goal_config[0], goal_config[1], goal_config[2], None, -1)

    open_set = PriorityQueue()
    h_start = heuristic(start_node, goal_node)
    open_set.put((h_start, 0, start_node))

    explored_config = []
    count = 1

    while not open_set.empty():
        current = open_set.get()
        current_node = current[2]

        if config(current_node) not in explored_config:
            explored_config.append(config(current_node))
        else:
            continue

        if collision_fn(config(current_node)):
            continue

        if is_close_enough(current_node, goal_node):
            path = pathtrace(current_node)
            return path, explored_config
        
        neighbors = neighbor_connected(current_node, mode)
        for neighbor in neighbors:
            if config(neighbor) not in explored_config:
                neighbor['gn'] = current_node['gn'] + action_cost(current_node, neighbor)
                neighbor_fn_cost = neighbor['gn'] + heuristic(neighbor, goal_node)
                open_set.put((neighbor_fn_cost, count, neighbor))
                count += 1

    return [], explored_config

#########################

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

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
    
    # Running the A* algorithm for 8-connected neighbors
    for mode in ['8-connected']:
        print(f"\nRunning A* algorithm based on {mode} neighbors: ")
        path, explored_config = astar(start_config, goal_config, collision_fn, mode)
        print("A* algorithm planning finish, run time is ", time.time() - start_time)
    
    # Check if a path has been found
    if path:
        path_cost = sum(action_cost(create_node(*path[i], None, -1), create_node(*path[i + 1], None, -1)) for i in range(len(path) - 1))
        print("Path cost for ", mode, ": ", path_cost)
        
    else: 
        print("No Solution Found.")
    
    # Visualize the path   
    for config in path:
        # Draw each configuration of the path as a black sphere
        draw_sphere_marker((config[0], config[1], 0.1), 0.07, (0, 0, 0, 1))

    # Visualize all explored configurations
    for draw_config in explored_config:
        # Draw configurations that are in collision as red spheres
        if collision_fn(draw_config):  
            draw_sphere_marker((draw_config[0], draw_config[1], 0.1), 0.05, (1, 0, 0, 1))
        else:  
            # Draw collision-free configurations as blue spheres
            draw_sphere_marker((draw_config[0], draw_config[1], 0.1), 0.05, (0, 0, 1, 1))

    ######################
    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main() 