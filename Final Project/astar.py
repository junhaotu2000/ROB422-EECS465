import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###    
from queue import PriorityQueue

class Node:
    def __init__(self, config: np.ndarray, parent = None, g_cost = 0, h_cost = 0,) -> None:
        self.config = np.array(config)
        self.parent = parent
        self.g_cost = g_cost
        self.h_cost = h_cost

    def __lt__(self, other: object) -> bool:
        return self.f_cost < other.f_cost

    def __eq__(self, node: object) -> bool:
        dx = np.abs(self.config[0] - node.config[0])
        dy = np.abs(self.config[1] - node.config[1])
        dtheta = np.abs(self.config[2] - node.config[2])
        return dx < 1e-4 and dy < 1e-4 and dtheta < 1e-4
    
    def __hash__(self) -> int:
        return hash((self.config[0], self.config[1], self.config[2]))
    
    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

class CloseList:
    def __init__(self) -> None:
        self.nodes = set()

    def append(self, node: Node) :
        self.nodes.add(node)

    def contains(self, node) -> bool:
        if node in self.nodes:
            return True
        return False

class OpenList:
    def __init__(self) -> None:
        self.queue = PriorityQueue()
        self.nodes = {}

    def empty(self) -> bool:
        result = self.queue.empty()
        return result

    def put(self, node: Node) -> None:
        self.queue.put(node)
        self.nodes[node] = node.g_cost
    
    def get(self) -> Node:
        node = self.queue.get()
        return node

    def contains(self, node: Node) -> bool:
        if node in self.nodes:
            return True
        return False
    
    def get_g_cost(self, node: Node) -> float:
        g_cost = self.nodes[node]
        return g_cost
    

def search_path(start, goal, is_collision, connectivity = '8', grid_size = 0.1):
    '''
    return path found
    '''
    print(f"############\nFinding {connectivity}-connected path...")
    path = []
    if tuple(start) == tuple(goal):
        return []

    decimal = count_decimal_places(grid_size)
    start = np.round(start, decimal)
    goal = np.round(goal, decimal)
    start = Node(start)
    goal = Node(goal)

    openlist = OpenList() # [(cost1, node1), (cost2, node2), ...] # store the nodes we want to explore
    closelist = CloseList() # {node1, node2, ...} # store nodes already explored
    collidelist = CloseList()
    id = 0

    curr = start
    curr.g_cost = g_cost(start, curr)
    curr.h_cost = h_cost(curr, goal)
    openlist.put(curr)
    id += 1

    while not openlist.empty():
    #     curr = openlist.get()[2]
    #     print(f"Curr config: {curr.config}, g: {curr.g_cost}, h: {curr.h_cost}, f: {curr.h_cost + curr.g_cost}")
    #     closelist = put_in_closelist(curr, closelist)
    #     if is_the_goal(curr, goal):
    #         path = extract_path(curr, path)
    #         return path
    #     neighbors = expand(curr, is_collision, connectivity, grid_size)
    #     for neighbor in neighbors:
    #         if is_in_closelist(neighbor, closelist):
    #             continue
    #         neighbor.g_cost = curr.g_cost + g_cost(curr, neighbor)
    #         neighbor.h_cost = h_cost(neighbor, goal)
    #         # found, higher = is_in_openlist(neighbor, openlist)
    #         # if found and higher:
    #         #     continue
    #         put_in_openlist(neighbor, id, openlist)
    #         id += 1
        
        # curr = openlist.get()[2]
        # curr.config = np.round(curr.config, decimal)
        # print(f"Curr config: {curr.config}, g: {curr.g_cost}, h: {curr.h_cost}, f: {curr.h_cost + curr.g_cost}")
        # neighbors = expand(curr, is_collision, connectivity, grid_size)
        # for neighbor in neighbors:
        #     neighbor.g_cost = curr.g_cost + g_cost(curr, neighbor)
        #     neighbor.h_cost = h_cost(neighbor, goal)
        #     if is_the_goal(neighbor, goal):
        #         path = extract_path(neighbor, path)
        #         return path
        #     if tuple(neighbor.config) not in closelist:
        #         found, higher = is_in_openlist(neighbor, openlist)
        #         if not found or not higher:
        #             openlist.put((curr.g_cost + curr.h_cost, id, neighbor))
        #             id += 1
        # closelist[tuple(curr.config)] = curr
        curr = openlist.get()
        if closelist.contains(curr):
            continue
        closelist.append(curr)
        if curr == goal:
            path = extract_path(curr, path)
            g = curr.g_cost
            print(f"Path found!")
            print(f"Path g cost: {g}")
            # for n in closelist.nodes:
            #     draw((n.config[0], n.config[1], 0.1), 'blue')
            # for n in collidelist.nodes:
            #     draw((n.config[0], n.config[1], 0.1), 'red')
            return path
        neighbors = expand(curr, is_collision, connectivity, grid_size, collidelist)
        for neighbor in neighbors:
            if closelist.contains(neighbor):
                continue
            nei_g_cost = curr.g_cost + g_cost(curr, neighbor)
            if not openlist.contains(neighbor) or nei_g_cost < openlist.get_g_cost(neighbor):
                # draw((neighbor.config[0], neighbor.config[1], 0.1), 'blue')
                neighbor.g_cost = nei_g_cost
                neighbor.h_cost = h_cost(neighbor, goal)
                neighbor.parent = curr
                openlist.put(neighbor)
    print(f"Path not found!")
    return []

# def put_in_openlist(node: Node, id, openlist: PriorityQueue):
#     item = (node.g_cost + node.h_cost, id, node)
#     openlist.put(item)
#     # return openlist

# def put_in_closelist(node: Node, closelist):
#     closelist.append(node)
#     return closelist

# def is_in_closelist(node: Node, closelist):
#     for n in closelist:
#         if (node.config == n.config).all():
#             return True
#     return False

# def is_in_openlist(node: Node, openlist: PriorityQueue):
#     temp_queue = PriorityQueue()
#     found = False
#     higher = False # higher the cost

#     # Iterate through the PriorityQueue
#     while not openlist.empty():
#         item = openlist.get()
#         temp_queue.put(item)

#         # Check if the node matches the target_node
#         if (item[2].config == node.config).all():
#             found = True
#             if node.g_cost > item[2].g_cost:
#                 higher = True

#     # Restore the original PriorityQueue
#     while not temp_queue.empty():
#         openlist.put(temp_queue.get())

#     return found, higher

# def is_the_goal(node, goal):
#     return (node.config[0:2] == goal.config[0:2]).all()

def count_decimal_places(number):
    # Convert the number to a string
    number_str = str(number)
    # Check if there is a decimal point
    if '.' in number_str:
        # Split the string using the decimal point as the delimiter
        integer_part, decimal_part = number_str.split('.')
        # Count the number of decimal places
        return len(decimal_part)
    else:
        # If there's no decimal point, there are no decimal places
        return 0

def g_cost(curr: Node, neighbor: Node):
    '''
    return g cost
    '''
    dx = curr.config[0] - neighbor.config[0]
    dy = curr.config[1] - neighbor.config[1]
    dtheta = np.min((np.abs(curr.config[2] - neighbor.config[2]), 2*np.pi - np.abs(curr.config[2] - neighbor.config[2])))
    g_cost = np.sqrt(dx**2 + dy**2 + dtheta**2)
    return g_cost

def h_cost(neighbor: Node, goal: Node):
    '''
    return h cost
    '''
    dx = neighbor.config[0] - goal.config[0]
    dy = neighbor.config[1] - goal.config[1]
    dtheta = np.min((np.abs(neighbor.config[2] - goal.config[2]), 2*np.pi - np.abs(neighbor.config[2] - goal.config[2])))
    h_cost = np.sqrt(dx**2 + dy**2 + dtheta**2)
    return h_cost

# def f_cost(start: Node, neighbor: Node, goal: Node):
#     '''
#     return total cost
#     '''
#     return g_cost(start, neighbor) + h_cost(neighbor, goal)

def expand(curr: Node, is_collision, connectivity, grid_size, collidelist: CloseList) -> list:
    '''return all neighborhood that are not an obstacle, np.array([node1, node2, ...])'''
    neighbors = []
    x = curr.config[0]
    y = curr.config[1]
    theta = curr.config[2]

    decimal = count_decimal_places(grid_size)

    # if connectivity == '4':
    #     moves = np.array([[grid_size, 0, 0],
    #             [-grid_size, 0, 0],
    #             [0, grid_size, 0],
    #             [0, -grid_size, 0],
    #             [0, 0, np.pi/2],
    #             [0, 0, -np.pi/2],
    #             [0, 0, np.pi]])
    #     for move in moves:
    #         if not is_collision(tuple(np.round(curr.config + move, decimal))):
    #             new_node = Node(np.round(curr.config + move, decimal), curr)
    #             neighbors.append(new_node)


    dxs = [0, grid_size, -grid_size]
    dys = [0, grid_size, -grid_size]
    dthetas = [0, np.pi/2, -np.pi/2, np.pi]
    # dthetas = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]

    if connectivity == '4':
        for dx in dxs:
            if not dx == 0:
                if not is_collision(tuple(np.round([x + dx, y, theta], decimal))):
                    new_node = Node(np.round([x + dx, y, theta], decimal), curr)
                    neighbors.append(new_node)
                else:
                    collidelist.append(Node(np.round([x + dx, y, theta], decimal), curr))
        for dy in dys:
            if not dy == 0:
                if not is_collision(tuple(np.round([x, y + dy, theta], decimal))):
                    new_node = Node(np.round([x, y + dy, theta], decimal), curr)
                    neighbors.append(new_node)
                else:
                    collidelist.append(Node(np.round([x, y + dy, theta], decimal), curr))
        for dtheta in dthetas:
            if not dtheta == 0:
                if not is_collision(tuple(np.round([x, y, dtheta], decimal))):
                    new_node = Node(np.round([x, y, dtheta], decimal), curr)
                    neighbors.append(new_node)
                else:
                    collidelist.append(Node(np.round([x, y, dtheta], decimal), curr))


    if connectivity == '8':        
        for dx in dxs:
            for dy in dys:
                for dtheta in dthetas:
                    if not dx == dy == dtheta == 0:
                        if not is_collision(tuple(np.round([x + dx, y + dy, dtheta], decimal))):
                            new_node = Node(np.round([x + dx, y + dy, dtheta], decimal), curr)
                            neighbors.append(new_node) 
                        else:
                            new_node = Node(np.round([x + dx, y + dy, dtheta], decimal), curr)
                            collidelist.append(new_node) 

    return np.array(neighbors)

def extract_path(final: Node, path: list) -> list:
    '''return path by calling parents'''
    curr = final
    path.append(list(curr.config))
    draw((curr.config[0], curr.config[1], 0.1),'black', radius=0.07)
    while curr.parent is not None:
        curr = curr.parent
        path.append(list(curr.config))
        draw((curr.config[0], curr.config[1], 0.1),'black', radius=0.07)
    path.reverse()
    return path

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
    
                
#########################

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    # robots, obstacles = load_env('pr2maze.json')
    robots, obstacles = load_env('pr2empty.json')
    # robots, obstacles = load_env('pr2complexMaze.json')

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
    # goal_config = (3.4, -1.3, -np.pi/2)
    goal_config = (3.4, 0, -np.pi/2)
    path = []
    start_time = time.time()
    ### YOUR CODE HERE ###
    path = search_path(np.array(start_config), np.array(goal_config), collision_fn, connectivity='8', grid_size=0.1)
    path = np.array(path)

    # store the path for localization
    path = path.T
    # with open('path_maze.txt', 'w') as file:
    with open('open_space.txt', 'w') as file:
    # with open('path_complexMaze.txt', 'w') as file:
        for item in path:
            file.write(f'{item}\n')
    path = list(path.T)
    ######################
    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()