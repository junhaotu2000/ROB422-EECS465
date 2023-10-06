import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###

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
    goal_config = (2.6, -1.3, -np.pi/2)
    path = []
    start_time = time.time()
    ### YOUR CODE HERE ###
    
    
    ######################
    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()