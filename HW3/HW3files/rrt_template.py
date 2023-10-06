import numpy as np
from utils import load_env, get_collision_fn_PR2, execute_trajectory
from pybullet_tools.utils import connect, disconnect, wait_if_gui, joint_from_name, get_joint_positions, set_joint_positions, get_joint_info, get_link_pose, link_from_name
import random
### YOUR IMPORTS HERE ###

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
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, 1.19, -1.548, 1.557, -1.32, -0.1928)))

    start_config = tuple(get_joint_positions(robots['pr2'], joint_idx))
    goal_config = (0.5, 0.33, -1.548, 1.557, -1.32, -0.1928)
    path = []
    ### YOUR CODE HERE ###
    

    ######################
    # Execute planned path
    execute_trajectory(robots['pr2'], joint_idx, path, sleep=0.1)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()