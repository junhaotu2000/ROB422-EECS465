from pf import *
from kf import *
from pr2_models import *
from utils_filter import *
import numpy as np
import pybullet as p
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, wait_for_user,get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
import copy
from tqdm import tqdm

def main(screenshot=False):
    map_list = ["open_space.json", "minor_obstacle.json" , "maze.json"]
    path_list = ["open_space.txt", "minor_obstacle.txt", "maze.txt"]

    print(f"====================================\nDemo running...\nThere are 3 different maps in demo: open space, minor obstacle, maze\n============================")
    wait_for_user()
    
    rmse_kf = []
    rmse_pf = []
    collision_count_kf = []
    collision_count_pf = []
    time_kf = []
    time_pf = []

    for map_name, path_name in zip(map_list, path_list):

        print(f"============================\nRunning Demo with Map: {path_name}...\nShowing path...")
        # show path
        connect(use_gui=True)
        # p.resetDebugVisualizerCamera(cameraDistance = 5, cameraYaw = 0, cameraPitch = -60, cameraTargetPosition = [0, 0, 0])
        p.resetDebugVisualizerCamera(cameraDistance = 5, cameraYaw = 0, cameraPitch = -89.999, cameraTargetPosition = [0, 0, 0])
        robots, obstacles = load_env(map_name)
        base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]
        path_gui = read_path_from_file_no_interpolate(path_name)
        for pos in path_gui.T:
            draw(pos,'black', radius=0.07)
        execute_trajectory(robots['pr2'], base_joints, path_gui.T, sleep=0.01)
        print(f"Path shown, please following the guidance:")
        # wait_if_gui()
        disconnect()

        # run kf and pf
        print(f"========\nRunning Kalman Filter...")
        start = time.time()
        rmse, collision_count = main_kf(path_name, map_name)
        end = time.time()
        rmse_kf.append(rmse)
        collision_count_kf.append(collision_count)
        time_kf.append(end - start)
        # input("Press Enter to continue:")
        # plt.close()
        
        print(f"========\nRunning Particle Filter...")
        start = time.time()
        rmse, collision_count = main_pf(path_name, map_name)
        end = time.time()
        rmse_pf.append(rmse)
        collision_count_pf.append(collision_count)
        time_pf.append(end - start)
        # input("Press enter to continue:")
        # plt.close()
        # plt.close()
        
        print(f"Demo with Map: {map_name} Finished!\n============================")


    wait_if_gui()
    
    
    print(f"All Demo Finished!\nResult as following:")
    # Column widths
    width_col1 = 20
    width_col2 = 35
    width_col3 = 35
    width_col4 = 35

    # Header
    print(f"| {'MAP':<{width_col1}} | {'Open space':<{width_col2}} | {'Some obstacle':<{width_col3}} | {'Maze':<{width_col4}} |")
    print("|" + "-" * (width_col1 + 2) + "|" + "-" * (width_col2 + 2) + "|" + "-" * (width_col3 + 2) + "|" + "-" * (width_col4 + 2) + "|")
    print(f"| {'RMSE KF':<{width_col1}} | {format_list(rmse_kf[0], width_col2)} | {format_list(rmse_kf[1], width_col3)} | {format_list(rmse_kf[2], width_col4)} |")
    print(f"| {'RMSE PF':<{width_col1}} | {format_list(rmse_pf[0], width_col2)} | {format_list(rmse_pf[1], width_col3)} | {format_list(rmse_pf[2], width_col4)} |")
    print(f"| {'Collision Count KF':<{width_col1}} | {format_item(collision_count_kf[0], width_col2)} | {format_item(collision_count_kf[1], width_col3)} | {format_item(collision_count_kf[2], width_col4)} |")
    print(f"| {'Collision Count PF':<{width_col1}} | {format_item(collision_count_pf[0], width_col2)} | {format_item(collision_count_pf[1], width_col3)} | {format_item(collision_count_pf[2], width_col4)} |")
    print(f"| {'Time KF':<{width_col1}} | {format_item(time_kf[0], width_col2)} | {format_item(time_kf[1], width_col3)} | {format_item(time_kf[2], width_col4)} |")
    print(f"| {'Time PF':<{width_col1}} | {format_item(time_pf[0], width_col2)} | {format_item(time_pf[1], width_col3)} | {format_item(time_pf[2], width_col4)} |")

    print(f"\n====================================")
    
    wait_for_user("All Demo Finished! Result summary shown in table above. Press enter to end demo.")
    wait_if_gui()

if __name__ == '__main__':
    main()