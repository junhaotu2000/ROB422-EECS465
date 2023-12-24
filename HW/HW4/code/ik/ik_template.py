import numpy as np
from pybullet_tools.utils import connect, disconnect, set_joint_positions, wait_if_gui, set_point, load_model,\
                                 joint_from_name, link_from_name, get_joint_info, HideOutput, get_com_pose, wait_for_duration
from pybullet_tools.transformations import quaternion_matrix
from pybullet_tools.pr2_utils import DRAKE_PR2_URDF
import time
import sys
### YOUR IMPORTS HERE ###

#########################

from utils import draw_sphere_marker

# Forward kinematics function
def get_ee_transform(robot, joint_indices, joint_vals=None):
    # returns end-effector transform in the world frame with input joint configuration or with current configuration if not specified
    if joint_vals is not None:
        set_joint_positions(robot, joint_indices, joint_vals)
    ee_link = 'l_gripper_tool_frame'
    pos, orn = get_com_pose(robot, link_from_name(robot, ee_link))
    res = quaternion_matrix(orn)
    res[:3, 3] = pos
    return res

def get_joint_axis(robot, joint_idx):
    # returns joint axis in the world frame
    j_info = get_joint_info(robot, joint_idx)
    jt_local_pos, jt_local_orn = j_info.parentFramePos, j_info.parentFrameOrn
    H_L_J = quaternion_matrix(jt_local_orn) # joint transform in parent link CoM frame
    H_L_J[:3, 3] = jt_local_pos
    parent_link_world_pos, parent_link_world_orn = get_com_pose(robot, j_info.parentIndex)
    H_W_L = quaternion_matrix(parent_link_world_orn) # parent link CoM transform in world frame
    H_W_L[:3, 3] = parent_link_world_pos
    H_W_J = np.dot(H_W_L, H_L_J)
    R_W_J = H_W_J[:3, :3]
    joint_axis_local = np.array(j_info.jointAxis)
    joint_axis_world = np.dot(R_W_J, joint_axis_local)
    return joint_axis_world

def get_joint_position(robot, joint_idx):
    # returns joint position in the world frame
    j_info = get_joint_info(robot, joint_idx)
    jt_local_pos, jt_local_orn = j_info.parentFramePos, j_info.parentFrameOrn
    H_L_J = quaternion_matrix(jt_local_orn) # joint transform in parent link CoM frame
    H_L_J[:3, 3] = jt_local_pos
    parent_link_world_pos, parent_link_world_orn = get_com_pose(robot, j_info.parentIndex)
    H_W_L = quaternion_matrix(parent_link_world_orn) # parent link CoM transform in world frame
    H_W_L[:3, 3] = parent_link_world_pos
    H_W_J = np.dot(H_W_L, H_L_J)
    j_world_posi = H_W_J[:3, 3]
    return j_world_posi

def set_joint_positions_np(robot, joints, q_arr):
    # set active DOF values from a numpy array
    q = [q_arr[0, i] for i in range(q_arr.shape[1])]
    set_joint_positions(robot, joints, q)


def get_translation_jacobian(robot, joint_indices):
    J = np.zeros((3, len(joint_indices)))
    ### YOUR CODE HERE ###
    for i, joint_idx in enumerate(joint_indices):
        # get joint axis vector
        joint_axis_world_i = get_joint_axis(robot, joint_idx)
        
        # get joint_ee_vector
        joint_position_vector_i = get_joint_position(robot, joint_idx) 
        ee_transform = get_ee_transform(robot, joint_indices)
        ee_position_vector  = ee_transform[:3, 3]
        joint_ee_vector_i = ee_position_vector - joint_position_vector_i
    
        # cross product the axis_vector with joint_ee_vector to obtain postion part of 
        # Jacbian matrix of a particular joint
        # fill the Jacbian matrix in the order of columns
        J[:, i] = np.cross(joint_axis_world_i, joint_ee_vector_i) 
    ### YOUR CODE HERE ###
    return J

def get_jacobian_pinv(J):
    J_pinv = []
    ### YOUR CODE HERE ###
    # N > M, this is a redundant robot, thus apply left least square
    # damped matrix to avoid singularity
    lambada = 0.01 
    damped_matrix = lambada * np.eye(J.shape[0])
    
    # formula of left side damped least square 
    J_pinv = np.transpose(J) @ np.linalg.inv( J @ np.transpose(J) + damped_matrix)
    ### YOUR CODE HERE ###
    return J_pinv

# set inital position of the robot joint
def tuck_arm(robot):
    joint_names = ['torso_lift_joint','l_shoulder_lift_joint','l_elbow_flex_joint',\
        'l_wrist_flex_joint','r_shoulder_lift_joint','r_elbow_flex_joint','r_wrist_flex_joint']
    joint_idx = [joint_from_name(robot, jn) for jn in joint_names]
    set_joint_positions(robot, joint_idx, (0.24,1.29023451,-2.32099996,-0.69800004,1.27843491,-2.32100002,-0.69799996))


def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print("Specify which target to run:")
        print("'python3 ik_template.py [target index]' will run the simulation for a specific target index (0-4)")
        exit()
    test_idx = 0
    try:
        test_idx = int(args[0])
    except:
        print("ERROR: Test index has not been specified")
        exit()

    # initialize PyBullet
    connect(use_gui=True, shadows=False) 
    
    # load robot
    with HideOutput():
        robot = load_model(DRAKE_PR2_URDF, fixed_base=True) # create a robot model with URDF, the robot arm is fixed at base
        set_point(robot, (-0.75, -0.07551, 0.02)) 
    tuck_arm(robot) # set inital position of the robot
    
    # define active DoFs
    joint_names =['l_shoulder_pan_joint','l_shoulder_lift_joint','l_upper_arm_roll_joint', \
        'l_elbow_flex_joint','l_forearm_roll_joint','l_wrist_flex_joint','l_wrist_roll_joint'] 
    joint_idx = [joint_from_name(robot, jn) for jn in joint_names]
    
    # intial config
    q_arr = np.zeros((1, len(joint_idx))) 
    set_joint_positions_np(robot, joint_idx, q_arr)
    
    # list of example targets
    targets = [[-0.15070158,  0.47726995, 1.56714123],
               [-0.36535318,  0.11249,    1.08326675],
               [-0.56491217,  0.011443,   1.2922572 ],
               [-1.07012697,  0.81909669, 0.47344636],
               [-1.11050811,  0.97000718,  1.31087581]] #定义一组示例目标点。
    
    # define joint limits
    joint_limits = {joint_names[i] : (get_joint_info(robot, joint_idx[i]).jointLowerLimit, get_joint_info(robot, joint_idx[i]).jointUpperLimit) for i in range(len(joint_idx))}
    q = np.zeros((1, len(joint_names))) # start at this configuration
    target = targets[test_idx]
    
    # draw a blue sphere at the target
    draw_sphere_marker(target, 0.05, (0, 0, 1, 1))
    
    ### YOUR CODE HERE ###
    q_current = q_arr
    x_target = np.array(targets[test_idx])
    
    threshold = 0.01
    step_size = 0.002
     
    while True:
        # Get the current end-effector position from the forward kinematics
        x_current = get_ee_transform(robot, joint_idx)[:3, 3]
        x_change = x_target - x_current
            
        # Check if the current end-effector position is close enough to the target
        if np.linalg.norm(x_change) < threshold:
            break
            
        # Compute the change in joint positions using the pseudoinverse Jacobian
        J = get_translation_jacobian(robot, joint_idx)
        J_pinv = get_jacobian_pinv(J)
        q_change = J_pinv @ x_change.reshape((3,1))
         
        # If the change is too large, scale it down to the step size   
        if np.linalg.norm(q_change) > step_size:
            q_change = step_size * (q_change / np.linalg.norm(q_change))
            
        # Update joint variables
        q_current += q_change.T
            
        # Check joint limits
        for idx, joint_id in enumerate(joint_idx):
            joint_info = get_joint_info(robot, joint_id)
            lower_limit, upper_limit = joint_info.jointLowerLimit, joint_info.jointUpperLimit
            
            # Check lower limit
            if q_current[0, idx] < lower_limit:
                q_current[0, idx] = lower_limit
            # Check upper limit
            elif q_current[0, idx] > upper_limit:
                q_current[0, idx] = upper_limit

        # Set joint variables on robot
        set_joint_positions_np(robot, joint_idx, q_current)
            
    print(f"Reach target point {test_idx} ")
    print(f"Configuration of robot: {q_current}")
    ### YOUR CODE HERE ###

    wait_if_gui() 
    disconnect()

if __name__ == '__main__':
    main()