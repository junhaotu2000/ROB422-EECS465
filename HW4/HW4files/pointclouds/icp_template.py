#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###


###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc_source = utils.load_pc('/home/aaron/ROB422/ROB422/HW4/HW4files/pointclouds/cloud_icp_source.csv')

    ###YOUR CODE HERE###
    pc_target = utils.load_pc('/home/aaron/ROB422/ROB422/HW4/HW4files/pointclouds/cloud_icp_target1.csv') # Change this to load in a different target

    
    pc_array_source = np.array([np.array(p).flatten() for p in pc_source]) 
    pc_array_target = np.array([np.array(p).flatten() for p in pc_target]) 
    
    print(pc_array_source.shape)














    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15])
    ###YOUR CODE HERE###

    plt.show()
    #raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
