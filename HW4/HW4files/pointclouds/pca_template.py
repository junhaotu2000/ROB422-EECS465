#!/usr/bin/env python
import utils
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###

def pca_plane_fitting(pc_array):
    
    # PCA process
    point = np.mean(pc_array, axis=0)
    mean_centered_data = pc_array - point
    covariance_matrix = mean_centered_data.T @ mean_centered_data / (mean_centered_data.shape[0] - 1)
    _, s, vt = np.linalg.svd(covariance_matrix)
    
    # Fitting process
    normal = vt[-1]  
    d = -point.dot(normal)
    
    return normal, d
    
###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')
    
    ###YOUR CODE HERE###
    fig = utils.view_pc([pc])
    
    # Converting data set into a ndarray for processing
    pc_array = np.array([np.array(p).flatten() for p in pc]) 

    # Perform PCA using the covariance matrix Q = XX^T / (n-1)
    mean_centered_data = pc_array - np.mean(pc_array, axis=0)
    covariance_matrix = mean_centered_data.T @ mean_centered_data / (mean_centered_data.shape[0] - 1)
    
    # Use SVD to compute the eigenvectors and eigenvalues
    _, s, vt = np.linalg.svd(covariance_matrix)

    # Determine the number of components to keep
    # Here, you can specify the number of components you want to keep
    n_components_to_keep = len(s) - 1  # This will discard the smallest one

    # Keep only the top n_components_to_keep
    vs = vt[:n_components_to_keep, :]

    # Part a: Rotate the point cloud to align with the XY plane using Vt matrix
    pc_rotated = mean_centered_data.dot(vt.T)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc_rotated[:, 0], pc_rotated[:, 1], pc_rotated[:, 2])
    plt.title("Point Cloud Aligned with XY Plane")

    # Part b: Remove noise and rotate to 2D using Vs
    pc_filtered_rotated = mean_centered_data.dot(vs.T)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc_filtered_rotated[:, 0], pc_filtered_rotated[:, 1])
    plt.title("Filtered and Aligned Point Cloud")

    # Part c: Fit a plane and draw it
    # The plane is spanned by the first two principal components (largest eigenvalues)  
    normal, d = pca_plane_fitting(pc_array)
    
    # Fit a plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xx, yy = np.meshgrid(range(-10, 10), range(-10, 10))
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    ax.plot_surface(xx, yy, zz, alpha=0.5, rstride=100, cstride=100, color='g', zorder=0)
    ax.scatter(pc_array[:, 0], pc_array[:, 1], pc_array[:, 2], color='b', zorder=10)
    plt.title("Point Cloud with Fitted Plane")
    
    ###YOUR CODE HERE###

    plt.show()

if __name__ == '__main__':
    main()
