#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
def ransac_plane_fitting(data_set, num_iterations = 100, threshold=0.05):
    best_inliers_count = -1
    best_plane = None
    
    for i in range(num_iterations):
        # Randomly sample 3 points
        indices = np.random.choice(data_set.shape[0], 3)
        p1, p2, p3 = data_set[indices]
        p1 = p1.squeeze()
        p2 = p2.squeeze()
        p3 = p3.squeeze()

        # Calculate the normal of the plane
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) == 0:  # Check for collinearity
            continue  
        normal = normal / np.linalg.norm(normal)  # Normalize
        d = -np.dot(normal, p1)

        # Make sure the data_set is a 2D array with shape (n_points, 3)
        data_set = data_set.squeeze()

        # Compute distances from points to the plane
        distances = np.abs(np.dot(data_set, normal) + d) / np.linalg.norm(normal)
        inliers = data_set[distances < threshold]

        # Update the best plane if needed
        if len(inliers) > best_inliers_count:
            best_inliers_count = len(inliers)
            best_plane = normal, d

    # Classify points based on the best plane
    normal, d = best_plane
    distances = np.abs(np.dot(data_set, normal) + d) / np.linalg.norm(normal)
    inliers = data_set[distances < threshold]
    outliers = data_set[distances >= threshold]
    # inliers_error = np.sum(np.square(np.abs(np.dot(inliers, normal) + d) / np.linalg.norm(normal)))
    
    return normal, d, inliers, outliers
###YOUR IMPORTS HERE###

def main():
    #Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')

    ###YOUR CODE HERE###
    # Convert data set into a ndarray for processing
    data_set = np.array(pc)

    # Parameters for RANSAC
    num_iterations = 100
    threshold = 0.05
    
     # Run RANSAC
    normal, d, inliers, outliers = ransac_plane_fitting(data_set, num_iterations, threshold)
    
    # Fitting equation
    print(f"The plane equation is: {normal[0]:.2f}x + {normal[1]:.2f}y + {normal[2]:.2f}z + {d:.2f} = 0")
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Inliers in red and Outliers in blue
    ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], color='r', label='Inliers')
    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], color='b', label='Outliers')

    # Plane in green
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = (-normal[0] * X - normal[1] * Y - d) / normal[2]
    ax.plot_surface(X, Y, Z, color='g', alpha=0.5, label='Fitted Plane')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    ###YOUR CODE HERE###
    plt.show()


if __name__ == '__main__':
    main()
