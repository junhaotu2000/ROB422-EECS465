#!/usr/bin/env python
import utils
import numpy as np
import time
import random
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
from ransac_template import ransac_plane_fitting
from pca_template import pca_plane_fitting

def calculate_pca_inliers_outliers(pc_array, normal, d, threshold=0.05):
    # Compute the distance of all points to the PCA plane
    distances = np.abs((pc_array.dot(normal) + d) / np.linalg.norm(normal))

    # Determine inliers and outliers
    inliers_mask = distances <= threshold
    outliers_mask = distances > threshold

    inliers = pc_array[inliers_mask]
    outliers = pc_array[outliers_mask]

    return inliers, outliers

###YOUR IMPORTS HERE###
def add_some_outliers(pc,num_outliers):
    pc = utils.add_outliers_centroid(pc, num_outliers, 0.75, 'uniform')
    random.shuffle(pc)
    return pc

def main():
    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    inliers_ransac_errors = []
    inliers_pca_errors = []
    outliers_ransacs = []
    outliers_pcas = [] 
    ransac_times = []
    pca_times = []
    
    num_tests = 10
    fig = None
    for i in range(0,num_tests):
        pc = add_some_outliers(pc,10) #adding 10 new outliers for each test
        ###YOUR CODE HERE###
        # Converting data set into a ndarray for processing
        pc_array = np.array([np.array(p).flatten() for p in pc]) 
        
        # Ransac algorithm - timing
        start_time = time.time()
        normal_ransac, d_ransac, inliers_ransac, outliers_ransac = ransac_plane_fitting(pc_array)
        ransac_time = time.time() - start_time
        ransac_times.append(ransac_time * 1000)
        outliers_ransacs.append(len(outliers_ransac))
        
        # PCA algorithm - timing
        start_time = time.time()
        normal_pca, d_pca = pca_plane_fitting(pc_array)
        inliers_pca, outliers_pca = calculate_pca_inliers_outliers(pc_array, normal_pca, d_pca)
        pca_time = time.time() - start_time
        pca_times.append(pca_time * 1000)
        outliers_pcas.append(len(outliers_pca))
        
        # Calculate inliers errors for ransac and pca
        inliers_ransac_error = np.sum(np.square(np.abs(np.dot(inliers_ransac, normal_ransac) + d_ransac) / np.linalg.norm(normal_ransac)))
        inliers_pca_error = np.sum(np.square(np.abs(np.dot(inliers_pca, normal_pca) + d_pca) / np.linalg.norm(normal_pca)))
        
        inliers_ransac_errors.append(inliers_ransac_error)
        inliers_pca_errors.append(inliers_pca_error)
        
        # For the last iteration, generate two plot
        if i == num_tests - 1:
            # ---------- RANSAC -----------
            normal_ransac, d_ransac, inliers_ransac, outliers_ransac = ransac_plane_fitting(pc_array)
            
            # Ransac - Plotting
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Ransac - Inliers in red and Outliers in blue
            ax.scatter(inliers_ransac[:, 0], inliers_ransac[:, 1], inliers_ransac[:, 2], color='r', label='Inliers')
            ax.scatter(outliers_ransac[:, 0], outliers_ransac[:, 1], outliers_ransac[:, 2], color='b', label='Outliers')

            # Ransac - Plane in green
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
            Z = (-normal_ransac[0] * X - normal_ransac[1] * Y - d_ransac) / normal_ransac[2]
            ax.plot_surface(X, Y, Z, color='g', alpha=0.5, label='Fitted Plane')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_title('RANSAC Fitted Plane')

            # ---------- PCA -----------
            normal_pca, d_pca = pca_plane_fitting(pc_array)
            inliers_pca, outliers_pca = calculate_pca_inliers_outliers(pc_array, normal_pca, d_pca)
            
            # PCA - plotting
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # PCA - Inliers in red and Outliers in blue
            ax.scatter(inliers_pca[:, 0], inliers_pca[:, 1], inliers_pca[:, 2], color='r', label='PCA Inliers')
            ax.scatter(outliers_pca[:, 0], outliers_pca[:, 1], outliers_pca[:, 2], color='b', label='PCA Outliers')

            # PCA - Plane in green
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
            Z = (-normal_pca[0] * X - normal_pca[1] * Y - d_pca) / normal_pca[2]
            ax.plot_surface(X, Y, Z, color='g', alpha=0.5, label='PCA Fitted Plane')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_title('PCA Fitted Plane')
    
    # Error vs. Detected Number of Outliers for both RANSAC and PCA
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(outliers_ransacs, inliers_ransac_errors, 'o-', label='RANSAC Error')
    plt.plot(outliers_pcas, inliers_pca_errors, 's-', label='PCA Error')
    plt.xlabel('Detected Number of Outliers')
    plt.ylabel('Total Inliers Error')
    plt.title('Error vs. Detected Number of Outliers')
    plt.legend()

    # Computation Times vs Number of Iterations for both RANSAC and PCA
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_tests + 1), ransac_times, 'o-', label='RANSAC Computation Time')
    plt.plot(range(1, num_tests + 1), pca_times, 's-', label='PCA Computation Time')
    plt.xlabel('Iteration')
    plt.ylabel('Computation Time (ms)')
    plt.title('Computation Times vs Number of Iterations')
    plt.legend()
            
    plt.tight_layout()
    # Shoe figure     
    plt.show()

        ###YOUR CODE HERE###
    input("Press enter to end")

if __name__ == '__main__':
    main()
