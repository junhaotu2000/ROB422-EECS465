#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
def distance(p, q):
    return np.linalg.norm(np.array(p) - np.array(q))

def findClosestPoint(p, pc_target):     
    dist = [distance(p, point) for point in pc_target]
    return (pc_target[np.argmin(dist)]) 

def GetTransform(P, Q):

    R, t = 0, 0
    pmean = sum(P)/len(P)
    qmean = sum(Q)/len(Q)

    X, Y= P, Q
    X[:] = [x - pmean for x in X]
    Y[:] = [y - qmean for y in Y]

    X = np.reshape(np.array(X), (len(P),3))
    Y = np.reshape(np.array(Y), (len(P),3))
    S = np.matmul(np.transpose(X),Y)
    U, _, VT = np.linalg.svd(S)
    UT = np.transpose(U)
    V = np.transpose(VT)
    M = np.array([[1,0,0],[0,1,0], [0,0,np.linalg.det(V @ UT)]])
    R = np.matmul( np.matmul(V, M), UT)
    t = np.asarray(qmean - np.matmul(R,pmean) )
    
    return R, t

def Computedistance(R, t, P, Q):
    
    error = 0
    dist_vec = np.matmul(R,P)+ t - Q 
    
    for i in range(len(P)):
        error += (dist_vec[i][0]*dist_vec[i][0] + dist_vec[i][1]*dist_vec[i][1] + dist_vec[i][2]*dist_vec[i][2])
    print("Error: ",error)
    
    return error

def icp(pc_source, pc_target, epsilon=0.5, MAX_ITER=100):
    idx = 0
    it, distances = [], []
    
    while True:   
        idx = idx + 1
        it.append(idx)
        if idx > MAX_ITER: 
            break

        P , Q = [], []
        for p in pc_source:
            q = findClosestPoint(p, pc_target)
            P.append(p)
            Q.append(q)

        R, t = GetTransform(P,Q)
        distance = Computedistance(R, t, pc_source, pc_target)
        distances.append(np.squeeze(distance).item(0))
        if distance <= epsilon:
            for i in range(len(P)):
                pc_source[i] = np.matmul(R,pc_source[i]) + t                
            break 
        for i in range(len(P)):
            pc_source[i] = np.matmul(R,pc_source[i]) + t
            
        print("\n======= Iteration ",idx,)
    return pc_source, pc_target, it, distances

###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')

    ###YOUR CODE HERE###
    target = 1
    pc_target = utils.load_pc(f'cloud_icp_target{target}.csv')
    
    
    ###YOUR CODE HERE###
    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15])
    plt.show()
    
    # raw_input("Press enter to end:")


if __name__ == '__main__':
    main()