#Follow General Outline Given my P1 Detailed Briefing
#For now we will use pseudo code because we have questions for Dr. Keogh in tomorrow's lecture. 
import numpy as np
# from user_interface import validate_file
import random
import time 
import math

# landing bay is point 0

def nearest_neighbor_search(data, period, verbose=True, testing=False):
    '''
        Input:
            data: np.ndarray, shape: nxn
            period: time (seconds) before interrupt
        Output:
            BSF distance: int
            BSF order: list 
    '''
    if testing:
        BSF_over_time = []

    # nxn np array w/ distances
    dist_mat = create_dist_matrix(data)
    # when to end
    time_limit = time.time() + period
    # get with pure nearest neighbor greedy choice
    BSF_dist, BSF_order = nearest_neighbor_helper(dist_mat.copy(), False)
    if verbose:
        print(f"\t\t{BSF_dist:.1f}")
    
    # run until interupt
    prev_time = time.time()
    while time.time() < time_limit:
        # add a bit of randomness
        distance , order = nearest_neighbor_helper(dist_mat.copy(), True, BSF_dist)
        # if distance != float('inf'):
        #     print(f"Distance: {distance}")

        if distance < BSF_dist:
            BSF_dist = distance
            BSF_order = order
            if verbose:
                print(f"\t\t{BSF_dist:.1f}")

        if testing and time.time() - prev_time >= 1 and len(BSF_over_time) < period:
            BSF_over_time.append(BSF_dist)
            prev_time = time.time()
    
    if testing:
        if len(BSF_over_time) < period:
            while len(BSF_over_time) < period:
                BSF_over_time.append(BSF_dist)
        return BSF_dist, BSF_order, BSF_over_time
    return BSF_dist, BSF_order

def nearest_neighbor_helper(dist_mat, simulated_annealing, dist_to_beat = float('inf')):
    '''
        Input:
            dist_mat: nxn np.ndarray
            simulated_annealing: whether to have 10% chance to not choose shortest
            dist_to_beat: current best
        Output:
            distance: path's distance
            order: path's order
    '''
    visited = set()
    order = []
    distance = 0
    point = 0 # start at landing bay
    order.append(0)
    visited.add(0)

    while len(visited) != len(dist_mat):
        if simulated_annealing:
            # 10% chance to not choose shortest
            choose_shortest = random.randint(0,9)

        next_point = np.argmin(dist_mat[point, :]) # index of min for row
        while next_point in visited:
            #print("Next point already visited (1)")
            dist_mat[point, next_point] = float('inf') # changes only for row
            next_point = np.argmin(dist_mat[point, :]) # index of min for row

        # choose second shortest instead
        if simulated_annealing and choose_shortest < 1:
            if len(visited) < len(dist_mat)-1: # if not one final point
                # ignore closest point
                dist_mat[point, next_point] = float('inf')
                next_point = np.argmin(dist_mat[point, :]) 
                while next_point in visited:
                    #print("Next point already visited (2)")
                    dist_mat[point, next_point] = float('inf') 
                    next_point = np.argmin(dist_mat[point, :]) 
        
        # add distance to total distance for this path
        distance += dist_mat[point, next_point]

        # Early Abandoning, remove for two_opt
        # if simulated_annealing and distance >= dist_to_beat:
        #     # print("Abandoned Early")
        #     return float('inf'), None
        
        # add to path
        order.append(int(next_point))
        visited.add(next_point)
        point = next_point


    order.append(0)
    return distance, order

def create_dist_matrix(data):
    '''
        Input:
            data: nxn np.array containing locations for drone to visit
        Output:
            dist_mat: nxn np.ndarray, dist_mat[i,j] is euclidean distance between point i and point j
    '''
    n = data.shape[0]
    dist_mat = np.zeros((n, n))
    for i in range(n):
        dist_mat[i] = np.sqrt(np.sum((data[i,:] - data[:,:])**2, axis = 1))
    dist_mat[np.arange(0,n),np.arange(0,n)] = float('inf')
    return dist_mat
