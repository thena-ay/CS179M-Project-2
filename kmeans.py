import numpy as np

def kmeans(data, k):
    '''
        Input:
            data: (n, 2) numpy array
            k: number of clusters
        Output:
            k files with cluster locations?
    '''
    #TODO: more research on kmeans initialization
    # initialize centers (random points from data set)
    chosen = np.random.choice(data.shape[0], size=k, replace=False)
    centers = data[chosen]

    # assign points to centers
    prev_buckets = assign_points(data, centers)

    while True:
        # move centers based on points and reassign points
        centers = update_centers(prev_buckets, data)
        # print(centers)
        buckets = assign_points(data, centers)

        # end when no points reassigned
        if buckets == prev_buckets:
            break
        prev_buckets = buckets
    return centers, objective_func(buckets, data, centers)

# update center based on points in its bucket
def update_centers(buckets, data):
    '''
        Input:
            buckets: idctionary with k keys and list of point indices for value
            data: nx2 numpy array
        Output:
            centers: kx2 numpy array, location of centers based on current bucket
    '''
    k = len(buckets)
    centers = np.empty((k, 2))

    # move center to average of points in its bucket
    for i in range(k):
        points = data[buckets[i]]
        centers[i] = np.average(points, axis=0)

    return centers

# assign points to center
def assign_points(data, centers):
    '''
        Input:
            data: nx2 numpy array
            centers: kx2 numpy array
        Output:
            points: dictionary with k keys and list of point indices for value
    '''
    n, k = data.shape[0], centers.shape[0]

    # dictionary to store which center point corresponds to 
    buckets = {}
    for i in range(k):
        buckets[i] = []
    
    # find closest cluster center for each point
    for i in range(n):
        center = 0
        smallest_dist = np.sqrt(np.sum((data[i,:] - centers[0,:])**2))
        for j in range(1, k):
            dist = np.sqrt(np.sum((data[i,:] - centers[j,:])**2))
            if dist < smallest_dist: # tie breaker or just first one?
                smallest_dist = dist
                center = j
        # add to center's group
        buckets[center].append(i)
            
    return buckets

# calculate mean squared error as objective function
def objective_func(buckets, data, centers):
    k = len(centers)

    sq_errors = []
    for i in range(k):
        # mean squared error euclidean distance
        sq_error = np.average(np.sqrt(np.sum((data[buckets[i],:] - centers[i,:])**2, axis=1))**2)
        sq_errors.append(sq_error)
    sq_errors = np.hstack(sq_errors)
    return np.average(sq_errors)

if __name__ == '__main__':
    # toy example from cs171
    centers = np.array([[1,3], [3, 2]])
    data = np.array([[0, 1], [1, 2], [2, 0], [2, 2], [2, 3], [3, 1], [4, 0], [4, 2]])

    prev_buckets = assign_points(data, centers)
    print(prev_buckets)

    change = True
    while change:
        print(objective_func(prev_buckets, data, centers))
        # move centers based on points and reassign points
        centers = update_centers(prev_buckets, data)
        print(centers)
        buckets = assign_points(data, centers)
        print(buckets)
        # check if any points have been reassigned
        if buckets == prev_buckets:
            change = False
        prev_buckets = buckets

    # works for this example, need to test more thoroughly to ensure correctness