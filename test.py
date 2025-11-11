import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from two_opt import two_opt_test as to
import time
import random
from kmeans import kmeans
from kmeans import assign_points

from validate_file import validate_file as vf

from data_visualization import plot_points

# create the unit square, the most basic test to see if our solution is working. 
def create_unit_square(n, filename):
    coords = [[0,0] for i in range(n)]
    for i, coord in enumerate(coords):
        if i == 0:
            continue
        #randomly choose whether to write the x coordinate first or the y coordinate first
        x_or_y = random.randint(0,1)
        coord[x_or_y] = random.random()
        coord[1-x_or_y] = random.randint(0,1)
    coords = np.array(coords)
    np.savetxt(filename, coords, delimiter = '   ')

def create_cluster(center, spread, num_points, filename):
    coords = []
    for _ in range(num_points):
        point = [
            random.gauss(center[0], spread),
            random.gauss(center[1], spread)
        ]
        coords.append(point)
    coords = np.array(coords)
    np.savetxt(filename, coords, delimiter = '   ')

if __name__ == '__main__':
    nodes = [
        64, 
        128, 
        256, 
        512, 
        1024
        ]

    # create t unit squares with n nodes
    if False:
        random.seed(42) # seed for reproducibility
        trials = 100
        for t in range(trials):
            for n in nodes:
                create_unit_square(n, f"data/unit_squares/{n}/{t+1}.txt")

    # create t random location files with n nodes
    if False:
        rng = np.random.default_rng(42) # seed for reproducibility
        trials = 100 # number of runs
        for t in range(trials):
            for n in nodes:
                datapath = f"data/random/{n}/{t+1}.txt"
                coords = rng.random(size=(n, 2))
                np.savetxt(datapath, coords, delimiter = '   ')

    # run trials for unit square optimality test
    if False:
        algo = 'TO' # change for each test
        trials = 30
        time = 20

        best_distances = []
        for n in nodes:
            distances = []
            for t in range(trials):
                data = vf(f"data/unit_squares/{n}/{t+1}.txt")
                dist, _, _ = search(algo, data, time, testing=True)
                distances.append(dist)
            print(f"Completed {trials} trials for {n} nodes unit square")
            np_distances = np.vstack(distances)
            best_distances.append(np.average(np_distances, axis=0))
        best_distances = np.concatenate(best_distances)
        np.savetxt(f"res/unit_square_optimality/{algo}_{trials}trials_{time}seconds.txt", best_distances, delimiter = '   ')

    # run trials for average bsf dist over time
    if False:
        t = 100 # number of runs
        n = 256
        s = 100
        output = f"{t}trials_rw1621_25051_{s}seconds"
        distances = []

        for i in range(t):
            datapath = f"rw1621_25051.txt"
            data = vf(datapath, sep = " ")
            dist, order, over_time = to(data, s)
            while len(over_time) < s:
                over_time = np.append(over_time, over_time[len(over_time) - 1])
            distances.append(over_time)

            # save in case something goes wrong mid trial
            np_distances = np.vstack(distances)
            np.savetxt(f"{output}_trial{i+1}.txt", np_distances)
            np.savetxt(f"{output}_trial{i+1}_average.txt", np.average(np_distances, axis=0))

            # delete previous file
            if i != 0:
                os.remove(f"{output}_trial{i}.txt")
                os.remove(f"{output}_trial{i}_average.txt")
    
    # run Kmeans trial times and take the minimum objective function value and plot it
    if False:
        time_taken = []
        for i in range(100):
            maxK = 4
            trials = 10
            datapath = f"ca4663_1290319.txt"
            data = vf(datapath, sep = " ")
            start_time = time.time()
            best_objs = []
            for k in range(1, maxK + 1):
                best_obj = float('inf')
                best_centers = None
                best_buckets = None
                for t in range(trials):
                    centers, obj = kmeans(data, k)
                    if obj < best_obj:
                        best_obj = obj
                        best_centers = centers
                        # assign points to buckets for best run
                        from kmeans import assign_points
                        best_buckets = assign_points(data, best_centers)
                print(f"Best objective function value for k={k}: {best_obj}")
                best_objs.append(best_obj)
            # k_values = list(range(1, maxK + 1))
            # plt.plot(k_values, best_objs, marker='o')
            # plt.title(f'K vs Kmeans Min Objective Function Value for {trials} Trials on {datapath}')
            # plt.xlabel('K value')
            # plt.ylabel('Objective Function Value (Lower is Better)')
            # plt.xticks(k_values)
            end_time = time.time()
            time_taken.append(end_time - start_time)
            # plt.show()
        avg_time = sum(time_taken) / len(time_taken)
        print(f"Average time taken for 1-4 Kmeans with {trials} trials on {datapath}: {avg_time:.2f} seconds")
    
    if False:
        # create clusters
        random.seed(69)
        create_cluster([0, 0], 0.5, 100, f"clusters/cluster_static.txt")
        for i in range(11):
            create_cluster([0, i*0.5], 0.5, 100, f"clusters/cluster{i}.txt")
    
    if False:
        # compare error between clusters
        clusters = []
        cluster_centers = []
        clusters.append(vf(f"clusters/cluster_static.txt"))
        cluster_centers.append(np.mean(clusters[0], axis=0))
        for i in range(11):
            # load cluster data and get their centers
            cluster_data = vf(f"clusters/cluster{i}.txt")
            cluster_center = np.mean(cluster_data, axis=0)
            print(f"Cluster {i} center: {cluster_center}")
            clusters.append(cluster_data)
            cluster_centers.append(cluster_center)
        errors = []
        for i in range(11):
            # calculate error as distance from cluster center
            combined_data = np.append(clusters[0], clusters[i], axis=0)
            output_centers, _ = kmeans(combined_data, 2)
            cluster1_error = np.linalg.norm(min(output_centers[0] - cluster_centers[0], output_centers[1] - cluster_centers[0], key=lambda x: np.linalg.norm(x)))
            cluster2_error = np.linalg.norm(min(output_centers[0] - cluster_centers[i], output_centers[1] - cluster_centers[i], key=lambda x: np.linalg.norm(x)))
            
            errors.append(np.sum(cluster1_error) + np.sum(cluster2_error))

    if False:
        datapath = f"ca4663_1290319.txt"
        data = vf(datapath, sep = " ")
        np.savetxt("ca4663_1290319.txt", data, delimiter = '   ')
        
