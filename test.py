import numpy as np

from TSP.main import validate_file as vf
from TSP.two_opt import two_opt as to
from visualization import plot_clusters, plot_time_taken
from kmeans import kmeans, objective_func, assign_points

if __name__ == "__main__":
    # visualize clusters with our kmeans algorithm
    if False:
        filename = "pecan1212.txt"
        data = vf(filename)
        for i in range(4):
            # 10 trials to avoid unlucky scenario
            centers, of = kmeans(data, i+1)
            best_centers, best_of = centers, of
            for j in range(9):
                centers, of = kmeans(data, i+1)
                if of < best_of:
                    best_centers, best_of = centers, of
            print(best_of)
            buckets = assign_points(data, best_centers)
            plot_clusters(buckets, data, best_centers)

    if False:
        filename = "Almond9832.txt"
        data = vf(filename)
        times = np.zeros((10, 4))
        setup = np.array([2, 4, 6, 8])
        for t in range(10):
            for i in range(4):
                print(f"Trial {t+1} Drone {i+1}")
                centers, of = kmeans(data, i+1)
                best_centers, best_of = centers, of
                for j in range(9):
                    centers, of = kmeans(data, i+1)
                    if of < best_of:
                        best_centers, best_of = centers, of
            
                buckets = assign_points(data, best_centers)

                max_dist = float('-inf')
                # calculate distances for best centers
                for k in range(i+1):
                    points = data[buckets[k]]
                    points = np.vstack((points, best_centers[k]))
                    if i < 2:
                        dist, _ = to(points, 1)
                    else:
                        dist, _ = to(points, 1)

                    if dist > max_dist:
                        max_dist = dist
                times[t][k] = max_dist // 100
                print(times)
        np.savetxt("time_taken.txt", times)
        np.savetxt("time_taken_average.txt", np.average(times))
        plot_time_taken(np.average(times))     
                        