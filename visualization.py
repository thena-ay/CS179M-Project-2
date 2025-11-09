import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_clusters(buckets, data, centers):
    k = centers.shape[0]
    plt.figure(figsize=(6, 6))
    colors = ['red', 'blue', 'green', 'purple', 'teal', 'orange', 'pink']
    for i in range(k):
        points = data[buckets[i]]
        sns.scatterplot(x=points[:,0], y=points[:, 1], color=colors[i], markers='o')
    sns.scatterplot(x=centers[:,0], y=centers[:,1], color='black', markers='o', s=100)
    plt.title('Locations')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

def plot_cluster_paths(points, centers, orders, filename):
    centers = np.array(centers)
    k = centers.shape[0]
    plt.figure(figsize=(6,6))
    colors = ['orange', 'blue', 'green', 'purple']
    
    for i in range(k):
        reordered_points = points[i][orders[i]]
        plt.plot(reordered_points[:, 0], reordered_points[:, 1], markerfacecolor='black', marker='o', linestyle='-', color=colors[i], zorder=1)

    plt.scatter(centers[:,0], centers[:,1], marker='o', color='red', s=50, zorder=2)
    plt.savefig(filename)
    plt.show()

def plot_time_taken(average_time):
    plt.figure(figsize=(6,6))
    setup = np.array([2, 4, 6, 8])
    plt.plot([1, 2, 3, 4], average_time, label = "Average Time for Drones")
    plt.plot([1, 2, 3, 4], setup, label = "Set up time")
    plt.plot([1, 2, 3, 4], setup + average_time, label = "Total time in Field")
    plt.title(f"Time in Field based on number of Drones")
    plt.legend()
    plt.xlabel("Number of Drones")
    plt.ylabel("Time (minutes)")
    plt.plot()
    plt.show()