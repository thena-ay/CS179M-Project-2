from nearest_neighbor_search import nearest_neighbor_search as nns
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

def plot_points(filename):
    coords = np.loadtxt(filename)
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=coords[:,0], y=coords[:, 1])
    plt.scatter(x=coords[0,0], y=coords[0,1], color='red', marker='o', s=50, label='Landing Pad')
    plt.title('Locations')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

def plot_path_taken(data, order, filename):
    reordered_data = data[order]
    plt.figure(figsize=(6,6))
    plt.plot(reordered_data[:, 0], reordered_data[:,1], marker = 'o', markerfacecolor = 'blue', linestyle = '-', color = "green")
    plt.plot(data[0,0], data[0,1], marker = 'o', markerfacecolor = 'red', color = "green")
    plt.title('Order taken')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    plt.savefig(filename)

def plot_over_time(distances, algorithms, time, trials, scenario):
    plt.figure(figsize=(6,6))
    for distance, algorithm in zip(distances, algorithms):
        plt.plot(np.arange(0, distance.shape[0]), np.average(distance, axis=0), label = algorithm)
    plt.title(f'Average best so far distance over {time} seconds, {trials} trials on {scenario} data')
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance(Lower is Better)')
    plt.show()

def plot_different_squares(distances, algorithms, nodes, time, trials):
    fig, ax = plt.subplots(figsize=(8,8))
    for distance, algorithm in zip(distances, algorithms):
        ax.plot(nodes, distance, label = algorithm, marker='o', ls = '')
    ideal_dist = []
    for i in distances[0]:
        ideal_dist.append(4)
    ax.plot(nodes, ideal_dist, label = "Ideal Distance", linestyle='--', color='black')
    title = f'Average best so far distance at {time} seconds, {trials} trials on Unit Square data'
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_xticks(nodes)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Distance(Lower is Better)')
    plt.show()

if __name__ == "__main__":
    algorithms = [
        "Random Search", 
        "Nearest Neighbor Search", 
        "2-OPT"
        ]
    nodes = [64, 128, 256, 512, 1024]
    # Plot distance over time for 256 Cashew for all algorithms
    # TO_distances = np.loadtxt("res/distances/TO/100trials_256Cashew_100seconds_trial100.txt")
    # NNS_distances = np.loadtxt("res/distances/NNS/100trials_256Cashew_100seconds_trial100.txt")
    # RS_distances = np.loadtxt("res/distances/RS/100trials_256Cashew_100seconds_trial100.txt")
    # distances = [
    #     RS_distances,
    #     NNS_distances,
    #     TO_distances
    #     ]
    # plot_over_time(distances, algorithms, 100, 100, "256 Cashew")
    # Plot unit square optimality results
    # RS_30trials_20seconds = np.loadtxt("res/unit_square_optimality/RS_30trials_20seconds.txt")
    # NNS_30trials_20seconds = np.loadtxt("res/unit_square_optimality/NNS_30trials_20seconds.txt")
    # TO_30trials_20seconds = np.loadtxt("res/unit_square_optimality/TO_30trials_20seconds.txt")
    # distances = [
    #     RS_30trials_20seconds,
    #     NNS_30trials_20seconds,
    #     TO_30trials_20seconds
    #     ]
    # plot_different_squares(distances, algorithms, nodes, 20, 30)

    # Plot points and path taken for a specific file
    unit_square_1024 = np.loadtxt("data/unit_squares/1024/45.txt")
    NNS_dist, NNS_order = nns(unit_square_1024, period=100)
    TO_dist, TO_order = to(unit_square_1024, period=100)
    plot_path_taken(unit_square_1024, NNS_order, "res/path_visuals/unit_square_1024_NNS.png")
    plot_path_taken(unit_square_1024, TO_order, "res/path_visuals/unit_square_1024_TO.png")


