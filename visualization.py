import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_clusters(buckets, data, centers):
    k = centers.shape[0]
    plt.figure(figsize=(19.2, 14.4), dpi=100)
    colors = ['red', 'blue', 'green', 'purple', 'teal', 'orange', 'pink']
    for i in range(k):
        points = data[buckets[i]]
        sns.scatterplot(x=points[:,0], y=points[:, 1], color=colors[i], markers='o')
    sns.scatterplot(x=centers[:,0], y=centers[:,1], color='black', markers='o', s=50)
    plt.title('Locations')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()