import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(buckets, data, centers):
    k = len(centers)
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=centers[:,0], y=centers[:,1], color='black', markers='o')
    colors = ['red', 'blue', 'green', 'purple']
    for i in range(k):
        points = data[buckets[i]]
        sns.scatterplot(x=points[:,0], y=points[:, 1], color=colors[i], markers='o')
    plt.title('Locations')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()