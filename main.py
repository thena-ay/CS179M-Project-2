import numpy as np

from TSP.main import validate_file as vf
from TSP.two_opt import two_opt as to
from kmeans import kmeans, assign_points
from visualization import plot_clusters

def main():
    print("ComputePossibleSolutions")
    input_file = input("Enter the name of the file: ")
    data = vf(input_file)
    # TODO: figure out time
    print(f"There are {data.shape[0]} nodes: Solutions will be available by")
    final_res = []
    # kmeans on each drone
    for i in range(4):
        res = []

        # 10 trials to avoid unlucky scenario
        centers, of = kmeans(data, i+1)
        best_centers, best_of = centers, of
        for j in range(9):
            centers, of = kmeans(data, i+1)
            if of < best_of:
                best_centers, best_of = centers, of

        buckets = assign_points(data, best_centers)
        res = []

        total_dist = 0
        # calculate distances for best centers
        for k in range(i+1):
            points = data[buckets[k]]
            np.insert(points, 0, centers[k])
            dist, order = to(points, 10)
            total_dist += dist
            res.append((dist, order))

        print(f"{i+1}) If you use {i+1} drone(s), the total route will be {total_dist:.1f} meters")

        count = "i"
        for k in range(i+1):
            print(f"{count}. Landing Pad {k+1} should be at {centers[k]}, serving {len(buckets[k])} locations, route is {res[k][0]:.1f} meters")
            count += "i"
            if count == "iiii":
                count = "iv"
        final_res.append(res)

    # get choice
    choice = input("Please select your choice 1 to 4: ")
    is_int = False
    while not is_int:
        try:
            choice = int(choice)
            is_int = True
        except ValueError:
            choice = input("Please select your choice 1 to 4: ")

    # save choices
    for i in range(choice):
        output_file = f"{input_file}_{choice}_SOLUTION_{final_res[choice-1][i][0]:.0f}.txt"
        print(f"Writing {output_file} to disk")
        np.savetxt(output_file, final_res[choice-1][i][1], fmt="%d")
        # TODO: image visualization of all routes (one png)

if __name__ == '__main__':
    main()