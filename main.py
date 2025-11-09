import numpy as np
from datetime import datetime, timedelta

from TSP.main import validate_file as vf
from TSP.two_opt import two_opt as to
from kmeans import kmeans, assign_points
from visualization import plot_clusters, plot_cluster_paths

def main():
    print("ComputePossibleSolutions")
    input_file = input("Enter the name of the file: ")
    data = vf(input_file)
    print(f"There are {data.shape[0]} nodes: Solutions will be available by {(datetime.now() + timedelta(minutes=5)).strftime("%I:%M %p")}")
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
        # print(best_of)

        buckets = assign_points(data, best_centers)
        res = []
        total_dist = 0

        # calculate distances for best centers
        for k in range(i+1):
            points = data[buckets[k]]
            points = np.vstack((points, best_centers[k]))
            if i < 2:
                dist, order = to(points, 40)
            else:
                dist, order = to(points, 20)
            order = np.array(order)
            order = order -1
            total_dist += dist
            res.append([dist, order, points, best_centers[k]])

        print(f"{i+1}) If you use {i+1} drone(s), the total route will be {total_dist:.1f} meters")

        # output distances for each drone
        count = "i"
        for k in range(i+1):
            print(f"\t{count}. Landing Pad {k+1} should be at ({best_centers[k][0]:.0f}, {best_centers[k][1]:.0f}) serving {len(buckets[k])} locations, route is {res[k][0]:.1f} meters")
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

    final_dist = sum(x[0] for x in final_res[choice-1])
    final_orders = [x[1] for x in final_res[choice-1]]
    final_clusters = [x[2] for x in final_res[choice-1]]
    final_centers = [x[3] for x in final_res[choice-1]]

    # save choices
    for i in range(choice):
        output_file = f"{input_file}_{i+1}_SOLUTION_{final_res[choice-1][i][0]:.0f}.txt"
        print(f"Writing {output_file} to disk")
        np.savetxt(output_file, final_orders[i], fmt="%d")

    # print resultin paths & save file
    plot_cluster_paths(final_clusters, final_centers, final_orders, f"{input_file}_{choice}_SOLUTION_{final_dist:.0f}.png")

if __name__ == '__main__':
    main()