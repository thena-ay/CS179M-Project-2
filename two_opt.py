from nearest_neighbor_search import nearest_neighbor_helper as nnh
from nearest_neighbor_search import nearest_neighbor_search as nns
from validate_file import validate_file as vf
from nearest_neighbor_search import create_dist_matrix
from data_visualization import plot_path_taken as ppt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random
import time
# from pynput import keyboard
import sys

# terminate_program = False



# def on_press(key):
#     global terminate_program
#     try:
#         # Check for enter key press, if so set terminate_program to True
#         if key == keyboard.Key.esc:
#             terminate_program = True
#     except AttributeError:
#         # ignore other keys
#         pass

# # Start the keyboard listener in a non-blocking way
# listener = keyboard.Listener(on_press=on_press)
# listener.start()

# def two_opt(data):
#     n = data.shape[0]
#     # BSF /initial tour from NN
#     dist_mat = create_dist_matrix(data)
#     BSF_dist, BSF_order = nnh(dist_mat.copy(), False)
#     dist = BSF_dist
#     order = BSF_order.copy()
#     # ensure the tour is closed and the distance matches the distance matrix
#     BSF_dist = sum(dist_mat[BSF_order[k], BSF_order[k+1]] for k in range(len(BSF_order) - 1))
#     print(f"\t\t{BSF_dist:.1f}")
#     while not terminate_program:
#         dist, order = two_opt_helper(dist_mat.copy(), order, BSF_dist)
#         if dist < BSF_dist:
#             BSF_dist = dist
#             BSF_order = order
#         _, order = nnh(dist_mat.copy(), True)
#     return BSF_dist, BSF_order

# def two_opt_helper(dist_mat, order, global_best):
#     n = dist_mat.shape[0]
#     # repeated brute force for loop
#     good_delta = True
#     BSF_order = order
#     BSF_dist = sum(dist_mat[BSF_order[k], BSF_order[k+1]] for k in range(len(BSF_order) - 1))

#     while good_delta:
#         good_delta = False
#         for i in range(n):
#             j = i + 2

#             while j < n and not terminate_program:
#                 new_order1 = BSF_order[:i+1]
#                 new_order2 = BSF_order[i+1:j+1]
#                 new_order2 = new_order2[::-1]
#                 new_order3 = BSF_order[j+1:]
#                 new_order_final = new_order1 + new_order2 + new_order3
#                 new_dist = sum(dist_mat[new_order_final[k], new_order_final[k+1]] for k in range(len(new_order_final) - 1))
#                 if new_dist < BSF_dist:
#                     good_delta = True
#                     BSF_order = new_order_final
#                     BSF_dist = new_dist
#                     if BSF_dist < global_best:
#                         global_best = BSF_dist
#                         print(f"\t\t{BSF_dist:.1f}")

#                 j += 1
#             if terminate_program:
#                 break
#     return BSF_dist, BSF_order

def two_opt_test(data, period):
    n = data.shape[0]
    # BSF /initial tour from NN
    dist_mat = create_dist_matrix(data)
    BSF_dist, BSF_order = nnh(dist_mat.copy(), False)
    dist = BSF_dist
    order = BSF_order.copy()
    # ensure the tour is closed and the distance matches the distance matrix
    BSF_dist = sum(dist_mat[BSF_order[k], BSF_order[k+1]] for k in range(len(BSF_order) - 1))
    dist_over_time = []
    time_limit = time.time() + period
    while time.time() < time_limit:
        dist, order, samples = two_opt_helper_test(dist_mat.copy(), order, BSF_dist, time_limit)
        dist_over_time = np.concatenate((dist_over_time, samples), axis=0)
        if dist < BSF_dist:
            BSF_dist = dist
            BSF_order = order
        _, order = nnh(dist_mat.copy(), True)
    return BSF_dist, BSF_order, dist_over_time

def two_opt_helper_test(dist_mat, order, global_best, time_limit):
    n = dist_mat.shape[0]
    # repeated brute force for loop
    good_delta = True
    BSF_order = order
    BSF_dist = sum(dist_mat[BSF_order[k], BSF_order[k+1]] for k in range(len(BSF_order) - 1))
    prev_time = time.time()
    dist_over_time = []
    while good_delta:
        good_delta = False
        for i in range(n):
            j = i + 2
            if time.time() > prev_time + 1:
                dist_over_time.append(BSF_dist)
                prev_time = time.time()
            while j < n and time.time() < time_limit:
                new_order1 = BSF_order[:i+1]
                new_order2 = BSF_order[i+1:j+1]
                new_order2 = new_order2[::-1]
                new_order3 = BSF_order[j+1:]
                new_order_final = new_order1 + new_order2 + new_order3
                new_dist = sum(dist_mat[new_order_final[k], new_order_final[k+1]] for k in range(len(new_order_final) - 1))
                if new_dist < BSF_dist:
                    good_delta = True
                    BSF_order = new_order_final
                    BSF_dist = new_dist

                j += 1
            if time.time() > time_limit:
                break
    return BSF_dist, BSF_order, dist_over_time
