import numba
import numpy as np
from math import sqrt
from hausdorff import hausdorff_distance

# two random 2D arrays (second dimension must match)
np.random.seed(0)
X = np.random.random((1,4,64,128,128))
Y = np.random.random((1,4,64,128,128))

# write your own crazy custom function here
# this function should take two 1-dimensional arrays as input
# and return a single float value as output.
# @numba.jit(nopython=True, fastmath=True)
# def custom_dist(array_x, array_y):
#     n = array_x.shape[0]
#     ret = 0.
#     for i in range(n):
#         ret += (array_x[i]-array_y[i])**2
#     return sqrt(ret)

# print(f"Hausdorff custom euclidean test: {hausdorff_distance(X, Y, distance=custom_dist)}")

# # a real crazy custom function
# @numba.jit(nopython=True, fastmath=True)
# def custom_dist(array_x, array_y):
#     n = array_x.shape[0]
#     ret = 0.
#     for i in range(n):
#         ret += (array_x[i]-array_y[i])**3 / (array_x[i]**2 + array_y[i]**2 + 0.1)
#     return ret

# print(f"Hausdorff custom crazy test: {hausdorff_distance(X, Y, distance=custom_dist)}")

from medpy.metric.binary import hd
print(hd(X,Y))