import numpy as np
import BTP.experimental.threeD.final_3D_multidimarr as D1
from scipy.spatial import Delaunay
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
import time

rc('text', usetex=True)

################################################################################

def run_it(i, j, k, running_times, num_points_arr):

    np.random.seed(seed=3*i+k)
    if k == 0:
        num_points = 10**i
    elif k == 1:
        num_points = 2*(10**i)
    elif k == 2:
        num_points = 5*(10**i)
    num_points_arr[3*(i-1)+k] = num_points
    print("------------ {} points ------------\n".format(num_points))
    points = np.random.rand(2*num_points)
    points_rest = points.reshape((num_points, 2))

    print("   --- final_3D ---   ")
    start = time.time()
    DT = D1.Delaunay3D(points_rest)
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 0] = end-start
    else:
        running_times[3*(i-1)+k, 0] = min(end-start, running_times[3*(i-1)+k, 0])
    del DT

    print("   --- qhull ---   ")
    start = time.time()
    DT_qhull = Delaunay(points_rest)
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 1] = end-start
    else:
        running_times[3*(i-1)+k, 1] = min(end-start, running_times[3*(i-1)+k, 1])
    del DT_qhull

################################################################################

temp = np.random.rand(2*10).reshape((10, 2))
tempDT = D1.Delaunay3D(temp)
# tempDT.makeDT()
del tempDT
del temp

################################################################################

lim = int(sys.argv[1])
running_times = np.empty(shape=(3*lim-2, 2))
num_points_arr = np.empty(shape=3*lim-2)

for j in range(3):
    print("------------------------------ RUN {} ------------------------------\n".format(j+1))
    for i in range(1, lim+1):
        if i < lim:
            run_it(i, j, 0, running_times, num_points_arr)
            run_it(i, j, 1, running_times, num_points_arr)
            run_it(i, j, 2, running_times, num_points_arr)
        elif i == lim:
            run_it(i, j, 0, running_times, num_points_arr)


df = pd.DataFrame(
    {
        "num_points":num_points_arr,
        "final_3D":running_times[:, 0],
        "qhull":running_times[:, 1],
    }
)
df.to_csv('results_3D_inhouse_vs_rest_01.csv')


plt.grid(True)
for i in range(0, 2):
    plt.loglog(num_points_arr, running_times[:, i], linestyle='--',
               marker='.', linewidth=0.75)

plt.legend([
    r"final\_3D",
    r"Qhull"
])

plt.xlabel(r"Number of Points", size=10)
plt.ylabel(r"Running Time $(s)$", size=10)
plt.suptitle(r"Results", size=13)

plt.savefig("results_3D_inhouse_vs_rest_01.png", dpi=300, bbox_to_inches="tight")