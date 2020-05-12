import numpy as np
import BTP.core.final_2D_multidimarr as D1
import BTP.core.final_2D_robust_multidimarr as D2
import BTP.experimental.adjusted_predicates.TwoD.final_2D_robust_multidimarr as D3
from scipy.spatial import Delaunay
import triangle as tr
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

    print("   --- final_2D (non-robust) ---   ")
    start = time.time()
    DT = D1.Delaunay2D(points_rest)
    DT.makeDT()
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 0] = end-start
    else:
        running_times[3*(i-1)+k, 0] = min(end-start, running_times[3*(i-1)+k, 0])
    del DT

    print("   --- final_2D_robust (no static filters) ---   ")
    start = time.time()
    DT = D2.Delaunay2D(points_rest)
    DT.makeDT()
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 1] = end-start
    else:
        running_times[3*(i-1)+k, 1] = min(end-start, running_times[3*(i-1)+k, 1])
    del DT

    print("   --- final_2D_robust (with static filters) ---   ")
    start = time.time()
    DT = D3.Delaunay2D(points_rest)
    DT.makeDT()
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 2] = end-start
    else:
        running_times[3*(i-1)+k, 2] = min(end-start, running_times[3*(i-1)+k, 2])
    del DT

    print("   --- triangle (incremental) ---   ")
    tri = {"vertices": points_rest}
    start = time.time()
    DT_triangle = tr.triangulate(tri, opts='i')
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 3] = end-start
    else:
        running_times[3*(i-1)+k, 3] = min(end-start, running_times[3*(i-1)+k, 3])
    del tri
    del DT_triangle

    print("   --- triangle (divide-and-conquer) ---   ")
    tri = {"vertices": points_rest}
    start = time.time()
    DT_triangle = tr.triangulate(tri)
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 4] = end-start
    else:
        running_times[3*(i-1)+k, 4] = min(end-start, running_times[3*(i-1)+k, 4])
    del tri
    del DT_triangle

    print("   --- qhull ---   ")
    start = time.time()
    DT_qhull = Delaunay(points_rest)
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 5] = end-start
    else:
        running_times[3*(i-1)+k, 5] = min(end-start, running_times[3*(i-1)+k, 5])
    del DT_qhull

################################################################################

temp = np.random.rand(2*10).reshape((10, 2))
tempDT = D1.Delaunay2D(temp)
tempDT.makeDT()
del tempDT
del temp

temp = np.random.rand(2*10).reshape((10, 2))
tempDT = D2.Delaunay2D(temp)
tempDT.makeDT()
del tempDT
del temp

temp = np.random.rand(2*10).reshape((10, 2))
tempDT = D3.Delaunay2D(temp)
tempDT.makeDT()
del tempDT
del temp

################################################################################

lim = int(sys.argv[1])
running_times = np.empty(shape=(3*lim-2, 6))
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
        "final_2D--non_robust)":running_times[:, 0],
        "final_2D_robust--no_static_filters)":running_times[:, 1],
        "final_2D_robust--with_static_filters)":running_times[:, 2],
        "triangle--incremental":running_times[:, 3],
        "triangle--dnc":running_times[:, 4],
        "qhull":running_times[:, 5],
    }
)
df.to_csv('results_2D_inhouse_vs_rest_01.csv')


plt.grid(True)
for i in range(6):
    plt.loglog(num_points_arr, running_times[:, i], linestyle='--',
               marker='.', linewidth=0.75)

plt.legend([
    r"final\_2D (non-robust)",
    r"final\_2D\_robust (no static filters)",
    r"final\_2D\_robust (with static filters)",
    r"Triangle (incremental)",
    r"Triangle (divide-and-conquer)",
    r"Qhull"
])

plt.xlabel(r"Number of Points", size=10)
plt.ylabel(r"Running Time $(s)$", size=10)
plt.title(r"Results", size=13)

plt.savefig("results_2D_inhouse_vs_rest_01.png", dpi=300, bbox_to_inches="tight")