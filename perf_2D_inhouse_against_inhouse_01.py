import numpy as np
import Code.test8_updated2D_sd as D1
import Code.test8_updated2D_no_sd as D2
# from scipy.spatial import Delaunay
# import triangle as tr
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

    print("   --- With sub_determinants | With BRIO | New nbr search ---   ")
    start = time.time()
    DT = D1.Delaunay2D(points)
    DT.makeDT()
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 0] = end-start
    else:
        running_times[3*(i-1)+k, 0] = min(end-start, running_times[3*(i-1)+k, 0])
    del DT

    print("   --- With sub_determinants | With BRIO | Old nbr search ---   ")
    start = time.time()
    DT = D1.Delaunay2D(points, new_nbr_search=False)
    DT.makeDT()
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 1] = end-start
    else:
        running_times[3*(i-1)+k, 1] = min(end-start, running_times[3*(i-1)+k, 1])
    del DT

    print("   --- With sub_determinants | Without BRIO | New nbr search ---   ")
    start = time.time()
    DT = D1.Delaunay2D(points, use_BRIO=False)
    DT.makeDT()
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 2] = end-start
    else:
        running_times[3*(i-1)+k, 2] = min(end-start, running_times[3*(i-1)+k, 2])
    del DT

    print("   --- With sub_determinants | Without BRIO | Old nbr search ---   ")
    start = time.time()
    DT = D1.Delaunay2D(points, use_BRIO=False, new_nbr_search=False)
    DT.makeDT()
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 3] = end-start
    else:
        running_times[3*(i-1)+k, 3] = min(end-start, running_times[3*(i-1)+k, 3])
    del DT

    print("   --- Without sub_determinants | With BRIO | New nbr search ---   ")
    start = time.time()
    DT = D2.Delaunay2D(points)
    DT.makeDT()
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 4] = end-start
    else:
        running_times[3*(i-1)+k, 4] = min(end-start, running_times[3*(i-1)+k, 4])
    del DT

    print("   --- Without sub_determinants | With BRIO | Old nbr search ---   ")
    start = time.time()
    DT = D2.Delaunay2D(points, new_nbr_search=False)
    DT.makeDT()
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 5] = end-start
    else:
        running_times[3*(i-1)+k, 5] = min(end-start, running_times[3*(i-1)+k, 5])
    del DT

    print("   --- Without sub_determinants | Without BRIO | New nbr search ---   ")
    start = time.time()
    DT = D2.Delaunay2D(points, use_BRIO=False)
    DT.makeDT()
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 6] = end-start
    else:
        running_times[3*(i-1)+k, 6] = min(end-start, running_times[3*(i-1)+k, 6])
    del DT

    print("   --- Without sub_determinants | Without BRIO | Old nbr search ---   ")
    start = time.time()
    DT = D2.Delaunay2D(points, use_BRIO=False, new_nbr_search=False)
    DT.makeDT()
    end = time.time()
    print("Time taken to make the triangulation : {} s.\n".format(end-start))
    if j == 0:
        running_times[3*(i-1)+k, 7] = end-start
    else:
        running_times[3*(i-1)+k, 7] = min(end-start, running_times[3*(i-1)+k, 7])
    del DT

################################################################################

temp = np.random.rand(2*10)
tempDT = D1.Delaunay2D(temp)
tempDT.makeDT()
del tempDT
del temp

temp = np.random.rand(2*10)
tempDT = D2.Delaunay2D(temp)
tempDT.makeDT()
del tempDT
del temp

################################################################################

lim = int(sys.argv[1])
running_times = np.empty(shape=(3*lim-2, 8))
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
        "with_sd_with_brio_new_nbr_search":running_times[:, 0],
        "with_sd_with_brio_old_nbr_search":running_times[:, 1],
        "with_sd_without_brio_new_nbr_search":running_times[:, 2],
        "with_sd_without_brio_old_nbr_search":running_times[:, 3],
        "without_sd_with_brio_new_nbr_search":running_times[:, 4],
        "without_sd_with_brio_old_nbr_search":running_times[:, 5],
        "without_sd_without_brio_new_nbr_search":running_times[:, 6],
        "without_sd_without_brio_old_nbr_search":running_times[:, 7],
    }
)
df.to_csv('results_2D_inhouse_against_inhouse_01.csv')


plt.grid(True)
for i in range(8):
    plt.loglog(num_points_arr, running_times[:, i], linestyle='--',
               marker='.', linewidth=0.75)

plt.legend([
    r"with sub-determinants $|$ with BRIO $|$ new nbr search ",
    r"with sub-determinants $|$ with BRIO $|$ old nbr search ",
    r"with sub-determinants $|$ without BRIO $|$ new nbr search ",
    r"with sub-determinants $|$ without BRIO $|$ old nbr search ",
    r"without sub-determinants $|$ with BRIO $|$ new nbr search ",
    r"without sub-determinants $|$ with BRIO $|$ old nbr search ",
    r"without sub-determinants $|$ without BRIO $|$ new nbr search ",
    r"without sub-determinants $|$ without BRIO $|$ old nbr search ",
])

plt.xlabel(r"Number of Points", size=10)
plt.ylabel(r"Running Time $(s)$", size=10)
plt.title(r"Results", size=13)

plt.savefig("results_2D_inhouse_against_inhouse_01.png", dpi=300, bbox_to_inches="tight")