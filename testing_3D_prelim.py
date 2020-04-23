import numpy as np
from scipy.spatial import Delaunay
import BTP.core.test7_3D_sd as t3
import BTP.tools.DelaunayTest as DT_test
import time


########## testing intersection algorithm ##########
'''
points = np.array([
    [0, 0, 0],
    [10, 0 ,0],
    [0, 10, 0],
    [0, 0, 10],
    [10, 10, 10],
    [10, 10, 0]
], dtype=np.float64)
points = points.reshape(3*len(points))


vertices = np.array([
    [0, 1, 2, 3],
    [4, 0, 3, 2]
], dtype=np.float64)
vertices = vertices.reshape(4*len(vertices))
DT_test.check_tri_intersections(points, vertices)
# The above should intersect


vertices = np.array([
    [0, 1, 2, 3],
    [2, 1, 5, 3]
], dtype=np.float64)
vertices = vertices.reshape(4*len(vertices))
DT_test.check_tri_intersections(points, vertices)
# The above should not intersect
'''
####################################################



#################### priming numba ####################

points_1 = np.array([
    [0, 0, 0],
    [0, 10, 0],
    [10, 0, 0],
    [0, 0, 10],
    [2, 2, 0],
    # [0, 0, -3]
], dtype=np.float64)
points_2 = points_1.reshape(3*len(points_1))

DT3 = t3.Delaunay3D(points_2)
DT3.makeDT(printTime=False)
points_1, vertices_1, sd_1 = DT3.exportDT()
DT_test.TestDelaunay3D_det(points_1, vertices_1)
DT_test.TestDelaunay3D_sd(points_1, vertices_1)
# DT3.export_VTK()

# DT3 = t3_org.Delaunay3D(points_2)
# DT3.makeDT(printTime=False)
# points_2, vertices_2, sd_2 = DT3.exportDT()
# DT_test.TestDelaunay3D_det(points_2, vertices_2)
# DT_test.TestDelaunay3D_sd(points_2, vertices_2)
# DT3.export_VTK()

#######################################################




############################## the main experiment ##############################


N = 5*(10**2)
num = 10
# flags = np.empty(shape=num, dtype=np.bool_)
pathCases = np.empty(shape=num, dtype=np.int64)

for i in np.arange(num):
    print("\n____________________ seed = {} ____________________".format(i))
    np.random.seed(seed=i)

    ########################################
    points_2 = 10*np.random.rand(3*N)
    # points_2[0::1] *= 10*np.random.rand(1)[0]
    points_1 = points_2.reshape((N, 3))

    print("\n Number of points : " + str(N))

    ########################################
    print("\n QHULL")
    start = time.time()
    tets = Delaunay(points_1)
    end = time.time()

    points_0 = points_2.copy()
    vertices_0 = tets.simplices.ravel()
    neighbour_ID_0 = tets.neighbors

    print("Time taken by scipy.spatial.Delaunay (QHull) : {} s.".format(end-start))
    print("Number of tets : {} \n".format(len(tets.simplices)))

    DT_test.TestDelaunay3D_det(points_0, vertices_0)
    DT_test.TestDelaunay3D_sd(points_0, vertices_0)

    # temp = DT_test.TestNeighbours_qhull(neighbour_ID_0)
    # DT_test.check_tri_intersections(points_0, vertices_0)

    ########################################
    print("\n UPDATED CODE")
    start = time.time()
    DT3 = t3.Delaunay3D(points_2)
    pathCases[i] = DT3.makeDT(printTime=False, returnPathCases=True)
    end = time.time()

    # points_1, vertices_1, sd_1, neighbour_ID_1 = DT3.exportDT()
    points_1, vertices_1, sd_1 = DT3.exportDT()

    print("Time taken by test7_3D_sd : {} s.".format(end-start))
    print("Number of tets (including ghost tets) : " + str(DT3.num_tet))
    print("Number of real tets : {} \n".format(int(len(vertices_1)*0.25)))

    DT_test.TestDelaunay3D_det(points_1, vertices_1)
    DT_test.TestDelaunay3D_sd(points_1, vertices_1)

    # neighbour_ID_1 = DT3.neighbour_ID[0:4*DT3.num_tet]
    # DT_test.TestNeighbours_qhull(neighbour_ID_1.reshape((int(0.25*len(neighbour_ID_1)), 4)))
    # flags[i] = DT_test.TestNeighbours(neighbour_ID_1)
    # DT_test.check_tri_intersections(points_1, vertices_1)

    # DT3.export_VTK()

    '''
    ########################################
    print("\n ORIGINAL CODE")
    DT3 = t3_org.Delaunay3D(points_2)
    DT3.makeDT(printTime=True)

    points_2, vertices_2, sd_2 = DT3.exportDT()

    print("Number of tets (including ghost tets) : " + str(DT3.num_tet))
    print("Number of real tets : " + str(int(len(vertices_2)*0.25)))

    DT_test.TestDelaunay3D(points_2, vertices_2)

    neighbour_ID_2 = DT3.neighbour_ID[0:4*DT3.num_tet]
    DT_test.TestNeighbours(neighbour_ID_2)
    DT_test.check_tri_intersections(points_2, vertices_2)

    # DT3.export_VTK()

    ########################################

    print("\n")

    flag_v = np.all(vertices_1 == vertices_2)
    print("'vertices_ID' same for both? " + str(flag_v))
    if flag_v == False:
        idx = vertices_1 == vertices_2
        print("vertices_1 : " + str(vertices_1[idx]))
        print("vertices_2 : " + str(vertices_2[idx]) + str("\n"))

    flag_p = np.all(points_1 == points_2)
    print("'points' same for both? " + str(flag_p))
    if flag_p == False:
        idx = points_1 == points_2
        print("points_1 == points_2 : " + str(points_1 == points_2))
        print("points_1 : " + str(points_1[idx]))
        print("points_2 : " + str(points_2[idx]))
'''

print("\n___________________________________________________")

# okay_seeds = len(np.where(flags == True)[0])
# bad_seeds = num - okay_seeds
# print("number of good seeds : {}".format(okay_seeds))
# print("number of bad seeds : {}\n".format(bad_seeds))


mean_pathCases = np.mean(pathCases)
percentage_pathCases = 100*mean_pathCases/(N-4)
print("Average number of pathCases : {}".format(mean_pathCases))
print("pathCases as percentage of total number of insertions : {} % \n".format(percentage_pathCases))

#################################################################################