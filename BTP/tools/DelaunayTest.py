import numpy as np
from numba import njit, prange


@njit(parallel=True)
def TestDelaunay3Dnjit_sd(points, vertices, non_Delaunay_tets):

    num_tets = int(len(vertices)*0.25)
    num_points = int(len(points)/3)

    for t in prange(num_tets):
        a_x = points[3*vertices[4*t+0]+0]
        a_y = points[3*vertices[4*t+0]+1]
        a_z = points[3*vertices[4*t+0]+2]

        abx_ = points[3*vertices[4*t+1]+0]-a_x  # b_x-a_x
        aby_ = points[3*vertices[4*t+1]+1]-a_y  # b_y-a_y
        abz_ = points[3*vertices[4*t+1]+2]-a_z  # b_z-a_z
        acx_ = points[3*vertices[4*t+2]+0]-a_x  # c_x-a_x
        acy_ = points[3*vertices[4*t+2]+1]-a_y  # c_y-a_y
        acz_ = points[3*vertices[4*t+2]+2]-a_z  # c_z-a_z
        adx_ = points[3*vertices[4*t+3]+0]-a_x  # d_x-a_x
        ady_ = points[3*vertices[4*t+3]+1]-a_y  # d_y-a_y
        adz_ = points[3*vertices[4*t+3]+2]-a_z  # d_z-a_z
        normsq_ba = abx_**2 + aby_**2 + abz_**2
        normsq_ca = acx_**2 + acy_**2 + acz_**2
        normsq_da = adx_**2 + ady_**2 + adz_**2

        A = normsq_ba*(acy_*adz_-ady_*acz_) - \
            normsq_ca*(aby_*adz_-ady_*abz_) + \
            normsq_da*(aby_*acz_-acy_*abz_)
        A *= -1.
        B = normsq_ba*(acx_*adz_-adx_*acz_) - \
            normsq_ca*(abx_*adz_-adx_*abz_) + \
            normsq_da*(abx_*acz_-acx_*abz_)
        C = normsq_ba*(acx_*ady_-adx_*acy_) - \
            normsq_ca*(abx_*ady_-adx_*aby_) + \
            normsq_da*(abx_*acy_-acx_*aby_)
        C *= -1.
        D = (abz_)*(acx_*ady_-adx_*acy_) - \
            (acz_)*(abx_*ady_-adx_*aby_) + \
            (adz_)*(abx_*acy_-acx_*aby_)

        for p in range(num_points):
            if p == vertices[4*t+0]:
                continue
            elif p == vertices[4*t+1]:
                continue
            elif p == vertices[4*t+2]:
                continue
            elif p == vertices[4*t+3]:
                continue
            else:
                point_x = points[3*p+0]
                point_y = points[3*p+1]
                point_z = points[3*p+2]

                det = (point_x-a_x)*A+(point_y-a_y)*B+(point_z-a_z)*C+(
                    (point_x-a_x)**2 + (point_y-a_y)**2 + (point_z-a_z)**2
                )*D

                if det > 0:
                    non_Delaunay_tets[t] = False
                    break

    return


def TestDelaunay3D_sd(points, vertices):

    num_tets = int(len(vertices)*0.25)
    non_Delaunay_tets = np.ones(shape=num_tets, dtype=np.bool_)

    TestDelaunay3Dnjit_sd(
        points,
        vertices,
        non_Delaunay_tets
    )

    num_nonDelaunay_tets = len(np.where(non_Delaunay_tets == np.bool_(0))[0])

    print("***Testing with sub-determinants***")
    if num_nonDelaunay_tets > 0:
        print("This is not a Delaunay tetrahedralisation.")
        print(
            "Number of non-Delaunay tets : {} \n".format(num_nonDelaunay_tets)
        )
    else:
        print("This is a Delaunay tetrahedralisation. \n")


@njit(parallel=True)
def TestDelaunay3Dnjit_det(points, vertices, non_Delaunay_tets):

    num_tets = int(len(vertices)*0.25)
    num_points = int(len(points)/3)

    for t in prange(num_tets):
        temp_arr = np.ones(shape=(5, 5), dtype=np.float64)

        for i in range(4):
            temp_arr[i, 3] = 0
            for j in range(3):
                temp_arr[i, j] = points[3*vertices[4*t+i]+j]
                temp_arr[i, 3] += points[3*vertices[4*t+i]+j]**2

        for p in range(num_points):
            if p == vertices[4*t+0]:
                continue
            elif p == vertices[4*t+1]:
                continue
            elif p == vertices[4*t+2]:
                continue
            elif p == vertices[4*t+3]:
                continue
            else:
                temp_arr[4, 3] = 0
                for j in range(3):
                    temp_arr[4, j] = points[3*p+j]
                    temp_arr[4, 3] += points[3*p+j]**2

                det = np.linalg.det(temp_arr)

                if det > 0:
                    non_Delaunay_tets[t] = False
                    break

    return


def TestDelaunay3D_det(points, vertices):

    num_tets = int(len(vertices)*0.25)
    non_Delaunay_tets = np.ones(shape=num_tets, dtype=np.bool_)

    TestDelaunay3Dnjit_det(
        points,
        vertices,
        non_Delaunay_tets,
    )

    num_nonDelaunay_tets = len(np.where(non_Delaunay_tets == np.bool_(0))[0])

    print("***Testing with 5x5 determinant***")
    if num_nonDelaunay_tets > 0:
        print("This is not a Delaunay tetrahedralisation.")
        print(
            "Number of non-Delaunay tets : {} \n".format(num_nonDelaunay_tets)
        )
    else:
        print("This is a Delaunay tetrahedralisation. \n")


@njit(parallel=True)
def TestNeighbours_njit(neighbour_ID, truth_array):

    num_tets = int(len(neighbour_ID)*0.25)

    for t in prange(num_tets):
        for j in range(4):
            neighbour = neighbour_ID[4*t+j]//4
            op_idx = neighbour_ID[4*t+j] % 4
            temp = neighbour_ID[4*neighbour+op_idx]
            if temp != 4*t+j:
                truth_array[4*t+j] = 0


def TestNeighbours(neighbour_ID):

    num_tets = int(len(neighbour_ID)*0.25)
    truth_array = np.ones(shape=4*num_tets, dtype=np.bool_)

    TestNeighbours_njit(neighbour_ID, truth_array)

    temp = truth_array.reshape((num_tets, 4))
    temp = np.apply_along_axis(np.all, 1, temp)
    num_prob_tets = num_tets-np.sum(temp)
    flag = np.all(truth_array)

    if not flag:
        print("Triangulation has neighbour errors.")
        print("Number of problematic tets : {} \n".format(num_prob_tets))
        # print(truth_array.reshape((num_tets, 4)))
        return False
    else:
        print("No neighbour errors. \n")
        return True


@njit
def FindRealTets(vertices_ID, gv, num_tet_n):

    counter = 0
    for i in np.arange(0, num_tet_n):
        if vertices_ID[4*i] == gv:
            counter += 1
            break
        elif vertices_ID[4*i+1] == gv:
            counter += 1
            break
        elif vertices_ID[4*i+2] == gv:
            counter += 1
            break
        elif vertices_ID[4*i+3] == gv:
            counter += 1

    return counter


@njit
def cross_pdt(a_x, a_y, a_z, b_x, b_y, b_z):
    return a_y*b_z-a_z*b_y, -a_x*b_z+a_z*b_x, a_x*b_y-a_y*b_x


@njit(parallel=True)
def check_tri_intersections_njit(points, vertices, intersected_arr):
    # Assuming 'vertices' only contains real tets

    num_tets = int(0.25*len(vertices))
    num_points = int(len(points)/3)

    for tet1 in prange(num_tets):
        # if intersected_arr[tet1] == True:
        #     continue
        for i in range(4):
            v1_0 = int(vertices[4*tet1 + (i+0) % 4])
            v1_1 = int(vertices[4*tet1 + (i+1) % 4])
            v1_2 = int(vertices[4*tet1 + (i+2) % 4])

            tri_intersect = False

            for tet2 in range(tet1+1, num_tets):
                for j in range(4):
                    v2_0 = int(vertices[4*tet2 + (j+0) % 4])
                    v2_1 = int(vertices[4*tet2 + (j+1) % 4])
                    v2_2 = int(vertices[4*tet2 + (j+2) % 4])

                    tri_intersect = False
                    same_tri = False
                    nbr_tri = False

                    if v1_0 == v2_0:
                        if v1_1 == v2_1 and v1_2 == v2_2:
                            same_tri = True
                        elif v1_1 == v2_2 and v1_2 == v2_1:
                            same_tri = True
                        elif v1_1 == v2_1 or v1_1 == v2_2:
                            nbr_tri = True
                        elif v1_2 == v2_1 or v1_2 == v2_2:
                            nbr_tri = True
                    elif v1_0 == v2_1:
                        if v1_1 == v2_0 and v1_2 == v2_2:
                            same_tri = True
                        elif v1_1 == v2_2 and v1_2 == v2_0:
                            same_tri = True
                        elif v1_1 == v2_0 or v1_1 == v2_2:
                            nbr_tri = True
                        elif v1_2 == v2_0 or v1_2 == v2_2:
                            nbr_tri = True
                    elif v1_0 == v2_2:
                        if v1_1 == v2_0 and v1_2 == v2_1:
                            same_tri = True
                        elif v1_1 == v2_1 and v1_2 == v2_0:
                            same_tri = True
                        elif v1_1 == v2_0 or v1_1 == v2_1:
                            nbr_tri = True
                        elif v1_2 == v2_0 or v1_2 == v2_1:
                            nbr_tri = True

                    if same_tri is False and nbr_tri is False:
                        v1_0x = points[3*v1_0+0]
                        v1_0y = points[3*v1_0+1]
                        v1_0z = points[3*v1_0+2]

                        v1_1x = points[3*v1_1+0]
                        v1_1y = points[3*v1_1+1]
                        v1_1z = points[3*v1_1+2]

                        v1_2x = points[3*v1_2+0]
                        v1_2y = points[3*v1_2+1]
                        v1_2z = points[3*v1_2+2]

                        v2_0x = points[3*v2_0+0]
                        v2_0y = points[3*v2_0+1]
                        v2_0z = points[3*v2_0+2]

                        v2_1x = points[3*v2_1+0]
                        v2_1y = points[3*v2_1+1]
                        v2_1z = points[3*v2_1+2]

                        v2_2x = points[3*v2_2+0]
                        v2_2y = points[3*v2_2+1]
                        v2_2z = points[3*v2_2+2]

                        N2_x, N2_y, N2_z = cross_pdt(
                            v2_1x-v2_0x,
                            v2_1y-v2_0y,
                            v2_1z-v2_0z,
                            v2_2x-v2_0x,
                            v2_2y-v2_0y,
                            v2_2z-v2_0z
                        )

                        N2_norm = (N2_x**2 + N2_y**2 + N2_z**2)**0.5
                        N2_x /= N2_norm
                        N2_y /= N2_norm
                        N2_z /= N2_norm
                        # d2 = -(N2_x*v2_0x+N2_y*v2_0y+N2_z*v2_0z)

                        line_intersect = True
                        in_same_plane = False

                        # dv1_0 = (N2_x*v1_0x+N2_y*v1_0y+N2_z*v1_0z) + d2
                        # dv1_1 = (N2_x*v1_1x+N2_y*v1_1y+N2_z*v1_1z) + d2
                        # dv1_2 = (N2_x*v1_2x+N2_y*v1_2y+N2_z*v1_2z) + d2

                        dv1_0 = N2_x*(v1_0x-v2_0x) + N2_y*(v1_0y-v2_0y) + \
                            N2_z*(v1_0z-v2_0z)
                        dv1_1 = N2_x*(v1_1x-v2_0x) + N2_y*(v1_1y-v2_0y) + \
                            N2_z*(v1_1z-v2_0z)
                        dv1_2 = N2_x*(v1_2x-v2_0x) + N2_y*(v1_2y-v2_0y) + \
                            N2_z*(v1_2z-v2_0z)

                        if dv1_0 >= 0 and dv1_1 >= 0 and dv1_2 >= 0:
                            if dv1_0 == 0 and dv1_1 == 0 and dv1_2 == 0:
                                in_same_plane = True
                            line_intersect = False
                        elif dv1_0 <= 0 and dv1_1 <= 0 and dv1_2 <= 0:
                            if dv1_0 == 0 and dv1_1 == 0 and dv1_2 == 0:
                                in_same_plane = True
                            line_intersect = False
                        else:
                            N1_x, N1_y, N1_z = cross_pdt(
                                v1_1x-v1_0x,
                                v1_1y-v1_0y,
                                v1_1z-v1_0z,
                                v1_2x-v1_0x,
                                v1_2y-v1_0y,
                                v1_2z-v1_0z
                            )

                            N1_norm = (N1_x**2 + N1_y**2 + N1_z**2)**0.5
                            N1_x /= N1_norm
                            N1_y /= N1_norm
                            N1_z /= N1_norm
                            # d1 = -(N1_x*v1_0x+N1_y*v1_0y+N1_z*v1_0z)

                            # dv2_0 = (N1_x*v2_0x+N1_y*v2_0y+N1_z*v2_0z) + d1
                            # dv2_1 = (N1_x*v2_1x+N1_y*v2_1y+N1_z*v2_1z) + d1
                            # dv2_2 = (N1_x*v2_2x+N1_y*v2_2y+N1_z*v2_2z) + d1

                            dv2_0 = N1_x*(v2_0x-v1_0x) + N1_y*(v2_0y-v1_0y) + \
                                N1_z*(v2_0z-v1_0z)
                            dv2_1 = N1_x*(v2_1x-v1_0x) + N1_y*(v2_1y-v1_0y) + \
                                N1_z*(v2_1z-v1_0z)
                            dv2_2 = N1_x*(v2_2x-v1_0x) + N1_y*(v2_2y-v1_0y) + \
                                N1_z*(v2_2z-v1_0z)

                            # don't need to check if all of these are zero
                            # together, since the case of the triangles being
                            # in the same plane would have been handled in the
                            # first two if-elif statements
                            if dv2_0 >= 0 and dv2_1 >= 0 and dv2_2 >= 0:
                                line_intersect = False
                            elif dv2_0 <= 0 and dv2_1 <= 0 and dv2_2 <= 0:
                                line_intersect = False

                        if line_intersect is True:
                            Dx, Dy, Dz = cross_pdt(
                                N1_x,
                                N1_y,
                                N1_z,
                                N2_x,
                                N2_y,
                                N2_z
                            )
                            D_norm = (Dx**2 + Dy**2 + Dz**2)**0.5
                            Dx /= D_norm
                            Dy /= D_norm
                            Dz /= D_norm

                            sign0 = dv1_0*dv1_1
                            sign1 = dv1_1*dv1_2
                            if sign0 > 0:
                                # interchanging v1_1 and v1_2
                                v1_1x, v1_2x = v1_2x, v1_1x
                                v1_1y, v1_2y = v1_2y, v1_1y
                                v1_1z, v1_2z = v1_2z, v1_1z
                                dv1_1, dv1_2 = dv1_2, dv1_1
                            elif sign1 > 0:
                                # interchanging v1_1 and v1_0
                                v1_1x, v1_0x = v1_0x, v1_1x
                                v1_1y, v1_0y = v1_0y, v1_1y
                                v1_1z, v1_0z = v1_0z, v1_1z
                                dv1_1, dv1_0 = dv1_0, dv1_1

                            p_v1_0 = Dx*v1_0x + Dy*v1_0y + Dz*v1_0z
                            p_v1_1 = Dx*v1_1x + Dy*v1_1y + Dz*v1_1z
                            p_v1_2 = Dx*v1_2x + Dy*v1_2y + Dz*v1_2z
                            t1_1 = p_v1_0 + (p_v1_1-p_v1_0)*dv1_0/(dv1_0-dv1_1)
                            t1_2 = p_v1_2 + (p_v1_1-p_v1_2)*dv1_2/(dv1_2-dv1_1)
                            if t1_1 > t1_2:
                                t1_1, t1_2 = t1_2, t1_1

                            sign0 = dv2_0*dv2_1
                            sign1 = dv2_1*dv2_2
                            if sign0 > 0:
                                # interchanging v1_1 and v1_2
                                v2_1x, v2_2x = v2_2x, v2_1x
                                v2_1y, v2_2y = v2_2y, v2_1y
                                v2_1z, v2_2z = v2_2z, v2_1z
                                dv2_1, dv2_2 = dv2_2, dv2_1
                            elif sign1 > 0:
                                # interchanging v1_1 and v1_0
                                v2_1x, v2_0x = v2_0x, v2_1x
                                v2_1y, v2_0y = v2_0y, v2_1y
                                v2_1z, v2_0z = v2_0z, v2_1z
                                dv2_1, dv2_0 = dv2_0, dv2_1

                            p_v2_0 = Dx*v2_0x + Dy*v2_0y + Dz*v2_0z
                            p_v2_1 = Dx*v2_1x + Dy*v2_1y + Dz*v2_1z
                            p_v2_2 = Dx*v2_2x + Dy*v2_2y + Dz*v2_2z
                            t2_1 = p_v2_0 + (p_v2_1-p_v2_0)*dv2_0/(dv2_0-dv2_1)
                            t2_2 = p_v2_2 + (p_v2_1-p_v2_2)*dv2_2/(dv2_2-dv2_1)
                            if t2_1 > t2_2:
                                t2_1, t2_2 = t2_2, t2_1

                            if t1_1 == t2_1 or t1_2 == t2_2:
                                tri_intersect = True
                                # break
                            elif t1_1 < t2_1 and t2_1 < t1_2:
                                tri_intersect = True
                                # break
                            elif t2_1 < t1_1 and t1_1 < t2_2:
                                tri_intersect = True
                                # break

                        # elif in_same_plane == True:
                        #     # print("in_same_plane")
                        #     pass

                if tri_intersect is True:
                    break

            if tri_intersect is True:
                intersected_arr[tet1] = True
                intersected_arr[tet2] = True
                break


def check_tri_intersections(points, vertices):

    num_tets = int(0.25*len(vertices))
    intersected_arr = np.zeros(shape=num_tets, dtype=np.bool_)

    check_tri_intersections_njit(points, vertices, intersected_arr)

    have_intersection = np.any(intersected_arr)
    intersected_tets = np.where(intersected_arr == np.bool_(1))[0]
    num_intersecting_tets = len(intersected_tets)

    if have_intersection:
        print("Some tets intersect.")
        print("Number of intersected tets : "+str(num_intersecting_tets)+"\n")
    else:
        print("No intersections. \n")


@njit(parallel=True)
def TestNeighbours_njit_qhull(neighbour_ID, truth_array):

    num_tets = int(len(neighbour_ID))

    for t in prange(num_tets):
        for j in range(4):
            neighbour = neighbour_ID[t, j]
            if neighbour != -1:
                for i in range(4):
                    temp = neighbour_ID[neighbour, i]
                    if temp == t:
                        truth_array[t, j] = 1
                        break
            else:
                truth_array[t, j] = 1


def TestNeighbours_qhull(neighbour_ID):

    num_tets = int(len(neighbour_ID))
    truth_array = np.zeros(shape=(num_tets, 4), dtype=np.bool_)

    TestNeighbours_njit_qhull(neighbour_ID, truth_array)

    temp = np.apply_along_axis(np.all, 1, truth_array)
    num_prob_tets = num_tets - np.sum(temp)
    flag = np.all(truth_array)

    if not flag:
        print("Triangulation has neighbour errors.")
        print("Number of problematic tets : {} \n".format(num_prob_tets))
        return False
    else:
        print("No neighbour errors. \n")
        return True


@njit(parallel=True)
def TestDelaunay2Dnjit_det(points, vertices, non_Delaunay_tri):

    num_tri = int(len(vertices)*0.25)
    num_points = int(len(points)/2)

    for t in prange(num_tri):
        temp_arr = np.ones(shape=(4, 4), dtype=np.float64)

        for i in range(3):
            temp_arr[i, 2] = 0
            for j in range(2):
                temp_arr[i, j] = points[2*vertices[3*t+i]+j]
                temp_arr[i, 2] += points[2*vertices[3*t+i]+j]**2

        for p in range(num_points):
            if p == vertices[3*t+0]:
                continue
            elif p == vertices[3*t+1]:
                continue
            elif p == vertices[3*t+2]:
                continue
            else:
                for j in range(2):
                    temp_arr[3, 2] = 0
                    for j in range(2):
                        temp_arr[3, j] = points[2*p+j]
                        temp_arr[3, 2] += points[2*p+j]**2

                det = np.linalg.det(temp_arr)

                if det > 0:
                    non_Delaunay_tri[t] = False
                    break

    return


def TestDelaunay2D_det(points, vertices):

    num_tri = int(len(vertices)/3)
    non_Delaunay_tri = np.ones(shape=num_tri, dtype=np.bool_)

    TestDelaunay2Dnjit_det(
        points,
        vertices,
        non_Delaunay_tri,
    )

    num_nonDelaunay_tri = len(np.where(non_Delaunay_tri == np.bool_(0))[0])

    print("***Testing with 4x4 determinant***")
    if num_nonDelaunay_tri > 0:
        print("This is not a Delaunay triangulation.")
        print("Number of non-Delaunay triangles : {} \n".format(
            num_nonDelaunay_tri
        ))
    else:
        print("This is a Delaunay triangulation. \n")
