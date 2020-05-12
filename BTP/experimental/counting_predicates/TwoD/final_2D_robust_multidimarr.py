import numpy as np
import tools.BRIO_2D_multidimarr as BRIO
from tools.adaptive_predicates import incircle, orient2d, exactinit2d, incircleadapt
# import BTP.tools.BRIO_2D_multidimarr as BRIO
# from BTP.experimental.counting_predicates.TwoD.tools.adaptive_predicates import incircle, orient2d, exactinit2d
import time


def njit(f):
    return f
from numba import njit


@njit
def _walk(
        point_id, t_index, vertices_ID, neighbour_ID, points, gv, splitter, B,
        C1, C2, D, u, ccwerrboundA, ccwerrboundB, ccwerrboundC,
        resulterrbound, static_filter_o2d, orient2d_count):
    '''
    Walks from the given tri (t_index) to the tri enclosing the given point.

        point_id : The index (corresponding to the points array) of the
                   point to be inserted into the triangulation.
         t_index : The index of the tri to start the walk from.
     vertices_ID : The global array storing all the indices (corresponding
                   to the points array) of the vertices of all the tri.
    neighbour_ID : The global array storing the indices of the neighbouring
                   tri.
          points : The global array storing the co-ordinates of all the
                   points to be triangulated.
              gv : Index assigned to the ghost vertex.
    '''

    gv_idx = 3
    if vertices_ID[t_index, 0] == gv:
        gv_idx = 0
    elif vertices_ID[t_index, 1] == gv:
        gv_idx = 1
    elif vertices_ID[t_index, 2] == gv:
        gv_idx = 2

    if gv_idx != 3:
        # t_index is a ghost tri, in this case simply step into the adjacent
        # real triangle.
        t_index = neighbour_ID[t_index, gv_idx]//3

    point_x = points[point_id, 0]
    point_y = points[point_id, 1]

    while True:
        # i.e. t_index is a real tri

        t_op_index_in_t = 4

        a_x = points[vertices_ID[t_index, 0], 0]
        a_y = points[vertices_ID[t_index, 0], 1]
        b_x = points[vertices_ID[t_index, 1], 0]
        b_y = points[vertices_ID[t_index, 1], 1]
        c_x = points[vertices_ID[t_index, 2], 0]
        c_y = points[vertices_ID[t_index, 2], 1]

        det_left = (point_x-b_x)*(c_y-b_y)
        det_right = (point_y-b_y)*(c_x-b_x)
        det = det_left - det_right
        num = 0
        if np.abs(det) < static_filter_o2d:
            detsum = np.abs(det_left) + np.abs(det_right)
            det, num = orient2d(
                point_x, point_y, c_x, c_y, b_x, b_y, splitter, B, C1, C2, D,
                u, ccwerrboundA, ccwerrboundB, ccwerrboundC, resulterrbound,
                det, detsum)
        orient2d_count[num] += 1
        if det > 0:
            t_op_index_in_t = 0
        else:
            det_left = (point_x-c_x)*(a_y-c_y)
            det_right = (point_y-c_y)*(a_x-c_x)
            det = det_left - det_right
            num = 0
            if np.abs(det) < static_filter_o2d:
                detsum = np.abs(det_left) + np.abs(det_right)
                det, num = orient2d(
                    point_x, point_y, a_x, a_y, c_x, c_y, splitter, B, C1, C2,
                    D, u, ccwerrboundA, ccwerrboundB, ccwerrboundC,
                    resulterrbound, det, detsum)
            orient2d_count[num] += 1
            if det > 0:
                t_op_index_in_t = 1
            else:
                det_left = (point_x-a_x)*(b_y-a_y)
                det_right =  (point_y-a_y)*(b_x-a_x)
                det = det_left - det_right
                num = 0
                if np.abs(det) < static_filter_o2d:
                    detsum = np.abs(det_left) + np.abs(det_right)
                    det, num = orient2d(
                        point_x, point_y, b_x, b_y, a_x, a_y, splitter, B, C1,
                        C2, D, u, ccwerrboundA, ccwerrboundB, ccwerrboundC,
                        resulterrbound, det, detsum)
                orient2d_count[num] += 1
                if det > 0:
                    t_op_index_in_t = 2

        if t_op_index_in_t != 4:
            t_index = neighbour_ID[t_index, t_op_index_in_t]//3
        else:
            # point_id lies inside t_index
            break

        if vertices_ID[t_index, 0] == gv:
            break
        elif vertices_ID[t_index, 1] == gv:
            break
        elif vertices_ID[t_index, 2] == gv:
            break

    return t_index


@njit
def _cavity_helper(
        point_id, t_index, points, vertices_ID, gv, B, C1, C2, D, u, v, bc, ca,
        ab, axbc, axxbc, aybc, ayybc, adet, bxca, bxxca, byca, byyca, bdet,
        cxab, cxxab, cyab, cyyab, cdet, abdet, fin1, fin2, aa, bb, cc, temp8,
        temp16a, temp16b, temp16c, temp32a, temp32b, temp48, temp64, axtbb,
        axtcc, aytbb, aytcc, bxtaa, bxtcc, bytaa, bytcc, cxtaa, cxtbb, cytaa,
        cytbb, axtbc, aytbc, bxtca, bytca, cxtab, cytab, axtbct, aytbct,
        bxtcat, bytcat, cxtabt, cytabt, axtbctt, aytbctt, bxtcatt, bytcatt,
        cxtabtt, cytabtt, abt, bct, cat, abtt, bctt, catt, splitter,
        iccerrboundA, iccerrboundB, iccerrboundC, resulterrbound, ccwerrboundA,
        ccwerrboundB, ccwerrboundC, static_filter_o2d, static_filter_i2d,
        orient2d_count, incircle_count):
    '''
    Checks whether the given point lies inside the circumsphere the given tri.
    Returns True if it does.

            point_id : The index (corresponding to the points array) of the
                       point to be inserted into the triangulation.
             t_index : The index of the tri to check.
              points : The global array storing the co-ordinates of all the
                       points to be triangulated.
         vertices_ID : The global array storing all the indices (corresponding
                       to the points array) of the vertices of all the tri.
    sub_determinants : The global array storing the sub-determinants of all the
                       tri.
                  gv : Index assigned to the ghost vertex.
    '''

    gv_idx = 3
    if vertices_ID[t_index, 0] == gv:
        gv_idx = 0
    elif vertices_ID[t_index, 1] == gv:
        gv_idx = 1
    elif vertices_ID[t_index, 2] == gv:
        gv_idx = 2

    point_x = points[point_id, 0]
    point_y = points[point_id, 1]

    if gv_idx != 3:
        # t_index is a ghost triangle
        if gv_idx == 0:
            b_x = points[vertices_ID[t_index, 1], 0]
            b_y = points[vertices_ID[t_index, 1], 1]
            c_x = points[vertices_ID[t_index, 2], 0]
            c_y = points[vertices_ID[t_index, 2], 1]
        elif gv_idx == 1:
            b_x = points[vertices_ID[t_index, 2], 0]
            b_y = points[vertices_ID[t_index, 2], 1]
            c_x = points[vertices_ID[t_index, 0], 0]
            c_y = points[vertices_ID[t_index, 0], 1]
        elif gv_idx == 2:
            b_x = points[vertices_ID[t_index, 0], 0]
            b_y = points[vertices_ID[t_index, 0], 1]
            c_x = points[vertices_ID[t_index, 1], 0]
            c_y = points[vertices_ID[t_index, 1], 1]

        det_left = (point_x-c_x)*(b_y-c_y)
        det_right = (point_y-c_y)*(b_x-c_x)
        det = det_left - det_right
        num = 0
        if np.abs(det) <= static_filter_o2d:
            detsum = np.abs(det_left) + np.abs(det_right)
            det, num = orient2d(
                point_x, point_y, b_x, b_y, c_x, c_y, splitter, B, C1, C2, D,
                u, ccwerrboundA, ccwerrboundB, ccwerrboundC, resulterrbound,
                det, detsum)
        orient2d_count[num] += 1

        if det > 0:
            return True
        elif det == 0:
            m1_x = point_x - b_x
            m2_x = c_x - point_x
            m1_y = point_y - b_y
            m2_y = c_y - point_y
            if m1_x*m2_x >= 0 and m1_y*m2_y >= 0:
                return True
            else:
                return False
        else:
            return False
    else:
        # t_index is a real triangle
        a_x = points[vertices_ID[t_index, 0], 0]
        a_y = points[vertices_ID[t_index, 0], 1]
        b_x = points[vertices_ID[t_index, 1], 0]
        b_y = points[vertices_ID[t_index, 1], 1]
        c_x = points[vertices_ID[t_index, 2], 0]
        c_y = points[vertices_ID[t_index, 2], 1]

        adx = a_x - point_x
        bdx = b_x - point_x
        cdx = c_x - point_x
        ady = a_y - point_y
        bdy = b_y - point_y
        cdy = c_y - point_y

        bdxcdy = bdx * cdy
        cdxbdy = cdx * bdy
        alift = adx * adx + ady * ady

        cdxady = cdx * ady
        adxcdy = adx * cdy
        blift = bdx * bdx + bdy * bdy

        adxbdy = adx * bdy
        bdxady = bdx * ady
        clift = cdx * cdx + cdy * cdy

        det = alift * (bdxcdy - cdxbdy) + \
              blift * (cdxady - adxcdy) + \
              clift * (adxbdy - bdxady)
        num = 0
        if np.abs(det) <= static_filter_i2d:
            permanent = (np.abs(bdxcdy) + np.abs(cdxbdy)) * alift + \
                        (np.abs(cdxady) + np.abs(adxcdy)) * blift + \
                        (np.abs(adxbdy) + np.abs(bdxady)) * clift
            # errbound = iccerrboundA * permanent
            # if np.abs(det) > errbound:
            #     num = 1
            # else:
            det, num = incircle(
                a_x, a_y, b_x, b_y, c_x, c_y, point_x, point_y, bc, ca, ab,
                axbc, axxbc, aybc, ayybc, adet, bxca, bxxca, byca, byyca, bdet,
                cxab, cxxab, cyab, cyyab, cdet, abdet, fin1, fin2, aa, bb, cc,
                u, v, temp8, temp16a, temp16b, temp16c, temp32a, temp32b,
                temp48, temp64, axtbb, axtcc, aytbb, aytcc, bxtaa, bxtcc,
                bytaa, bytcc, cxtaa, cxtbb, cytaa, cytbb, axtbc, aytbc, bxtca,
                bytca, cxtab, cytab, axtbct, aytbct, bxtcat, bytcat, cxtabt,
                cytabt, axtbctt, aytbctt, bxtcatt, bytcatt, cxtabtt, cytabtt,
                abt, bct, cat, abtt, bctt, catt, splitter, iccerrboundA,
                iccerrboundB, iccerrboundC, resulterrbound, det, permanent)
                # det, num = incircleadapt(
                #     a_x, a_y, b_x, b_y, c_x, c_y, point_x, point_y, permanent,
                #     bc, ca, ab, axbc, axxbc, aybc, ayybc, adet, bxca, bxxca,
                #     byca, byyca, bdet, cxab, cxxab, cyab, cyyab, cdet, abdet,
                #     fin1, fin2, aa, bb, cc, u, v, temp8, temp16a, temp16b,
                #     temp16c, temp32a, temp32b, temp48, temp64, axtbb, axtcc,
                #     aytbb, aytcc, bxtaa, bxtcc, bytaa, bytcc, cxtaa, cxtbb,
                #     cytaa, cytbb, axtbc, aytbc, bxtca, bytca, cxtab, cytab,
                #     axtbct, aytbct, bxtcat, bytcat, cxtabt, cytabt, axtbctt,
                #     aytbctt, bxtcatt, bytcatt, cxtabtt, cytabtt, abt, bct, cat,
                #     abtt, bctt, catt, splitter, iccerrboundB, iccerrboundC,
                #     resulterrbound)
        incircle_count[num] += 1

        if det >= 0.0:
            return True
        else:
            return False


@njit
def _identify_cavity(
        points, point_id, t_index, neighbour_ID, vertices_ID, ic_bad_tri,
        ic_boundary_tri, ic_boundary_vtx, gv, bad_tri_indicator_arr, B, C1, C2,
        D, u, v, bc, ca, ab, axbc, axxbc, aybc, ayybc, adet, bxca, bxxca, byca,
        byyca, bdet, cxab, cxxab, cyab, cyyab, cdet, abdet, fin1, fin2, aa, bb,
        cc, temp8, temp16a, temp16b, temp16c, temp32a, temp32b, temp48, temp64,
        axtbb, axtcc, aytbb, aytcc, bxtaa, bxtcc, bytaa, bytcc, cxtaa, cxtbb,
        cytaa, cytbb, axtbc, aytbc, bxtca, bytca, cxtab, cytab, axtbct, aytbct,
        bxtcat, bytcat, cxtabt, cytabt, axtbctt, aytbctt, bxtcatt, bytcatt,
        cxtabtt, cytabtt, abt, bct, cat, abtt, bctt, catt, splitter,
        iccerrboundA, iccerrboundB, iccerrboundC, resulterrbound, ccwerrboundA,
        ccwerrboundB, ccwerrboundC, static_filter_o2d, static_filter_i2d,
        orient2d_count, incircle_count):
    '''
    Identifies all the 'bad' triangles, i.e. the triangles whose circumcircles
    enclose the given point. Returns a list of the indices of the bad triangles
    and a list of the triangles bordering the cavity.

              points : The global array containing the co-ordinates of all the
                       points to be triangulated.
            point_id : The index (corresponding to the points array) of the
                       point to be inserted into the triangulation.
             t_index : The index of the tri enclosing point_id.
        neighbour_ID : The global array containing the indices of the
                       neighbours of all the triangles.
         vertices_ID : The global array containing the indices (corresponding
                       to the points array) of the vertices of all the tri.
          ic_bad_tri : Helper array, used to store the indices of the 'bad'
                       tri, i.e. those whose circumspheres containt point_id.
     ic_boundary_tri : Helper array, used to store the tri on the boundary of
                       the cavity.
     ic_boundary_vtx : Helper array, used to store the points on the boundary
                       of the cavity.
                  gv : Index assigned to the ghost vertex.
    '''

    ic_len_bad_tri = len(ic_bad_tri)
    ic_bad_tri_end = np.int64(0)

    ic_len_boundary_tri = len(ic_boundary_tri)
    ic_boundary_tri_end = np.int64(0)

    ic_len_boundary_vtx = len(ic_boundary_vtx)
    ic_boundary_vtx_end = np.int64(0)

    # Adding the first bad triangle, i.e. the enclosing triangle
    ic_bad_tri[ic_bad_tri_end] = t_index
    bad_tri_indicator_arr[t_index] = True
    ic_bad_tri_end += 1

    ic_idx = 0
    while True:
        t_index = ic_bad_tri[ic_idx]

        for j in range(3):
            jth_nbr_idx = neighbour_ID[t_index, j]//3

            if not bad_tri_indicator_arr[jth_nbr_idx]:
                # i.e. jth_nbr_idx has not been stored in the ic_bad_tri
                # array yet.
                inside_tri = _cavity_helper(
                    point_id, jth_nbr_idx, points, vertices_ID, gv, B, C1, C2,
                    D, u, v, bc, ca, ab, axbc, axxbc, aybc, ayybc, adet, bxca,
                    bxxca, byca, byyca, bdet, cxab, cxxab, cyab, cyyab, cdet,
                    abdet, fin1, fin2, aa, bb, cc, temp8, temp16a, temp16b,
                    temp16c, temp32a, temp32b, temp48, temp64, axtbb, axtcc,
                    aytbb, aytcc, bxtaa, bxtcc, bytaa, bytcc, cxtaa, cxtbb,
                    cytaa, cytbb, axtbc, aytbc, bxtca, bytca, cxtab, cytab,
                    axtbct, aytbct, bxtcat, bytcat, cxtabt, cytabt, axtbctt,
                    aytbctt, bxtcatt, bytcatt, cxtabtt, cytabtt, abt, bct, cat,
                    abtt, bctt, catt, splitter, iccerrboundA, iccerrboundB,
                    iccerrboundC, resulterrbound, ccwerrboundA, ccwerrboundB,
                    ccwerrboundC, static_filter_o2d, static_filter_i2d,
                    orient2d_count, incircle_count)
                if inside_tri is True:
                    # i.e. the j'th neighbour is a bad triangle
                    if ic_bad_tri_end >= ic_len_bad_tri:
                        temp_arr1 = np.empty(2*ic_len_bad_tri, dtype=np.int64)
                        for l in range(ic_bad_tri_end):
                            temp_arr1[l] = ic_bad_tri[l]
                        ic_len_bad_tri = 2*ic_len_bad_tri
                        ic_bad_tri = temp_arr1

                    ic_bad_tri[ic_bad_tri_end] = jth_nbr_idx
                    ic_bad_tri_end += 1
                    bad_tri_indicator_arr[jth_nbr_idx] = True
                else:
                    # i.e. the j'th neighbour is a boundary triangle
                    if ic_boundary_tri_end >= ic_len_boundary_tri:
                        temp_arr2 = np.empty(
                            2*ic_len_boundary_tri,
                            dtype=np.int64
                        )
                        for l in range(ic_boundary_tri_end):
                            temp_arr2[l] = ic_boundary_tri[l]
                        ic_len_boundary_tri = 2*ic_len_boundary_tri
                        ic_boundary_tri = temp_arr2

                    ic_boundary_tri[ic_boundary_tri_end] = neighbour_ID[
                        t_index, j]
                    ic_boundary_tri_end += 1

                    # Storing the vertices of t_index that lie on the boundary
                    if ic_boundary_vtx_end >= ic_len_boundary_vtx:
                        temp_arr3 = np.empty(
                            shape=(2*ic_len_boundary_vtx, 2),
                            dtype=np.int64
                        )
                        for l in range(ic_boundary_vtx_end):
                            temp_arr3[l, 0] = ic_boundary_vtx[l, 0]
                            temp_arr3[l, 1] = ic_boundary_vtx[l, 1]
                        ic_len_boundary_vtx = 2*ic_len_boundary_vtx
                        ic_boundary_vtx = temp_arr3

                    ic_boundary_vtx[ic_boundary_vtx_end, 0] = vertices_ID[
                        t_index, (j+1) % 3]
                    ic_boundary_vtx[ic_boundary_vtx_end, 1] = vertices_ID[
                        t_index, (j+2) % 3]

                    ic_boundary_vtx_end += 1

        ic_idx += 1

        if ic_idx == ic_bad_tri_end:
            break

    return ic_bad_tri, ic_bad_tri_end, ic_boundary_tri, \
           ic_boundary_tri_end, ic_boundary_vtx


@njit
def _make_Delaunay_ball(
        point_id, bad_tri, bad_tri_end, boundary_tri, boundary_tri_end,
        boundary_vtx, points, neighbour_ID, vertices_ID, num_tri, gv):
    '''
    Joins all the vertices on the boundary to the new point, and forms
    the corresponding triangles along with their adjacencies. Returns the index
    of a new triangle, to be used as the starting point of the next walk.

         point_id : The index corresponding to the points array of the point to
                    be inserted into the triangulation.
         bad_tri : The list of tri whose circumcircle contains point_id.
    boundary_tri : The list of triangles lying on the boundary of the cavity
                    formed by the bad triangles.
     boundary_vtx : The vertices lying on the boundary of the cavity formed by
                    all the bad triangles.
           points : The global array storing the co-ordinates of all the points
                    to be triangulated.
    '''

    # populating the cavity with new triangles
    for i in range(boundary_tri_end):
        if i < bad_tri_end:
            t_index = bad_tri[i]
        else:
            t_index = num_tri
            num_tri += 1

        t_info = boundary_tri[i]
        neighbour_ID[t_index, 0] = t_info
        vertices_ID[t_index, 0] = point_id
        vertices_ID[t_index, 1] = boundary_vtx[i, 0]
        vertices_ID[t_index, 2] = boundary_vtx[i, 1]
        neighbour_ID[t_info//3, t_info % 3] = 3*t_index

    for i in range(boundary_tri_end):
            if i < bad_tri_end:
                t1 = bad_tri[i]
            else:
                t1 = num_tri - (boundary_tri_end-i)
            for j in range(boundary_tri_end):
                if j < bad_tri_end:
                    t2 = bad_tri[j]
                else:
                    t2 = num_tri - (boundary_tri_end-j)
                if vertices_ID[t1, 1] == vertices_ID[t2, 2]:
                    neighbour_ID[t1, 2] = 3*t2+1
                    neighbour_ID[t2, 1] = 3*t1+2
                    break

    old_tri =  bad_tri[bad_tri_end-1]

    if boundary_tri_end < bad_tri_end:
        old_tri = bad_tri[boundary_tri_end-1]
        for k in range(boundary_tri_end, bad_tri_end):
            tri = bad_tri[k]
            for t in range(tri, num_tri):
                vertices_ID[t, 0] = vertices_ID[t+1, 0]
                vertices_ID[t, 1] = vertices_ID[t+1, 1]
                vertices_ID[t, 2] = vertices_ID[t+1, 2]

                neighbour_ID[t, 0] = neighbour_ID[t+1, 0]
                neighbour_ID[t, 1] = neighbour_ID[t+1, 1]
                neighbour_ID[t, 2] = neighbour_ID[t+1, 2]

            num_tri -= 1

            for i in range(num_tri):
                for j in range(3):
                    if neighbour_ID[i, j]//3 > tri:
                        neighbour_ID[i, j] = 3*(neighbour_ID[i, j]//3-1) + \
                                              neighbour_ID[i, j] % 3

            for i in range(k+1, bad_tri_end):
                if bad_tri[i] > tri:
                    bad_tri[i] -= 1

    return num_tri, old_tri


@njit
def assembly(
        old_tri, ic_bad_tri, ic_boundary_tri, ic_boundary_vtx, points,
        vertices_ID, neighbour_ID, num_tri, gv, bad_tri_indicator_arr,
        global_arr, orient2d_count, incircle_count):

    resulterrbound, ccwerrboundA, ccwerrboundB, ccwerrboundC, iccerrboundA, \
    iccerrboundB, iccerrboundC, splitter, static_filter_o2d, \
    static_filter_i2d = exactinit2d(points)
    B = global_arr[0:4]
    C1 = global_arr[4:12]
    C2 = global_arr[12:24]
    D = global_arr[24:40]
    u = global_arr[40:44]
    v = global_arr[44:48]
    bc = global_arr[48:52]
    ca = global_arr[52:56]
    ab = global_arr[56:60]
    axbc = global_arr[60:68]
    axxbc = global_arr[68:84]
    aybc = global_arr[84:92]
    ayybc = global_arr[92:108]
    adet = global_arr[108:140]
    bxca = global_arr[140:148]
    bxxca = global_arr[148:164]
    byca = global_arr[164:172]
    byyca = global_arr[172:188]
    bdet = global_arr[188:220]
    cxab = global_arr[220:228]
    cxxab = global_arr[228:244]
    cyab = global_arr[244:252]
    cyyab = global_arr[252:268]
    cdet = global_arr[268:300]
    abdet = global_arr[300:364]
    fin1 = global_arr[364:1516]
    fin2 = global_arr[1516:2668]
    aa = global_arr[2668:2672]
    bb = global_arr[2672:2676]
    cc = global_arr[2676:2680]
    temp8 = global_arr[2680:2688]
    temp16a = global_arr[2688:2704]
    temp16b = global_arr[2704:2720]
    temp16c = global_arr[2720:2736]
    temp32a = global_arr[2736:2768]
    temp32b = global_arr[2768:2800]
    temp48 = global_arr[2800:2848]
    temp64 = global_arr[2848:2912]
    axtbb = global_arr[2912:2920]
    axtcc = global_arr[2920:2928]
    aytbb = global_arr[2928:2936]
    aytcc = global_arr[2936:2944]
    bxtaa = global_arr[2944:2952]
    bxtcc = global_arr[2952:2960]
    bytaa = global_arr[2960:2968]
    bytcc = global_arr[2968:2976]
    cxtaa = global_arr[2976:2984]
    cxtbb = global_arr[2984:2992]
    cytaa = global_arr[2992:3000]
    cytbb = global_arr[3000:3008]
    axtbc = global_arr[3008:3016]
    aytbc = global_arr[3016:3024]
    bxtca = global_arr[3024:3032]
    bytca = global_arr[3032:3040]
    cxtab = global_arr[3040:3048]
    cytab = global_arr[3048:3056]
    axtbct = global_arr[3056:3072]
    aytbct = global_arr[3072:3088]
    bxtcat = global_arr[3088:3104]
    bytcat = global_arr[3104:3120]
    cxtabt = global_arr[3120:3136]
    cytabt = global_arr[3136:3152]
    axtbctt = global_arr[3152:3160]
    aytbctt = global_arr[3160:3168]
    bxtcatt = global_arr[3168:3176]
    bytcatt = global_arr[3176:3184]
    cxtabtt = global_arr[3184:3192]
    cytabtt = global_arr[3192:3200]
    abt = global_arr[3200:3208]
    bct = global_arr[3208:3216]
    cat = global_arr[3216:3224]
    abtt = global_arr[3224:3228]
    bctt = global_arr[3228:3232]
    catt = global_arr[3232:3236]

    for point_id in range(3, gv):

        enclosing_tri = _walk(
            point_id, old_tri, vertices_ID, neighbour_ID, points, gv, splitter,
            B, C1, C2, D, u, ccwerrboundA, ccwerrboundB, ccwerrboundC, 
            resulterrbound, static_filter_o2d, orient2d_count)

        ic_bad_tri, ic_bad_tri_end, ic_boundary_tri, ic_boundary_tri_end, \
        ic_boundary_vtx = _identify_cavity(
            points, point_id, enclosing_tri, neighbour_ID, vertices_ID,
            ic_bad_tri, ic_boundary_tri, ic_boundary_vtx, gv,
            bad_tri_indicator_arr, B, C1, C2, D, u, v, bc, ca, ab, axbc, axxbc,
            aybc, ayybc, adet, bxca, bxxca, byca, byyca, bdet, cxab, cxxab,
            cyab, cyyab, cdet, abdet, fin1, fin2, aa, bb, cc, temp8, temp16a,
            temp16b, temp16c, temp32a, temp32b, temp48, temp64, axtbb, axtcc,
            aytbb, aytcc, bxtaa, bxtcc, bytaa, bytcc, cxtaa, cxtbb, cytaa,
            cytbb, axtbc, aytbc, bxtca, bytca, cxtab, cytab, axtbct, aytbct,
            bxtcat, bytcat, cxtabt, cytabt, axtbctt, aytbctt, bxtcatt, bytcatt,
            cxtabtt, cytabtt, abt, bct, cat, abtt, bctt, catt, splitter,
            iccerrboundA, iccerrboundB, iccerrboundC, resulterrbound,
            ccwerrboundA, ccwerrboundB, ccwerrboundC, static_filter_o2d,
            static_filter_i2d, orient2d_count, incircle_count)

        num_tri, old_tri = _make_Delaunay_ball(
            point_id, ic_bad_tri, ic_bad_tri_end, ic_boundary_tri,
            ic_boundary_tri_end, ic_boundary_vtx, points, neighbour_ID,
            vertices_ID, num_tri, gv)

        for i in range(ic_bad_tri_end):
            t = ic_bad_tri[i]
            bad_tri_indicator_arr[t] = False

    return num_tri


@njit
def initialize(points, vertices_ID, neighbour_ID):

    N = len(points)

    a_x = points[0, 0]
    a_y = points[0, 1]
    b_x = points[1, 0]
    b_y = points[1, 1]

    num_tri = np.int64(0)

    idx = 2
    while True:
        p_x = points[idx, 0]
        p_y = points[idx, 1]
        signed_area = (b_x-a_x)*(p_y-a_y)-(p_x-a_x)*(b_y-a_y)
        if signed_area > 0:
            points[2, 0], points[idx, 0] = points[idx, 0], points[2, 0]
            points[2, 1], points[idx, 1] = points[idx, 1], points[2, 1]
            break
        elif signed_area < 0:
            points[2, 0], points[idx, 0] = points[idx, 0], points[2, 0]
            points[2, 1], points[idx, 1] = points[idx, 1], points[2, 1]
            points[0, 0], points[1, 0] = points[1, 0], points[0, 0]
            points[0, 1], points[1, 1] = points[1, 1], points[0, 1]
            break
        else:
            idx += 1

    vertices_ID[0, 0] = 0      #
    vertices_ID[0, 1] = 1      # ---> 0th triangle [real]
    vertices_ID[0, 2] = 2      #

    vertices_ID[1, 0] = 0      #
    vertices_ID[1, 1] = N      # ---> 1st triangle [ghost]
    vertices_ID[1, 2] = 1      #

    vertices_ID[2, 0] = 1      #
    vertices_ID[2, 1] = N      # ---> 2nd triangle [ghost]
    vertices_ID[2, 2] = 2      #

    vertices_ID[3, 0] = 2      #
    vertices_ID[3, 1] = N      # ---> 3rd triangle [ghost]
    vertices_ID[3, 2] = 0      #

    neighbour_ID[0, 0] = 3*2+1     #
    neighbour_ID[0, 1] = 3*3+1     # ---> 0th triangle [real]
    neighbour_ID[0, 2] = 3*1+1     #

    neighbour_ID[1, 0] = 3*2+2     #
    neighbour_ID[1, 1] = 3*0+2     # ---> 1st triangle [ghost]
    neighbour_ID[1, 2] = 3*3+0     #

    neighbour_ID[2, 0] = 3*3+2     #
    neighbour_ID[2, 1] = 3*0+0     # ---> 2nd triangle [ghost]
    neighbour_ID[2, 2] = 3*1+0     #

    neighbour_ID[3, 0] = 3*1+2     #
    neighbour_ID[3, 1] = 3*0+1     # ---> 3rd triangle [ghost]
    neighbour_ID[3, 2] = 3*2+0     #

    num_tri += 4

    return num_tri


class Delaunay2D:

    def __init__(self, points):
        '''
        points : N x 2 array/list of points
        '''

        N = len(points)

        self.gv = N

        self.vertices_ID = N*np.ones(shape=(2*N-2, 3), dtype=np.int64)
        self.neighbour_ID = np.empty(shape=(2*N-2, 3), dtype=np.int64)

        self.points = BRIO.make_BRIO(np.asarray(points, dtype=np.float64))

        self.num_tri = initialize(self.points, self.vertices_ID,
                                  self.neighbour_ID)

    def makeDT(self):

        old_tri = np.int64(0)

        # Arrays that will be passed into the jit-ed functions so that they
        # don't have to get their hands dirty with object creation.
        ic_bad_tri = np.empty(50, dtype=np.int64)
        ic_boundary_tri = np.empty(50, dtype=np.int64)
        ic_boundary_vtx = np.empty(shape=(50, 2), dtype=np.int64)
        bad_tri_indicator_arr = np.zeros(shape=2*self.gv-2, dtype=np.bool_)

        global_arr = np.empty(shape=3236, dtype=np.float64)
        orient2d_count = np.zeros(shape=6, dtype=np.int64)
        incircle_count = np.zeros(shape=6, dtype=np.int64)

        self.num_tri = assembly(
            old_tri, ic_bad_tri, ic_boundary_tri, ic_boundary_vtx, self.points,
            self.vertices_ID, self.neighbour_ID, self.num_tri, self.gv,
            bad_tri_indicator_arr, global_arr, orient2d_count, incircle_count)

        print("orient2d_count : {}".format(orient2d_count))
        print(orient2d_count*100/np.sum(orient2d_count))
        print("incircle_count : {}".format(incircle_count))
        print(incircle_count*100/np.sum(incircle_count))


def perf(N):
    import time

    np.random.seed(seed=10)

    print("\npriming numba")
    temp_pts = np.random.rand(10, 2)
    tempDT = Delaunay2D(temp_pts)
    print("DT initialized")
    tempDT.makeDT()
    print("numba primed \n")

    del temp_pts
    del tempDT

    num_runs = 5
    time_arr = np.empty(shape=num_runs, dtype=np.float64)

    # points = np.zeros(shape=(2*N, 2), dtype=np.float64)
    # points[0:N, 0] = np.linspace(-100.0, 100.0, N)
    # # points[:, 1] = 0.001*np.random.randn(N)
    # points[0:N, 1] = 2*points[0:N, 0] + 2.0# + 0.001*np.random.rand(N)
    # points[0, 1] = 0.0
    # theta = np.arange(N)*2*np.pi/N
    # points[N:, 0] = np.cos(theta)
    # points[N:, 1] = np.sin(theta)

    np.random.seed(seed=12345)
    for i in range(num_runs):
        points = np.random.randn(N, 2)
        start = time.time()
        DT = Delaunay2D(points)
        DT.makeDT()
        end = time.time()
        time_arr[i] = end - start
        print("RUN {} : {} s. \n".format(i, time_arr[i]))
        del DT
        del points

    return np.min(time_arr)

if __name__ == "__main__":
    import sys
    N = int(sys.argv[1])
    time = perf(N)
    print("   Time taken to make the triangulation : {} s".format(time))
