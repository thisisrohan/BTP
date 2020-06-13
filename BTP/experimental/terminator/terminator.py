import numpy as np
from CDT.DT.tools.adaptive_predicates import incircle, orient2d, exactinit2d
from CDT.AngladaGiftWrapping import _insert_segment
from CDT.DT.final_2D_robust_multidimarr import _walk, _identify_cavity, \
                                               _make_Delaunay_ball, initialize, \
                                               _cavity_helper
import CDT.DT.tools.BRIO_2D_multidimarr as BRIO
from CDT.hash_table import initialize_seg_ht, get_seg_idx, \
                           delete_entry_seg_ht, add_entry_seg_ht, \
                           initialize_tri_ht, get_tri_idx, \
                           delete_entry_tri_ht, add_entry_tri_ht, \
                           delete_bad_tri_from_tri_ht, add_new_tri_to_tri_ht, \
                           dequeue_st, dequeue_ss, enqueue_st, enqueue_ss
from CDT.reset_triangles import reset_tri_2


def njit(f=None, cache=None):
    if cache is None:
        return f
    else:
        def wrap(f):
            return f
        return wrap
# from numba import njit


class TranspilationError(Exception):
    def __init__(self, func_name, message):
        self.func_name = func_name
        self.message = message


class InputError(Exception):
    def __init__(self, input_, message):
        self.input = input_
        self.message = message

class SegmentsArrayError(Exception):
    def __init__(self):
        self.message = 'Error in assigning clusters to the segments. You ' + \
                       'might have faults in the input segments array.'


@njit
def quicksort_segs(segments, start, end, idx1, idx2, idx3):
    '''
    Sorts the segments according to their the entries in idx1, with entries in
    idx2 as the secondary key and entries in idx3 as the tertiary key.
    '''
    array_size = end - start
    if array_size == 1 or array_size == 0:
        # happens only if the original input array is of length 1 or 0
        return
    elif array_size == 2:
        # recursive base case
        e0_1 = segments[start, idx1]
        e0_2 = segments[start, idx2]
        e0_3 = segments[start, idx3]
        e1_1 = segments[start + 1, idx1]
        e1_2 = segments[start + 1, idx2]
        e1_3 = segments[start + 1, idx3]
        if e0_1 > e1_1 or (e0_1 == e1_1 and e0_2 > e1_2) or \
                (e0_1 == e1_1 and e0_2 == e1_2 and e0_3 > e1_3):
            segments[start, idx1] = e1_1
            segments[start, idx2] = e1_2
            segments[start, idx3] = e1_3
            segments[start + 1, idx1] = e0_1
            segments[start + 1, idx2] = e0_2
            segments[start + 1, idx3] = e0_3
            return
    else:
        pivot_idx = array_size // 2
        pivot1 = segments[start + pivot_idx, idx1]
        pivot2 = segments[start + pivot_idx, idx2]
        pivot3 = segments[start + pivot_idx, idx3]

        left = int(-1)
        right = array_size

        while left < right:
            while True:
                left += 1
                e_1 = segments[start + left, idx1]
                e_2 = segments[start + left, idx2]
                e_3 = segments[start + left, idx3]
                if left <= right and \
                        (e_1 < pivot1 or \
                        (e_1 == pivot1 and e_2 < pivot2) or \
                        (e_1 == pivot1 and e_2 == pivot2 and e_3 < pivot3)):
                    pass
                else:
                    break

            while True:
                right -= 1
                e_1 = segments[start + right, idx1]
                e_2 = segments[start + right, idx2]
                e_3 = segments[start + right, idx3]
                if left <= right and \
                        (e_1 > pivot1 or \
                        (e_1 == pivot1 and e_2 > pivot2) or \
                        (e_1 == pivot1 and e_2 == pivot2 and e_3 > pivot3)):
                    pass
                else:
                    break

            if left < right:
                temp1 = segments[start + left, idx1]
                temp2 = segments[start + left, idx2]
                temp3 = segments[start + left, idx3]
                segments[start + left, idx1] = segments[start + right, idx1]
                segments[start + left, idx2] = segments[start + right, idx2]
                segments[start + left, idx3] = segments[start + right, idx3]
                segments[start + right, idx1] = temp1
                segments[start + right, idx2] = temp2
                segments[start + right, idx3] = temp3

        if left > 1:
            quicksort_segs(segments, start, start + left, idx1, idx2, idx3)
        if right < array_size - 2:
            quicksort_segs(segments, start + right + 1, end, idx1, idx2, idx3)
        return


@njit
def quicksort(array, start, end):
    '''
    Sorts a one dimensional array. Used for sorting the duplicate_segs array.
    '''
    array_size = end - start
    if array_size == 1 or array_size == 0:
        # happens only if the original input array is of length 1 or 0
        return
    elif array_size == 2:
        # recursive base case
        if array[start] > array[start + 1]:
            temp = array[start]
            array[start] = array[start + 1]
            array[start + 1] = temp
            return
    else:
        pivot_idx = array_size // 2
        pivot = array[start + pivot_idx]

        left = int(-1)
        right = array_size

        while left < right:
            while True:
                left += 1
                if left <= right and array[start + left] < pivot:
                    pass
                else:
                    break

            while True:
                right -= 1
                if left <= right and array[start + right] > pivot:
                    pass
                else:
                    break

            if left < right:
                temp = array[start + left]
                array[start + left] = array[start + right]
                array[start + right] = temp

        if left > 1:
            quicksort(array, start, start + left)
        if right < array_size - 2:
            quicksort(array, start + right + 1, end)
        return


@njit
def binary_search_segs(segments, start, end, vtx1, vtx2):
    array_size = end - start
    if array_size == 0:
        return -1

    mid = array_size // 2
    mid1 = segments[start + mid, 0]
    mid2 = segments[start + mid, 1]
    if vtx1 == mid1:
        if vtx2 == mid2:
            idx = start + mid
        elif vtx2 > mid2:
            idx = binary_search_segs(
                segments, start + mid + 1, end, vtx1, vtx2)
        elif vtx2 < mid2:
            idx = binary_search_segs(segments, start, start + mid, vtx1, vtx2)
    elif vtx1 > mid1:
        idx = binary_search_segs(segments, start + mid + 1, end, vtx1, vtx2)
    elif vtx1 < mid1:
        idx = binary_search_segs(segments, start, start + mid, vtx1, vtx2)

    return idx


@njit
def _identify_clusters(
        points, segments, duplicate_segs, hist, theta, exactinit_arr,
        global_arr, num_segs, vertices_ID, neighbour_ID, tri_ht_cap,
        tri_ht_arr):
    '''
    Identifies clusters and reorders segments so that :
        1) all segments that share a vertex have that vertex as their
           first point
        2) the third column stores which cluster a segment belongs to (stores
           -1 if it belongs to no clusters)
    '''

    # num_segs = int(segments.shape[0]/2)
    segs_cap = segments.shape[0]
    # store the end points of the segments and the frequency with which they
    # occur, i.e. the number of segments they are a part of
    for i in range(num_segs):
        vtx1 = segments[i, 0]
        vtx2 = segments[i, 1]
        hist[vtx1] += 1
        hist[vtx2] += 1
    # print(hist)

    # readjust the segments array so that the vertices that are a part of more
    # than one segment are the first stored vertex
    end = num_segs
    for i in range(num_segs):
        vtx1 = segments[i, 0]
        vtx2 = segments[i, 1]
        if hist[vtx1] > 1:
            if segments[i, 2] != -1:
                # segment i has already been assigned a potential cluster, but
                # it is also a part of the potential cluster of vtx1; in this
                # case clone it and assign the clone to the potential cluster
                # of vtx1
                segments[end, 0] = vtx1
                segments[end, 1] = vtx2
                segments[end, 2] = vtx1
                end += 1
            else:
                segments[i, 2] = vtx1
        if hist[vtx2] > 1:
            if segments[i, 2] != -1:
                # segment i has already been assigned a potential cluster, but
                # it is also a part of the potential cluster of vtx2; in this
                # case clone it and assign the clone to the potential cluster
                # of vtx2
                segments[end, 0] = vtx2
                segments[end, 1] = vtx1
                segments[end, 2] = vtx2
                end += 1
            else:
                segments[i, 0] = vtx2
                segments[i, 1] = vtx1
                segments[i, 2] = vtx2
    # print(segments[0:end])
    quicksort_segs(segments, 0, end, 0, 1, 2)
    # print(segments[0:end])

    cluster = -2
    cluster_iter = 0
    sixty = np.pi/3
    for i in range(end):
        cluster_i = segments[i, 2]
        if cluster_i != -1:
            if i == end - 1:
                cluster_ip1 = -1
            else:
                cluster_ip1 = segments[i + 1, 2]
            if cluster_i != cluster:
                cluster = cluster_i
                i_org = i
            if cluster_ip1 != cluster:
                i_end = i + 1
                end_c = 0
                v0 = segments[i_org, 0]
                v1 = segments[i_org, 1]
                v0_x = points[v0, 0]
                v0_y = points[v0, 1]
                v1_x = points[v1, 0]
                v1_y = points[v1, 1]
                v10_x = v1_x - v0_x
                v10_y = v1_y - v0_y
                v10_norm = (v10_x*v10_x + v10_y*v10_y)**0.5
                theta[end_c] = 0.0
                end_c += 1
                for j in range(i_org, i_end):
                    segments[j, 2] = -1
                for j in range(i_org + 1, i_end):
                    v2 = segments[j, 1]
                    v2_x = points[v2, 0]
                    v2_y = points[v2, 1]
                    v20_x = v2_x - v0_x
                    v20_y = v2_y - v0_y
                    v20_norm = (v20_x*v20_x + v20_y*v20_y)**0.5
                    cos_theta = (v10_x*v20_x + v10_y*v20_y)/(v10_norm*v20_norm)
                    theta_j = np.arccos(cos_theta)
                    area = orient2d(
                        v0_x, v0_y, v1_x, v1_y, v2_x, v2_y, exactinit_arr,
                        global_arr)
                    if area < 0:
                        theta_j *= -1.0
                    theta[end_c] = theta_j
                    end_c += 1
                for j in range(i_org, i_end):
                    j_adj = j - i_org
                    theta_j = theta[j_adj]
                    for k in range(i_org, i_end):
                        if k != j:
                            k_adj = k - i_org
                            theta_k = theta[k_adj]
                            delta_theta = theta_k - theta_j
                            if delta_theta < 0:
                                delta_theta *= -1.0
                                tri_idx_j, jj = get_tri_idx(
                                    segments[j, 1], segments[j, 0], tri_ht_cap,
                                    tri_ht_arr, vertices_ID)
                            else:
                                tri_idx_j, jj = get_tri_idx(
                                    segments[j, 0], segments[j, 1], tri_ht_cap,
                                    tri_ht_arr, vertices_ID)
                            tri_idx_k = neighbour_ID[tri_idx_j, jj] // 3
                            # if delta_theta < sixty:
                            if delta_theta < sixty and \
                                    vertices_ID[tri_idx_j, 3] == 0 and \
                                    vertices_ID[tri_idx_k, 3] == 0:
                                c_j = segments[j, 2]
                                c_k = segments[k, 2]
                                if c_j == -1 and c_k == -1:
                                    cluster_iter += 1
                                    segments[j, 2] = cluster_iter
                                    segments[k, 2] = cluster_iter
                                elif c_j != -1 and c_k == -1:
                                    segments[k, 2] = c_j
                                elif c_j == -1 and c_k != -1:
                                    segments[j, 2] = c_k
                                elif c_j == c_k:
                                    pass
                                else:
                                    raise SegmentsArrayError()
    # print(segments[0:end])

    ### now merging duplicate segments
    ds_end = 0
    for i in range(end):
        vtx1 = segments[i, 0]
        vtx2 = segments[i, 1]
        if hist[vtx1] > 1 and hist[vtx2] > 1:
            j = binary_search_segs(segments, 0, end, vtx2, vtx1)
            # print('i : {}, j : {}'.format(i, j))
            # print('i : ')
            # print(i)
            # print('j : ')
            # print(j)
            c_i = segments[i, 2]
            c_j = segments[j, 2]
            if c_i != -1 and c_j == -1:
                pass
            elif c_i == -1 and c_j == -1:
                if i < j:
                    pass
                else:
                    continue
            elif c_i == -1 and c_j != -1:
                continue
            elif c_i <= num_segs and c_j <= num_segs:
                segments[i, 2] = c_i + segs_cap*c_j
            else:
                continue
            duplicate_segs[ds_end] = j
            ds_end += 1
    # print(segments[0:end])
    # print(duplicate_segs[0:ds_end])

    if ds_end != 0:
        quicksort(duplicate_segs, 0, ds_end)
        # print(duplicate_segs[0:ds_end])
        ds_end -= 1
        seg_idx = duplicate_segs[ds_end]
        temp_end = end
        for i in range(end - 1, -1, -1):
            # print(i)
            if i == seg_idx:
                # delete i'th segment
                for j in range(i, temp_end - 1):
                    segments[j, 0] = segments[j + 1, 0]
                    segments[j, 1] = segments[j + 1, 1]
                    segments[j, 2] = segments[j + 1, 2]
                temp_end -= 1
                ds_end -= 1
                if ds_end >= 0:
                    seg_idx = duplicate_segs[ds_end]
                else:
                    seg_idx = end
    # print(segments[0:temp_end])

    return


@njit
def _identify_cavity_CDT(
        points, point_id, t_index, neighbour_ID, vertices_ID, bad_tri,
        boundary_tri, boundary_vtx, gv, bad_tri_indicator_arr, exactinit_arr,
        global_arr, seg_ht_cap, seg_ht_arr, segments, bt_end=0,
        boundary_end=0, seg_idx=-2):
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

    bt_len = bad_tri.shape[0]
    # bt_end = int(0)

    boundary_len = boundary_tri.shape[0]
    # boundary_end = int(0)

    # Adding the first bad triangle, i.e. the enclosing triangle
    if bt_end >= bt_len:
        temp_arr1 = np.empty(2*bt_len, dtype=np.int64)
        for l in range(bt_end):
            temp_arr1[l] = bad_tri[l]
        bt_len *= 2
        bad_tri = temp_arr1
    bad_tri[bt_end] = t_index
    bad_tri_indicator_arr[t_index] = True
    bt_end += 1

    bt_iter = bt_end - 1
    while True:
        t_index = bad_tri[bt_iter]

        for j in range(3):
            jth_nbr_idx = neighbour_ID[t_index, j] // 3
            v1 = vertices_ID[t_index, (j + 1) % 3]
            v2 = vertices_ID[t_index, (j + 2) % 3]
            s = get_seg_idx(v1, v2, seg_ht_cap, seg_ht_arr, segments)
            # print("v1 : {}, v2 : {}, s : {}".format(v1, v2, s))

            if bad_tri_indicator_arr[jth_nbr_idx] == False:
                # i.e. jth_nbr_idx has not been stored in the ic_bad_tri
                # array yet and it doesn't lie across a segment.
                inside_tri = False
                if s == -1:
                    inside_tri = _cavity_helper(
                        point_id, jth_nbr_idx, points, vertices_ID, gv,
                        exactinit_arr, global_arr)
                if inside_tri == True:
                    # i.e. the j'th neighbour is a bad triangle
                    if bt_end >= bt_len:
                        temp_arr1 = np.empty(2*bt_len, dtype=np.int64)
                        for l in range(bt_end):
                            temp_arr1[l] = bad_tri[l]
                        bt_len *= 2
                        bad_tri = temp_arr1

                    bad_tri[bt_end] = jth_nbr_idx
                    bt_end += 1
                    bad_tri_indicator_arr[jth_nbr_idx] = True
                else:
                    if s != seg_idx:
                        # i.e. the j'th neighbour is a boundary triangle
                        if boundary_end >= boundary_len:
                            temp_arr2 = np.empty(
                                2*boundary_len, dtype=np.int64)
                            temp_arr3 = np.empty(
                                shape=(2*boundary_len, 2), dtype=np.int64)
                            for l in range(boundary_end):
                                temp_arr2[l] = boundary_tri[l]
                                temp_arr3[l, 0] = boundary_vtx[l, 0]
                                temp_arr3[l, 1] = boundary_vtx[l, 1]
                            boundary_len *= 2
                            boundary_tri = temp_arr2
                            boundary_vtx = temp_arr3

                        boundary_tri[boundary_end] = neighbour_ID[t_index, j]
                        boundary_vtx[boundary_end, 0] = v1
                        boundary_vtx[boundary_end, 1] = v2
                        boundary_end += 1

        bt_iter += 1

        if bt_iter == bt_end:
            break

    return bad_tri, bt_end, boundary_tri, boundary_end, boundary_vtx


@njit
def find_smallest_angle(a_sq, b_sq, c_sq):
    '''
    a_sq, b_sq, c_sq are the squares of the side lengths.
    '''
    a = a_sq**0.5
    b = b_sq**0.5
    c = c_sq**0.5
    if a < b:
        temp = a
        a = b
        b = temp
    if a < c:
        temp = a
        a = c
        c = temp
    if b < c:
        temp = b
        b = c
        c = temp

    # if c <= 0.1*a:
    #     C = 2*np.arctan(((((a-b)+c)*(c+(b-a)))/((a+(b+c))*((a-c)+b)))**0.5)
    # else:
    #     C = np.arccos(((b/a)+(a/b)-(c/a)*(c/b))*0.5)
    C = np.arccos( ( (b/a) + (a/b) - (c/a)*(c/b) )*0.5 )
    # print("a : {}, b : {}, c : {}, C : {}".format(a, b, c, C))
    return C


@njit
def _make_Delaunay_ball_CDT(
        point_id, bad_tri, bad_tri_end, boundary_tri, boundary_tri_end,
        boundary_vtx, points, neighbour_ID, vertices_ID, num_tri, gv,
        tri_ht_cap, tri_ht_arr):
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
        vertices_ID[t_index, 3] = 0
        neighbour_ID[t_info // 3, t_info % 3] = 3*t_index

    for i in range(boundary_tri_end):
        if i < bad_tri_end:
            t1 = bad_tri[i]
        else:
            t1 = num_tri - (boundary_tri_end - i)
        for j in range(boundary_tri_end):
            if j < bad_tri_end:
                t2 = bad_tri[j]
            else:
                t2 = num_tri - (boundary_tri_end - j)
            if vertices_ID[t1, 1] == vertices_ID[t2, 2]:
                neighbour_ID[t1, 2] = 3*t2 + 1
                neighbour_ID[t2, 1] = 3*t1 + 2
                break

    old_tri =  bad_tri[bad_tri_end - 1]

    add_new_tri_flag = True
    if boundary_tri_end < bad_tri_end:
        add_new_tri_flag = False
        print("oops")
        print("boundary_tri_end : {}".format(boundary_tri_end))
        print("bad_tri_end : {}".format(bad_tri_end))
        old_tri = bad_tri[boundary_tri_end-1]
        for k in range(boundary_tri_end, bad_tri_end):
            tri = bad_tri[k]
            for t in range(tri, num_tri - 1):
                vertices_ID[t, 0] = vertices_ID[t+1, 0]
                vertices_ID[t, 1] = vertices_ID[t+1, 1]
                vertices_ID[t, 2] = vertices_ID[t+1, 2]
                vertices_ID[t, 3] = vertices_ID[t+1, 3]

                neighbour_ID[t, 0] = neighbour_ID[t+1, 0]
                neighbour_ID[t, 1] = neighbour_ID[t+1, 1]
                neighbour_ID[t, 2] = neighbour_ID[t+1, 2]

            num_tri -= 1

            for i in range(num_tri):
                for j in range(3):
                    if neighbour_ID[i, j] // 3 > tri:
                        neighbour_ID[i, j] = 3*(neighbour_ID[i, j]//3-1) + \
                                              neighbour_ID[i, j] % 3

            for i in range(k+1, bad_tri_end):
                if bad_tri[i] > tri:
                    bad_tri[i] -= 1

        for i in range(tri_ht_cap):
            for j in range(6):
                tri_ht_arr[i, j] = -1
        initialize_tri_ht(tri_ht_cap, tri_ht_arr, vertices_ID, num_tri)

    return num_tri, old_tri, add_new_tri_flag



@njit
def new_vertex(
        p, points, vertices_ID, bad_tri, bt_end, boundary_end, num_tri, gv,
        seg_ht_cap, seg_ht_arr, segments, split_segs, ss_params, split_tri,
        st_params, min_angle, qual_f):

    print("new_vertex entered")
    p_x = points[p, 0]
    p_y = points[p, 1]
    for i in range(boundary_end):
        if i < bt_end:
            t = bad_tri[i]
        else:
            t = num_tri - (boundary_end - i)
        v1 = vertices_ID[t, 1]
        v2 = vertices_ID[t, 2]
        print("t : {}, vid : {}".format(t, vertices_ID[t, :]))
        if vertices_ID[t, 3] != -1:
            seg_idx = get_seg_idx(v1, v2, seg_ht_cap, seg_ht_arr, segments)
            v1_x = points[v1, 0]
            v1_y = points[v1, 1]
            v2_x = points[v2, 0]
            v2_y = points[v2, 1]
            # print("p : ({}, {}), v1 : ({}, {}), v2 : ({}, {})".format(p_x, p_y, v1_x, v1_y, v2_x, v2_y))
            proceed = True
            if seg_idx != -1:
                v1p_x = v1_x - p_x
                v1p_y = v1_y - p_y
                v2p_x = v2_x - p_x
                v2p_y = v2_y - p_y
                sign_of_cos_theta = v1p_x*v2p_x + v1p_y*v2p_y
                if sign_of_cos_theta <= 0.0:
                    split_segs = enqueue_ss(
                        split_segs, ss_params, seg_idx, v1, v2)
                    proceed = False

            if proceed == True:
                delta_t = qual_f(p_x, p_y, v1_x, v1_y, v2_x, v2_y)
                side_a_sq = (v2_x - v1_x)*(v2_x - v1_x) + \
                            (v2_y - v1_y)*(v2_y - v1_y)
                side_b_sq = (v2_x - p_x)*(v2_x - p_x) + \
                            (v2_y - p_y)*(v2_y - p_y)
                side_c_sq = (p_x - v1_x)*(p_x - v1_x) + \
                            (p_y - v1_y)*(p_y - v1_y)
                min_tri_angle = find_smallest_angle(
                    side_a_sq, side_b_sq, side_c_sq)
                if side_a_sq >= side_b_sq and side_a_sq >= side_c_sq:
                    if side_b_sq >= side_c_sq:
                        seg_idx1 = get_seg_idx(
                            v2, p, seg_ht_cap, seg_ht_arr, segments)
                        seg_idx2 = get_seg_idx(
                            v2, v1, seg_ht_cap, seg_ht_arr, segments)
                    else:
                        seg_idx1 = get_seg_idx(
                            v1, p, seg_ht_cap, seg_ht_arr, segments)
                        seg_idx2 = get_seg_idx(
                            v1, v2, seg_ht_cap, seg_ht_arr, segments)
                elif side_b_sq >= side_a_sq and side_b_sq >= side_c_sq:
                    if side_a_sq >= side_c_sq:
                        seg_idx1 = get_seg_idx(
                            v2, p, seg_ht_cap, seg_ht_arr, segments)
                        seg_idx2 = get_seg_idx(
                            v2, p, seg_ht_cap, seg_ht_arr, segments)
                    else:
                        seg_idx1 = get_seg_idx(
                            p, v1, seg_ht_cap, seg_ht_arr, segments)
                        seg_idx2 = get_seg_idx(
                            p, v2, seg_ht_cap, seg_ht_arr, segments)
                else:
                    if side_a_sq >= side_b_sq:
                        seg_idx1 = get_seg_idx(
                            v1, p, seg_ht_cap, seg_ht_arr, segments)
                        seg_idx2 = get_seg_idx(
                            v1, v2, seg_ht_cap, seg_ht_arr, segments)
                    else:
                        seg_idx1 = get_seg_idx(
                            p, v1, seg_ht_cap, seg_ht_arr, segments)
                        seg_idx2 = get_seg_idx(
                            p, v2, seg_ht_cap, seg_ht_arr, segments)

                if delta_t == True or (min_tri_angle < min_angle and \
                        (seg_idx1 == -1 or seg_idx2 == -1)):
                    split_tri = enqueue_st(split_tri, st_params, t, p, v1, v2)
    print("new_vertex exited")
    return split_tri, split_segs


@njit
def split_permitted(
        seg_idx, d, segments, cluster_key, points, cluster_segs, theta,
        exactinit_arr, global_arr):

    tol = 10**-4
    c = 0.01

    cluster = segments[seg_idx, 2]
    cluster1 = cluster % cluster_key
    cluster2 = cluster // cluster_key

    v1 = segments[seg_idx, 0]
    v2 = segments[seg_idx, 1]
    v1_x = points[v1, 0]
    v1_y = points[v1, 1]
    v2_x = points[v2, 0]
    v2_y = points[v2, 1]
    seg_len = ((v2_x - v1_x)*(v2_x - v1_x) + (v2_y - v1_y)*(v2_y - v1_y))**0.5
    exponent = np.log2(seg_len/c)

    if (cluster1 == 0 and cluster2 == 0) or \
            (cluster1 != 0 and cluster2 != 0) or exponent % 1.0 > tol:
        return True

    cs_end
    for i in range(num_segs):
        if segments[i, 2] % cluster_key == cluster1 or \
                segments[i, 2] // cluster_key == cluster1:
            cluster_segs[cs_end] = i
            v3 = segments[i, 0]
            v4 = segments[i, 1]
            v3_x = points[v3, 0]
            v3_y = points[v3, 1]
            v4_x = points[v4, 0]
            v4_y = points[v4, 1]
            i_len = ((v3_x - v4_x)*(v3_x - v4_x) + 
                     (v3_y - v4_y)*(v3_y - v4_y))**0.5
            if seg_len - i_len > tol:
                return True

    seg_idx = cluster_segs[0]
    if segments[si, 2] % segs_cap == cluster1:
        v1 = segments[si, 0]
        v2 = segments[s1, 1]
    else:
        v1 = segments[si, 1]
        v2 = segments[s1, 0]
    v1_x = points[v1, 0]
    v1_y = points[v1, 1]
    v2_x = points[v2, 0]
    v2_y = points[v2, 1]
    v21_x = v2_x - v1_x
    v21_y = v2_y - v1_y
    v21_norm = (v21_x*v21_x + v21_y*v21_y)**0.5
    theta[0] = 0.0
    for i in range(1, cs_end):
        si = cluster_segs[i]
        v3 = segments[si, 1]
        if v3 == v1:
            v3 = segments[si, 0]
        v3_x = points[v3, 0]
        v3_y = points[v3, 1]
        v31_x = v3_x - v1_x
        v31_y = v3_y - v1_y
        v31_norm = (v31_x*v31_x + v31_y*v31_y)**0.5
        cos_theta = (v31_x*v21_x + v31_y*v21_y)/(v31_norm*v21_norm)
        theta_i = np.arccos(cos_theta)
        area = orient2d(
            v1_x, v1_y, v2_x, v2_y, v3_x, v3_y, exactinit_arr, global_arr)
        if area < 0.0:
            theta_i *= -1.0
        theta[i] = theta_i

    delta_theta_min = theta[1] - theta[0]
    if delta_theta_min < 0.0:
        delta_theta_min *= -1.0
    for i in range(cs_end):
        theta_i = theta[i]
        for j in range(i, cs_end):
            theta_j = theta[j]
            delta_theta = theta_j - theta_i
            if delta_theta < 0.0:
                delta_theta *= -1.0
            if delta_theta < delta_theta_min:
                delta_theta_min = delta_theta

    r_min = seg_len*np.sin(0.5*delta_theta_min)
    if r_min >= d:
        return True

    return False


@njit
def split_encroached_segments(
        points, vertices_ID, neighbour_ID, segments, split_segs, ss_params,
        split_tri, st_params, seg_ht_arr, tri_ht_arr, num_entities, ht_cap,
        bad_tri, boundary_tri, boundary_vtx, bad_tri_indicator_arr,
        exactinit_arr, global_arr, min_angle, qual_f, gv):

    print("split_encroached_segments entered")
    seg_ht_cap = ht_cap[0]
    tri_ht_cap = ht_cap[1]
    num_points = num_entities[0]
    num_segs = num_entities[1]
    num_tri = num_entities[2]
    num_points_org = num_entities[3]

    c = 0.01
    while ss_params[3] != 0:
        s_info = dequeue_ss(split_segs, ss_params)
        s = s_info[0]
        v1 = s_info[1]
        v2 = s_info[2]
        # checking if 's' is still a valid segment
        if segments[s, 0] == v1 and segments[s, 1] == v2:
            print("\nyes")
            v1_x = points[v1, 0]
            v1_y = points[v1, 1]
            v2_x = points[v2, 0]
            v2_y = points[v2, 1]
            radius_square = (v1_x - v2_x)**2 + (v1_y - v2_y)**2

            # concentric shell splitting
            if v1 < num_points_org and v2 < num_points_org:
                # both v1 and v2 are input vertices
                p_x = 0.5*(v1_x + v2_x)
                p_y = 0.5*(v1_y + v2_y)
            if v1 < num_points_org and v2 >= num_points_org:
                # v1 is an input vertex
                shell_number = int(np.round(np.log2(radius_square**0.5/c)))
                radius = radius_square**0.5
                frac = c*(2**(shell_number - 1))/radius
                p_x = v1_x*(1 - frac) + v2_x*frac
                p_y = v1_y*(1 - frac) + v2_y*frac
            elif v1 >= num_points_org and v2 < num_points_org:
                # v2 is an input vertex
                shell_number = int(np.round(np.log2(radius_square**0.5/c)))
                radius = radius_square**0.5
                frac = c*(2**(shell_number - 1))/radius
                p_x = v2_x*(1 - frac) + v1_x*frac
                p_y = v2_y*(1 - frac) + v1_y*frac
            print("p : ({}, {})".format(p_x, p_y))

            # adding (p_x, p_y) to the 'points' array
            if num_points >= points.shape[0]:
                temp_pts = np.empty(
                    shape=(2*points.shape[0], 2), dtype=np.float64)
                for i in range(points.shape[0]):
                    temp_pts[i, 0] = points[i, 0]
                    temp_pts[i, 1] = points[i, 1]
                points = temp_pts
            points[num_points, 0] = p_x
            points[num_points, 1] = p_y
            point_id = num_points
            num_points += 1

            # inserting (p_x, p_y) into the triangulation
            t1, j1 = get_tri_idx(v1, v2, tri_ht_cap, tri_ht_arr, vertices_ID)
            t2 = neighbour_ID[t1, j1] // 3

            print("t1 : {}, vid : {}".format(t1, vertices_ID[t1]))
            print("t2 : {}, vid : {}".format(t2, vertices_ID[t2]))

            t1_or_t2 = 0  # -1 ==> t1
                          #  0 ==> neither
                          # +1 ==> t2
            if vertices_ID[t1, 3] == -1:
                t1_or_t2 = -1
            elif vertices_ID[t2, 3] == -1:
                t1_or_t2 = 1
            print("t1_or_t2 : {}".format(t1_or_t2))

            bad_tri, bt_end1, boundary_tri, boundary_end1, \
            boundary_vtx = _identify_cavity_CDT(
                points, point_id, t1, neighbour_ID, vertices_ID, bad_tri,
                boundary_tri, boundary_vtx, gv, bad_tri_indicator_arr,
                exactinit_arr, global_arr, seg_ht_cap, seg_ht_arr, segments,
                seg_idx=s)
            print("bad_tri from t1 : {}".format(bad_tri[0:bt_end1]))

            bad_tri, bt_end, boundary_tri, boundary_end, \
            boundary_vtx = _identify_cavity_CDT(
                points, point_id, t2, neighbour_ID, vertices_ID, bad_tri,
                boundary_tri, boundary_vtx, gv, bad_tri_indicator_arr,
                exactinit_arr, global_arr, seg_ht_cap, seg_ht_arr, segments,
                bt_end=bt_end1, boundary_end=boundary_end1, seg_idx=s)
            print("bad_tri from t2 : {}".format(bad_tri[bt_end1:bt_end]))

            new_len = num_tri + boundary_end - bt_end
            if new_len >= vertices_ID.shape[0]:
                # new_len *= 2
                temp_vid = np.empty(shape=(2*new_len, 4), dtype=np.int64)
                temp_nid = np.empty(shape=(2*new_len, 3), dtype=np.int64)
                temp_bti = np.empty(shape=2*new_len, dtype=np.bool_)
                for i in range(vertices_ID.shape[0]):
                    for j in range(3):
                        temp_vid[i, j] = vertices_ID[i, j]
                        temp_nid[i, j] = neighbour_ID[i, j]
                    temp_vid[i, 3] = vertices_ID[i, 3]
                    temp_bti[i] = False
                vertices_ID = temp_vid
                neighbour_ID = temp_nid
                bad_tri_indicator_arr = temp_bti

            delete_bad_tri_from_tri_ht(
                bad_tri, bt_end, vertices_ID, tri_ht_arr, tri_ht_cap)

            num_tri, old_tri, add_new_tri_flag = _make_Delaunay_ball_CDT(
                point_id, bad_tri, bt_end, boundary_tri, boundary_end,
                boundary_vtx, points, neighbour_ID, vertices_ID, num_tri, gv,
                tri_ht_cap, tri_ht_arr)

            for i in range(boundary_end):
                if i < bt_end:
                    t = bad_tri[i]
                else:
                    t = num_tri - (boundary_end - i)
                vertices_ID[t, 3] = 0
            if t1_or_t2 == -1:
                for i in range(boundary_end1):
                    if i < bt_end:
                        t = bad_tri[i]
                    else:
                        t = num_tri - (boundary_end - i)
                    print("post-setting vertices_ID[t, 3] for t : {}".format(t))
                    vertices_ID[t, 3] = -1
            elif t1_or_t2 == 1:
                for i in range(boundary_end1, boundary_end):
                    if i < bt_end:
                        t = bad_tri[i]
                    else:
                        t = num_tri - (boundary_end - i)
                    print("post-setting vertices_ID[t, 3] for t : {}".format(t))
                    vertices_ID[t, 3] = -1

            for i in range(bt_end):
                t = bad_tri[i]
                bad_tri_indicator_arr[t] = False

            if add_new_tri_flag == True:
                tri_ht_cap, tri_ht_arr = add_new_tri_to_tri_ht(
                    bad_tri, bt_end, boundary_end, num_tri, vertices_ID,
                    tri_ht_arr, tri_ht_cap)

            # adding v1 -- (p_x, p_y) {== s} and v2 -- (p_x, p_y) {== s_new} to
            # the 'segments' array and the 'seg_ht_arr' hash table
            cluster_key = segments.shape[0]
            if num_segs >= segments.shape[0]:
                print("expanding segments")
                temp_segs = np.empty(
                    shape=(2*segments.shape[0], 3), dtype=np.int64)
                cluster_key_new = 2*segments.shape[0]
                for i in range(segments.shape[0]):
                    temp_segs[i, 0] = segments[i, 0]
                    temp_segs[i, 1] = segments[i, 1]
                    if segments[i, 2] != -1:
                        temp_segs[i, 2] = segments[i, 2] % cluster_key + \
                                          (segments[i, 2] // cluster_key)*cluster_key_new
                    else:
                        temp_segs[i, 2] = -1
                segments = temp_segs
                cluster_key = cluster_key_new
                seg_ht_arr = -1*np.ones(
                    shape=(segments.shape[0], 2), dtype=np.int64)
                seg_ht_cap = seg_ht_arr.shape[0]
                initialize_seg_ht(seg_ht_cap, seg_ht_arr, segments, num_segs)

            segments[s, 0] = v1
            segments[s, 1] = point_id

            s_new = num_segs
            segments[s_new, 0] = v2
            segments[s_new, 1] = point_id
            segments[s_new, 2] = segments[s, 2] // cluster_key

            segments[s, 2] = segments[s, 2] % cluster_key

            num_segs += 1

            delete_entry_seg_ht(v1, v2, s, seg_ht_cap, seg_ht_arr)
            add_entry_seg_ht(v1, point_id, s, seg_ht_cap, seg_ht_arr)
            add_entry_seg_ht(v2, point_id, s_new, seg_ht_cap, seg_ht_arr)

            split_tri, split_segs = new_vertex(
                point_id, points, vertices_ID, bad_tri, bt_end,
                boundary_end, num_tri, gv, seg_ht_cap, seg_ht_arr,
                segments, split_segs, ss_params, split_tri, st_params,
                min_angle, qual_f)

            # check if 's' is encroached; if it is add it to the 'split_segs'
            # queue
            t1, j1 = get_tri_idx(
                v1, point_id, tri_ht_cap, tri_ht_arr, vertices_ID)
            # test if the point corresponding to j1 lies in i's diametral circle
            w1 = vertices_ID[t1, j1]
            proceed = True
            if w1 != -1 and vertices_ID[t1, 3] != -1:
                w1_x = points[w1, 0]
                w1_y = points[w1, 1]
                v1w1_x = v1_x - w1_x
                v1w1_y = v1_y - w1_y
                pw1_x = p_x - w1_x
                pw1_y = p_y - w1_y
                sign_of_cos_theta = v1w1_x*pw1_x + v1w1_y*pw1_y
                if sign_of_cos_theta <= 0:
                    split_segs = enqueue_ss(
                        split_segs, ss_params, s, v1, point_id)
                    proceed = False
            if proceed == True:
                t2 = neighbour_ID[t1, j1] // 3
                j2 = neighbour_ID[t1, j1] % 3
                # test if the point corresponding to j2 lies in i's diamteral
                # circle
                w2 = vertices_ID[t2, j2]
                if w2 != -1 and vertices_ID[t2, 3] != -1:
                    w2_x = points[w2, 0]
                    w2_y = points[w2, 1]
                    v1w2_x = v1_x - w2_x
                    v1w2_y = v1_y - w2_y
                    pw2_x = p_x - w2_x
                    pw2_y = p_y - w2_y
                    sign_of_cos_theta = v1w2_x*pw2_x + v1w2_y*pw2_y
                    if sign_of_cos_theta <= 0:
                        split_segs = enqueue_ss(
                            split_segs, ss_params, s, v1, point_id)

            # check if 's_new' is encroached; if it is add it to the
            # 'split_segs' queue
            t1, j1 = get_tri_idx(
                v2, point_id, tri_ht_cap, tri_ht_arr, vertices_ID)
            # test if the point corresponding to j1 lies in i's diametral
            # circle
            w1 = vertices_ID[t1, j1]
            proceed = True
            if w1 != -1 and vertices_ID[t1, 3] != -1:
                w1_x = points[w1, 0]
                w1_y = points[w1, 1]
                v2w1_x = v2_x - w1_x
                v2w1_y = v2_y - w1_y
                pw1_x = p_x - w1_x
                pw1_y = p_y - w1_y
                sign_of_cos_theta = v2w1_x*pw1_x + v2w1_y*pw1_y
                if sign_of_cos_theta <= 0:
                    split_segs = enqueue_ss(
                        split_segs, ss_params, s_new, v2, point_id)
                    proceed = False
            if proceed == True:
                t2 = neighbour_ID[t1, j1] // 3
                j2 = neighbour_ID[t1, j1] % 3
                # test if the point corresponding to j2 lies in i's diamteral
                # circle
                # print("t2 : {}, j2 : {}".format(t2, j2))
                w2 = vertices_ID[t2, j2]
                if w2 != -1 and vertices_ID[t2, 3] != -1:
                    w2_x = points[w2, 0]
                    w2_y = points[w2, 1]
                    v2w2_x = v2_x - w2_x
                    v2w2_y = v2_y - w2_y
                    pw2_x = p_x - w2_x
                    pw2_y = p_y - w2_y
                    sign_of_cos_theta = v2w2_x*pw2_x + v2w2_y*pw2_y
                    if sign_of_cos_theta <= 0:
                        split_segs = enqueue_ss(
                            split_segs, ss_params, s_new, v2, point_id)

    ht_cap[0] = seg_ht_cap
    ht_cap[1] = tri_ht_cap
    num_entities[0] = num_points
    num_entities[1] = num_segs
    num_entities[2] = num_tri

    print("split_encroached_segments exited")

    return segments, points, vertices_ID, neighbour_ID, split_segs, \
           split_tri, seg_ht_arr, tri_ht_arr


@njit
def delta_false(a_x, a_y, b_x, b_y, c_x, c_y):
    return False


@njit
def find_circumcenter(a, b, c, points):
    a_x = points[a, 0]
    a_y = points[a, 1]
    b_x = points[b, 0]
    b_y = points[b, 1]
    c_x = points[c, 0]
    c_y = points[c, 1]

    p1_x = 0.5*(b_x + a_x)
    p1_y = 0.5*(b_y + a_y)

    p2_x = 0.5*(c_x + a_x)
    p2_y = 0.5*(c_y + a_y)

    ex_1 = a_x - b_x
    ey_1 = a_y - b_y
    e0_1 = -(ex_1*p1_x + ey_1*p1_y)
    if np.abs(ex_1) > np.abs(ey_1):
        ey_1 /= ex_1
        e0_1 = -(p1_x + ey_1*p1_y)
        ex_1 = 1.0
    else:
        ex_1 /= ey_1
        e0_1 = -(ex_1*p1_x + p1_y)
        ey_1 = 1.0

    ex_2 = a_x - c_x
    ey_2 = a_y - c_y
    e0_2 = -(ex_2*p2_x + ey_2*p2_y)
    if np.abs(ex_2) > np.abs(ey_2):
        ey_2 /= ex_2
        e0_2 = -(p2_x + ey_2*p2_y)
        ex_2 = 1.0
    else:
        ex_2 /= ey_2
        e0_2 = -(ex_2*p2_x + p2_y)
        ey_2 = 1.0


    temp = ex_1*ey_2 - ey_1*ex_2
    circumcenter_x = (ey_1*e0_2 - e0_1*ey_2)/temp
    circumcenter_y = (e0_1*ex_2 - ex_1*e0_2)/temp

    return circumcenter_x, circumcenter_y


@njit
def final_assembly(
        points, vertices_ID, neighbour_ID, segments, num_entities,
        exactinit_arr, global_arr, insertion_seq, rev_insertion_seq, gv,
        bad_tri, boundary_tri, boundary_vtx, bad_tri_indicator_arr, which_side,
        polygon_vertices, new_tri, new_nbr, ht_cap, seg_ht_arr, tri_ht_arr,
        insertion_points, duplicate_segs, theta, ss_params, split_segs,
        st_params, split_tri, min_angle, qual_f, encroached_segs):

    print("assembly entered")

    num_points = num_entities[0]
    num_segs = num_entities[1]
    num_entities[3] = num_points

    ###########################################################################
    ################## CDT Construction of the Initial PSLG ###################
    ###########################################################################

    exactinit2d(points, exactinit_arr)
    print("exactinit2d done")

    num_tri = initialize(points, vertices_ID, neighbour_ID, insertion_seq, gv)
    print("initialize done")

    # build rev_insertion_seq, mapping the old indices of the 'points' array to
    # the ones after the BRIO shuffling
    for i in range(num_points):
        rev_insertion_seq[insertion_seq[i]] = i

    # assign the segment vertices new indices based on the insertion sequence
    # determined by BRIO
    for i in range(num_segs):
        segments[i, 0] = rev_insertion_seq[segments[i, 0]]
        segments[i, 1] = rev_insertion_seq[segments[i, 1]]

    # build the strictly Delaunay triangulation of the input points
    old_tri = np.int64(0)
    for point_id in range(3, gv):
        enclosing_tri = _walk(
            point_id, old_tri, vertices_ID, neighbour_ID, points, gv,
            exactinit_arr, global_arr)

        bad_tri, bt_end, boundary_tri, boundary_end, \
        boundary_vtx = _identify_cavity(
            points, point_id, enclosing_tri, neighbour_ID, vertices_ID,
            bad_tri, boundary_tri, boundary_vtx, gv, bad_tri_indicator_arr,
            exactinit_arr, global_arr)

        num_tri, old_tri = _make_Delaunay_ball(
            point_id, bad_tri, bt_end, boundary_tri, boundary_end,
            boundary_vtx, points, neighbour_ID, vertices_ID, num_tri, gv)

        for i in range(bt_end):
            t = bad_tri[i]
            bad_tri_indicator_arr[t] = False

    print("DT made")
    # build the constrained Delaunay triangulation by inserting the segments
    seg_cav_tri = bad_tri
    for i in range(num_segs):
        a = segments[i, 0]
        b = segments[i, 1]
        _insert_segment(
            points, vertices_ID, neighbour_ID, a, b, gv, exactinit_arr,
            global_arr, which_side, seg_cav_tri, boundary_tri,
            polygon_vertices, new_tri, new_nbr, num_tri)

    num_entities[2] = num_tri

    ########################## CDT Construction Ends ##########################
    print("CDT made")
    ###########################################################################
    ####################### Delaunay Refinement Begins ########################
    ###########################################################################

    seg_ht_cap = ht_cap[0]
    tri_ht_cap = ht_cap[1]

    # Build the segments hash table. This helps with fast lookups based on the
    # end-points of any given segment.
    initialize_seg_ht(seg_ht_cap, seg_ht_arr, segments, num_segs)
    cluster_key = seg_ht_cap
    print("segments hash table initialized")

    # Mark all exterior triangles, as specified by the user through the
    # 'insertion_points' array.
    reset_tri_2(
        insertion_points, vertices_ID, num_tri, neighbour_ID, points,
        num_points, segments, num_segs, exactinit_arr, global_arr, seg_ht_cap,
        seg_ht_arr, bad_tri)
    gv = -1
    print("exterior triangles marked")

    # Build the triangle hash tables. These are three hash tables keyed to the
    # three directed (in c.c.w. order) edges of a triangle. These help with
    # fast lookups based on the vertices of any given triangle.
    initialize_tri_ht(tri_ht_cap, tri_ht_arr, vertices_ID, num_tri)
    print("triangle hash table initialized")

    # Identify segment clusters, as defined in Shewchuk's Terminator algorithm.
    hist = rev_insertion_seq
    for i in range(num_points):
        hist[i] = 0
    _identify_clusters(
        points, segments, duplicate_segs, hist, theta, exactinit_arr,
        global_arr, num_segs, vertices_ID, neighbour_ID, tri_ht_cap,
        tri_ht_arr)
    print("clusters identified")

    for i in range(seg_ht_cap):
        seg_ht_arr[i, 0] = -1
        seg_ht_arr[i, 1] = -1
    initialize_seg_ht(seg_ht_cap, seg_ht_arr, segments, num_segs)    
    print("segments hash table re-initialized after shuffling by _identify_clusters")


    ####           initial identification of encroached segments           ####
    ####           ---------------------------------------------           ####

    # 'split_segs' and 'split_tri' are circular queues that keep track of
    # which segments and triangles need to be split. The arrays 'ss_params' and
    # 'st_params' keep a track of their relevant parameters. The functions
    # 'enqueue_ss', 'enqueue_st', 'dequeue_ss' and 'dequeue_st' are used to
    # interface with said queues.
    ss_params[0] = 0                    # head of the 'split_segs' queue
    ss_params[1] = 0                    # tail of the 'split_segs' queue
    ss_params[2] = split_segs.shape[0]  # capacity of the 'split_segs' queue
    ss_params[3] = 0                    # number of items in 'split_segs'

    st_params[0] = 0                   # head of the 'split_tri' queue
    st_params[1] = 0                   # tail of the 'split_tri' queue
    st_params[2] = split_tri.shape[0]  # capacity of the 'split_tri' queue
    st_params[3] = 0                   # number of items in 'split_tri'

    for i in range(num_segs):
        v1 = segments[i, 0]
        v2 = segments[i, 1]
        v1_x = points[v1, 0]
        v1_y = points[v1, 1]
        v2_x = points[v2, 0]
        v2_y = points[v2, 1]

        t1, j1 = get_tri_idx(v1, v2, tri_ht_cap, tri_ht_arr, vertices_ID)
        # test if the point corresponding to j1 lies in i's diametral circle
        test_t2 = True
        w1 = vertices_ID[t1, j1]
        if w1 != -1:
            w1_x = points[w1, 0]
            w1_y = points[w1, 1]
            v1w1_x = v1_x - w1_x
            v1w1_y = v1_y - w1_y
            v2w1_x = v2_x - w1_x
            v2w1_y = v2_y - w1_y
            sign_of_cos_theta = v1w1_x*v2w1_x + v1w1_y*v2w1_y
            if sign_of_cos_theta <= 0:
                split_segs = enqueue_ss(split_segs, ss_params, i, v1, v2)
                test_t2 = False
        if test_t2 == True:
            t2 = neighbour_ID[t1, j1] // 3
            j2 = neighbour_ID[t1, j1] % 3
            # test if the point corresponding to j2 lies in i's diamteral
            # circle
            w2 = vertices_ID[t2, j2]
            if w2 != -1:
                w2_x = points[w2, 0]
                w2_y = points[w2, 1]
                v1w2_x = v1_x - w2_x
                v1w2_y = v1_y - w2_y
                v2w2_x = v2_x - w2_x
                v2w2_y = v2_y - w2_y
                sign_of_cos_theta = v1w2_x*v2w2_x + v1w2_y*v2w2_y
                if sign_of_cos_theta <= 0:
                    # add 'i' to the queue of segments to split
                    split_segs = enqueue_ss(split_segs, ss_params, i, v1, v2)
    print(split_segs[0:ss_params[1]])


    ####                split presently encroached segments                ####
    ####                -----------------------------------                ####

    segments, points, vertices_ID, neighbour_ID, split_segs, split_tri, \
    seg_ht_arr, tri_ht_arr = split_encroached_segments(
        points, vertices_ID, neighbour_ID, segments, split_segs, ss_params,
        split_tri, st_params, seg_ht_arr, tri_ht_arr, num_entities, ht_cap,
        bad_tri, boundary_tri, boundary_vtx, bad_tri_indicator_arr,
        exactinit_arr, global_arr, 0.0, delta_false, gv)
    num_points = num_entities[0]
    num_segs = num_entities[1]
    num_tri = num_entities[2]
    cluster_key = segments.shape[0]
    seg_ht_cap = ht_cap[0]
    tri_ht_cap = ht_cap[1]


    ####             initial identification of 'bad' triangles             ####
    ####             -----------------------------------------             ####

    for i in range(num_tri):
        # checking if the traingle is an 'exterior' triangle, proceeding only
        # if it isn't
        if vertices_ID[i, 3] == 0:
            a = vertices_ID[i, 0]
            b = vertices_ID[i, 1]
            c = vertices_ID[i, 2]
            a_x = points[a, 0]
            a_y = points[a, 1]
            b_x = points[b, 0]
            b_y = points[b, 1]
            c_x = points[c, 0]
            c_y = points[c, 1]
            side_a_sq = (b_x - c_x)*(b_x - c_x) + (b_y - c_y)*(b_y - c_y)
            side_b_sq = (c_x - a_x)*(c_x - a_x) + (c_y - a_y)*(c_y - a_y)
            side_c_sq = (a_x - b_x)*(a_x - b_x) + (a_y - b_y)*(a_y - b_y)
            delta_t = qual_f(a_x, a_y, b_x, b_y, c_x, c_y)
            min_tri_angle = find_smallest_angle(
                side_a_sq, side_b_sq, side_c_sq)
            if side_a_sq >= side_b_sq and side_a_sq >= side_c_sq:
                if side_b_sq >= side_c_sq:
                    seg_idx1 = get_seg_idx(
                        c, a, seg_ht_cap, seg_ht_arr, segments)
                    seg_idx2 = get_seg_idx(
                        c, b, seg_ht_cap, seg_ht_arr, segments)
                else:
                    seg_idx1 = get_seg_idx(
                        b, a, seg_ht_cap, seg_ht_arr, segments)
                    seg_idx2 = get_seg_idx(
                        b, c, seg_ht_cap, seg_ht_arr, segments)
            elif side_b_sq >= side_a_sq and side_b_sq >= side_c_sq:
                if side_a_sq >= side_c_sq:
                    seg_idx1 = get_seg_idx(
                        c, a, seg_ht_cap, seg_ht_arr, segments)
                    seg_idx2 = get_seg_idx(
                        c, b, seg_ht_cap, seg_ht_arr, segments)
                else:
                    seg_idx1 = get_seg_idx(
                        a, b, seg_ht_cap, seg_ht_arr, segments)
                    seg_idx2 = get_seg_idx(
                        a, c, seg_ht_cap, seg_ht_arr, segments)
            else:
                if side_a_sq >= side_b_sq:
                    seg_idx1 = get_seg_idx(
                        b, a, seg_ht_cap, seg_ht_arr, segments)
                    seg_idx2 = get_seg_idx(
                        b, c, seg_ht_cap, seg_ht_arr, segments)
                else:
                    seg_idx1 = get_seg_idx(
                        a, b, seg_ht_cap, seg_ht_arr, segments)
                    seg_idx2 = get_seg_idx(
                        a, c, seg_ht_cap, seg_ht_arr, segments)

            print("t : {}, min_tri_angle : {}".format(i, min_tri_angle*180/np.pi))
            if delta_t == True or (min_tri_angle < min_angle and \
                    (seg_idx1 == -1 or seg_idx2 == -1)):
                # add 'i' to the queue of triangles to be split
                split_tri = enqueue_st(split_tri, st_params, i, a, b, c)
    print(split_tri[0:st_params[1]])
    # print("st_params : {}".format(st_params))


    ####                       main refinement loop                        ####
    ####                       --------------------                        ####

    while st_params[3] != 0:
        t_info = dequeue_st(split_tri, st_params)
        print("st_params : {}".format(st_params))
        # t = t_info[0]
        t_a = t_info[1]
        t_b = t_info[2]
        t_c = t_info[3]
        # a = vertices_ID[t, 0]
        # b = vertices_ID[t, 1]
        # c = vertices_ID[t, 2]
        tt1, tj1 = get_tri_idx(t_a, t_b, tri_ht_cap, tri_ht_arr, vertices_ID)
        tt2, tj2 = get_tri_idx(t_b, t_c, tri_ht_cap, tri_ht_arr, vertices_ID)
        tt3, tj3 = get_tri_idx(t_c, t_a, tri_ht_cap, tri_ht_arr, vertices_ID)
        print("tt1 : {}, tt2 : {}, tt3 : {}".format(tt1, tt2, tt3))
        # making sure t is still in the triangulation
        # if t_a == a and t_b == b and t_c == c:
        t = -1
        if tt1 == tt2 and tt2 == tt3:
            t = tt1
        if t != -1:
            print("t : {}".format(t))
            a = vertices_ID[t, 0]
            b = vertices_ID[t, 1]
            c = vertices_ID[t, 2]
            cc_x, cc_y = find_circumcenter(a, b, c, points)
            # 'insert' (cc_x, cc_y) into the triangulation, check if any of the
            # boundary edges of the Delaunay cavity are sub-segments. If they
            # are then (cc_x, cc_y) encroaches upon them.
            if num_points >= points.shape[0]:
                temp_pts = np.empty(
                    shape=(2*points.shape[0], 2), dtype=np.float64)
                for i in range(points.shape[0]):
                    temp_pts[i, 0] = points[i, 0]
                    temp_pts[i, 1] = points[i, 1]
                points = temp_pts
            points[num_points, 0] = cc_x
            points[num_points, 1] = cc_y
            point_id = num_points
            num_points += 1

            enclosing_tri = _walk(
                point_id, t, vertices_ID, neighbour_ID, points, gv,
                exactinit_arr, global_arr)

            bad_tri, bt_end, boundary_tri, boundary_end, \
            boundary_vtx = _identify_cavity_CDT(
                points, point_id, t, neighbour_ID, vertices_ID, bad_tri,
                boundary_tri, boundary_vtx, gv, bad_tri_indicator_arr,
                exactinit_arr, global_arr, seg_ht_cap, seg_ht_arr, segments)

            num_encroached_segs = 0
            for i in range(boundary_end):
                v1 = boundary_vtx[i, 0]
                v2 = boundary_vtx[i, 0]
                seg_idx = get_seg_idx(v1, v2, seg_ht_cap, seg_ht_arr, segments)
                if seg_idx != -1:
                    v1_x = points[v1, 0]
                    v1_y = points[v1, 1]
                    v2_x = points[v2, 0]
                    v2_y = points[v2, 1]
                    v1cc_x = v1_x - cc_x
                    v1cc_y = v1_y - cc_y
                    v2cc_x = v2_x - cc_x
                    v2cc_y = v2_y - cc_y
                    sign_of_cos_theta = v1cc_x*v2cc_x + v1cc_y*v2cc_y
                    if sign_of_cos_theta <= 0.0:
                        encroached_segs[num_encroached_segs] = seg_idx
                        num_encroached_segs += 1

            if num_encroached_segs == 0:
                # make the Delaunay ball
                new_len = num_tri + boundary_end - bt_end
                if new_len >= vertices_ID.shape[0]:
                    # new_len *= 2
                    temp_vid = np.empty(shape=(2*new_len, 4), dtype=np.int64)
                    temp_nid = np.empty(shape=(2*new_len, 3), dtype=np.int64)
                    temp_bti = np.empty(shape=2*new_len, dtype=np.bool_)
                    for i in range(vertices_ID.shape[0]):
                        for j in range(3):
                            temp_vid[i, j] = vertices_ID[i, j]
                            temp_nid[i, j] = neighbour_ID[i, j]
                        temp_vid[i, 3] = vertices_ID[i, 3]
                        temp_bti[i] = False
                    vertices_ID = temp_vid
                    neighbour_ID = temp_nid
                    bad_tri_indicator_arr = temp_bti

                delete_bad_tri_from_tri_ht(
                    bad_tri, bt_end, vertices_ID, tri_ht_arr, tri_ht_cap)

                num_tri, old_tri, add_new_tri_flag = _make_Delaunay_ball_CDT(
                    point_id, bad_tri, bt_end, boundary_tri, boundary_end,
                    boundary_vtx, points, neighbour_ID, vertices_ID, num_tri,
                    gv, tri_ht_cap, tri_ht_arr)

                for i in range(bt_end):
                    t = bad_tri[i]
                    bad_tri_indicator_arr[t] = False

                if add_new_tri_flag == True:
                    tri_ht_cap, tri_ht_arr = add_new_tri_to_tri_ht(
                        bad_tri, bt_end, boundary_end, num_tri, vertices_ID,
                        tri_ht_arr, tri_ht_cap)

                split_tri, split_segs = new_vertex(
                    point_id, points, vertices_ID, bad_tri, bt_end,
                    boundary_end, num_tri, gv, seg_ht_cap, seg_ht_arr,
                    segments, split_segs, ss_params, split_tri, st_params,
                    min_angle, qual_f)

                num_entities[0] = num_points
                num_entities[2] = num_tri
            else:
                num_points -= 1
                a_x = points[a, 0]
                a_y = points[a, 1]
                b_x = points[b, 0]
                b_y = points[b, 1]
                c_x = points[c, 0]
                c_y = points[c, 1]
                side_a_sq = (b_x - c_x)*(b_x - c_x) + (b_y - c_y)*(b_y - c_y)
                side_b_sq = (c_x - a_x)*(c_x - a_x) + (c_y - a_y)*(c_y - a_y)
                side_c_sq = (a_x - b_x)*(a_x - b_x) + (a_y - b_y)*(a_y - b_y)
                min_side_sq = side_a_sq
                if side_b_sq <= min_side_sq:
                    min_side_sq = side_b_sq
                if side_c_sq < min_side_sq:
                    min_side_sq = side_c_sq
                d = min_side_sq**0.5

                delta_t = qual_f(a_x, a_y, b_x, b_y, c_x, c_y)
                for i in range(num_encroached_segs):
                    seg_idx = encroached_segs[i]
                    sp = split_permitted(
                        seg_idx, d, segments, cluster_key, points,
                        cluster_segs, theta, exactinit_arr, global_arr)
                    if delta_t == True or sp == True:
                        # add seg_idx to split_segs
                        v1 = segments[seg_idx, 0]
                        v2 = segments[seg_idx, 1]
                        split_segs = enqueue_ss(
                            split_segs, ss_params, seg_idx, v1, v2)

                if ss_params[3] != 0:
                    # add t to split_tri
                    split_tri = enqueue_st(split_tri, st_params, t, a, b, c)

                    segments, points, vertices_ID, neighbour_ID, split_segs, \
                    split_tri, seg_ht_arr, \
                    tri_ht_arr = split_encroached_segments(
                        points, vertices_ID, neighbour_ID, segments,
                        split_segs, ss_params, split_tri, st_params,
                        seg_ht_arr, tri_ht_arr, num_entities, ht_cap, bad_tri,
                        boundary_tri, boundary_vtx, bad_tri_indicator_arr,
                        exactinit_arr, global_arr, min_angle, qual_f, gv)

                    num_points = num_entities[0]
                    num_segs = num_entities[1]
                    num_tri = num_entities[2]
                    cluster_key = segments.shape[0]
                    seg_ht_cap = ht_cap[0]
                    tri_ht_cap = ht_cap[1]

    ######################## Delaunay Refinement Ends #########################

    return points, vertices_ID, neighbour_ID, segments


class Terminator():

    def __init__(
            self, points, segments, insertion_points=None, min_angle=0.0,
            qual_f=None):
        ### handling qual_f ###
        # checking if a triangle-quality function has been supplied
        if qual_f == None:
            def f(ax, ay, bx, by, cx, cy):
                return False
            qual_f = njit(f)
            res = qual_f(1.0, 0.0, 0.0, 1.0, -1.0, 0.0)

        # checking if it is has been compiled with numba, if not then try
        # compiling in the no-python mode
        elif str(type(qual_f)) != \
                "<class 'numba.targets.registry.CPUDispatcher'>":
            qual_f = njit(qual_f)
            try:
                res = qual_f(1.0, 0.0, 0.0, 1.0, -1.0, 0.0)
            except Exception:
                raise TranspilationError(
                    qual_f.py_func.__name__,
                    'Could not successfully njit the given' + \
                    'triangle-quality function')

        else:
            object_mode = list(qual_f.overloads.values())[0][6]
            # checking if the function has been compiled in object mode, if so
            # this will not work with the rest of the code which is compiled in
            # no-python mode
            if object_mode == True:
                raise InputError(
                    qual_f.__name__,
                    'The given triangle-quality function is compiled in ' + \
                    'object mode, this will not work with the rest of the ' + \
                    'code which is compiled in no-python mode')
            else:
                try:
                    res = qual_f(1.0, 0.0, 0.0, 1.0, -1.0, 0.0)
                except Exception:
                    raise TranspilationError(
                        qual_f.py_func.__name__,
                        'The given triangle-quality function does not ' + \
                        'compile properly')
        #######################

        ### handling the points array ###
        shape = points.shape
        if len(shape) != 2:
            raise InputError(
                points, 'The supplied points array is not 2 dimensional')
        elif shape[1] != 2:
            raise InputError(
                points,
                'The supplied points array has {} '.format(shape[1]) + \
                'column{}, expected 2'.format('s' if shape[1] > 1 else ''))

        # setting up and sorting the points array
        N = points.shape[0]
        self._gv = N
        self._insertion_seq, self._points = BRIO.make_BRIO(
            np.asarray(points, dtype=np.float64))
        #################################

        # setting up the segments array
        num_segs = segments.shape[0]
        self._segments = np.empty(shape=(2*num_segs, 3), dtype=np.int64)
        self._segments[0:num_segs, 0:2] = segments
        self._segments[0:num_segs, 2] = -1


        self._vertices_ID = np.empty(shape=(2*(2*N - 2), 4), dtype=np.int64)
        self._neighbour_ID = np.empty(shape=(2*(2*N - 2), 4), dtype=np.int64)
        self._num_entities = np.asarray([N, num_segs, 0, N], dtype=np.int64)
        if insertion_points is None:
            self._insertion_points = np.asarray([])
        else:
            self._insertion_points = np.asarray(
                insertion_points, dtype=np.float64)

        min_angle *= np.pi/180

        exactinit_arr = np.empty(shape=10, dtype=np.float64)
        global_arr = np.empty(shape=3236, dtype=np.float64)
        rev_insertion_seq = np.empty(shape=N, dtype=np.int64)
        bad_tri = np.empty(shape=50, dtype=np.int64)
        boundary_tri = np.empty(shape=50, dtype=np.int64)
        boundary_vtx = np.empty(shape=(50, 2), dtype=np.int64)
        bad_tri_indicator_arr = np.zeros(shape=2*(2*N - 2), dtype=np.int64)
        which_side = np.zeros(shape=N, dtype=np.float64)
        polygon_vertices = np.empty(shape=50, dtype=np.int64)
        new_tri = np.empty(shape=(50, 3), dtype=np.int64)
        new_nbr = np.empty(shape=(50, 3), dtype=np.int64)
        ht_cap = np.asarray([2*num_segs, 2*(2*N - 2)], dtype=np.int64)
        seg_ht_arr = -1*np.ones(shape=(2*num_segs, 2), dtype=np.int64)
        tri_ht_arr = -1*np.ones(shape=(2*(2*N - 2), 6), dtype=np.int64)
        duplicate_segs = np.empty(shape=num_segs, dtype=np.int64)
        theta = np.empty(shape=num_segs, dtype=np.float64)
        ss_params = np.empty(shape=4, dtype=np.int64)
        split_segs = np.empty(shape=(2*num_segs, 3), dtype=np.int64)
        st_params = np.empty(shape=4, dtype=np.int64)
        split_tri = np.empty(shape=(2*(2*N - 2), 4), dtype=np.int64)
        encroached_segs = np.empty(shape=50, dtype=np.int64)

        # assembly
        self._points, self._vertices_ID, self._neighbour_ID, \
        self._segments = final_assembly(
            self._points, self._vertices_ID, self._neighbour_ID,
            self._segments, self._num_entities, exactinit_arr, global_arr,
            self._insertion_seq, rev_insertion_seq, self._gv,
            bad_tri, boundary_tri, boundary_vtx, bad_tri_indicator_arr,
            which_side, polygon_vertices, new_tri, new_nbr, ht_cap, seg_ht_arr,
            tri_ht_arr, self._insertion_points, duplicate_segs,
            theta, ss_params, split_segs, st_params, split_tri, min_angle,
            qual_f, encroached_segs)

    def export_tri(self):
        num_points = self._num_entities[0]
        num_tri = self._num_entities[2]
        points = self._points[0:num_points, :].copy()
        v_idx = np.where(self._vertices_ID[0:num_tri, 3] == 0)[0]
        vertices = self._vertices_ID[v_idx, 0:3].copy()
        return points, vertices


def perf_segs():

    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.25],
        [1.0, 0.5],
        [1.0, 0.75],
        [-1.0, 0.0],
        [-2.0, 0.0],
        [-3.0, 0.0],
        [2.0, 2.0]
    ], dtype=np.float64)

    segments = np.array([
        [0, 1],
        [2, 0],
        [0, 3],
        [4, 0],
        [0, 5],
        [6, 7],
        [3, 8],
        [4, 3]
    ], dtype=np.int64)
    np.random.shuffle(segments)

    new_segs = np.empty(shape=(16, 3), dtype=np.int64)
    new_segs[0:8, 0:2] = segments
    new_segs[:, 2] = -1
    hist = np.zeros(shape=9, dtype=np.int64)
    theta = np.empty(shape=8, dtype=np.float64)
    exactinit_arr = np.empty(shape=10, dtype=np.float64)
    duplicate_segs = np.empty(shape=8, dtype=np.int64)
    global_arr = np.empty(shape=3236, dtype=np.float64)

    exactinit2d(points, exactinit_arr)
    _identify_clusters(points, new_segs, duplicate_segs, hist, theta, exactinit_arr, global_arr)

    print(new_segs[0:8])

if __name__ == '__main__':
    # perf_segs()

    from CDT.DT.data import make_data
    points, segments = make_data()
    insertion_points = np.array([[0.1, 0.05]])

    tri_ = Terminator(points, segments, insertion_points, min_angle=20.0)
    points2, vertices = tri_.export_tri()
    import matplotlib.pyplot as plt
    plt.triplot(points2[:, 0], points2[:, 1], vertices)
    plt.axis('equal')
    plt.show()