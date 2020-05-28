import numpy as np
from DT.final_2D_robust_multidimarr import _walk, _identify_cavity, \
                                         _make_Delaunay_ball, initialize, \
                                         exportDT_njit
import tools.BRIO_2D_multidimarr as BRIO
from tools.adaptive_predicates import incircle, orient2d, exactinit2d
import matplotlib.pyplot as plt


def njit(f):
    return f
# from numba import njit


@njit
def _walk_to_tri_with_vtx(
        point_id, t_index, vertices_ID, neighbour_ID, points, gv, res_arr,
        global_arr):
    '''
    Walks from the given tri (t_index) to a tri with the given point as a
    vertex.

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
    ### for verbosity ###
    print('\n    _walk_to_tri_with_vtx')
    #####################

    gv_idx = 3
    if vertices_ID[t_index, 1] == gv:
        gv_idx = 1
    elif vertices_ID[t_index, 2] == gv:
        gv_idx = 2

    if gv_idx != 3:
        # t_index is a ghost tri, in this case simply step into the adjacent
        # real triangle.
        t_index = neighbour_ID[t_index, gv_idx]//3

    p_x = points[point_id, 0]
    p_y = points[point_id, 1]

    while True:
        # i.e. t_index is a real tri

        a = vertices_ID[t_index, 0]
        b = vertices_ID[t_index, 1]
        c = vertices_ID[t_index, 2]
        
        this_tri = False
        if a == point_id:
            this_tri = True
        elif b == point_id:
            this_tri = True
        elif c == point_id:
            this_tri = True

        if this_tri == True:
            break

        a_x = points[a, 0]
        a_y = points[a, 1]
        b_x = points[b, 0]
        b_y = points[b, 1]
        c_x = points[c, 0]
        c_y = points[c, 1]

        t_op_index_in_t = 4

        det = orient2d(p_x, p_y, c_x, c_y, b_x, b_y, res_arr, global_arr)
        if det > 0:
            t_op_index_in_t = 0
        else:
            det = orient2d(p_x, p_y, a_x, a_y, c_x, c_y, res_arr, global_arr)
            if det > 0:
                t_op_index_in_t = 1
            else:
                det = orient2d(
                    p_x, p_y, b_x, b_y, a_x, a_y, res_arr, global_arr)
                if det > 0:
                    t_op_index_in_t = 2

        t_index = neighbour_ID[t_index, t_op_index_in_t]//3

    ### for verbosity ###
    print(t_index)
    #####################

    return t_index


@njit
def _find_itri0(
        a, b, t_index, vertices_ID, neighbour_ID, points, res_arr, global_arr,
        which_side, gv):
    '''
    Finds the triangle that has `a` as one of its vertices and is crossed by
    the line segment `ab`. t_index is a triangle that has `a` as one of its
    vertices, and is not necessarily crossed by `ab`.
    '''
    ### for verbosity ###
    print('\n    _find_itri0')
    #####################

    a_x = points[a, 0]
    a_y = points[a, 1]
    b_x = points[b, 0]
    b_y = points[b, 1]
    while True:
        if vertices_ID[t_index, 0] == a:
            v0 = 0
        elif vertices_ID[t_index, 1] == a:
            v0 = 1
        elif vertices_ID[t_index, 2] == a:
            v0 = 2
        v1 = (v0 + 1) % 3
        v2 = (v0 + 2) % 3
        v1_vtx = vertices_ID[t_index, v1]
        v2_vtx = vertices_ID[t_index, v2]

        if v1_vtx == b or v2_vtx == b:
            t_index = -1
            break
        else:
            if v1_vtx != gv and v2_vtx != gv:
                v1_x = points[v1_vtx, 0]
                v1_y = points[v1_vtx, 1]
                v2_x = points[v2_vtx, 0]
                v2_y = points[v2_vtx, 1]

                det1 = orient2d(
                    a_x, a_y, b_x, b_y, v1_x, v1_y, res_arr, global_arr)
                det2 = orient2d(
                    a_x, a_y, b_x, b_y, v2_x, v2_y, res_arr, global_arr)
                # if (det1 < 0 and det2 < 0) or (det1 > 0 and det2 > 0):
                #     t_index = neighbour_ID[t_index, v2]//3
                # else:
                #     which_side[vertices_ID[t_index, v1]] = det1
                #     which_side[vertices_ID[t_index, v2]] = det2
                #     break
                if det1 < 0 and det2 > 0:
                    which_side[v1_vtx] = det1
                    which_side[v2_vtx] = det2
                    break
                else:
                    t_index = neighbour_ID[t_index, v2]//3
            else:
                t_index = neighbour_ID[t_index, v2]//3



    ### for verbosity ###
    print(t_index)
    #####################

    return t_index


@njit
def _find_seg_cavity(
        a, b, t_index, vertices_ID, neighbour_ID, points, res_arr, global_arr,
        which_side, seg_cav_tri, boundary_tri):
    '''
    Finds the cavity of triangles that are crossed by the segment `ab`,
    starting from the initial triangle t_index (t_index has `a` as one of its
    vertices and is crossed by `ab`).
    '''
    ### for verbosity ###
    print('\n    _find_seg_cavity')
    print('segment : {}, {}'.format(a, b))
    #####################

    a_x = points[a, 0]
    a_y = points[a, 1]
    b_x = points[b, 0]
    b_y = points[b, 1]

    sct_len = seg_cav_tri.shape[0]
    sct_end = 0
    seg_cav_tri[sct_end] = t_index
    sct_end += 1

    if vertices_ID[t_index, 0] == a:
        v0 = 0
    elif vertices_ID[t_index, 1] == a:
        v0 = 1
    elif vertices_ID[t_index, 2] == a:
        v0 = 2

    boundary_len = boundary_tri.shape[0]
    boundary_end = 0
    boundary_tri[boundary_end] = neighbour_ID[t_index, (v0 + 1) % 3]
    boundary_end += 1
    boundary_tri[boundary_end] = neighbour_ID[t_index, (v0 + 2) % 3]
    boundary_end += 1

    ### for verbosity ###
    print('-------')
    print('t_index : {}'.format(t_index))
    print('v0 : {}, v1 : {}, v2 : {}'.format(
        vertices_ID[t_index, v0],
        vertices_ID[t_index, (v0 + 1) % 3],
        vertices_ID[t_index, (v0 + 2) % 3]))
    plot_seg_cav(seg_cav_tri, sct_end, points, vertices_ID, a, b)
    #####################

    ivtx = v0
    v0 = neighbour_ID[t_index, ivtx] % 3
    t_index = neighbour_ID[t_index, ivtx]//3
    while True:
        # add t_index to seg_cav_tri array
        if sct_end >= sct_len:
            temp1 = np.empty(shape=2*sct_len, dtype=np.int64)
            for i in range(sct_len):
                temp1[i] = seg_cav_tri[i]
            seg_cav_tri = temp1
            sct_len *= 2
        seg_cav_tri[sct_end] = t_index
        sct_end += 1

        # check if any of the vertices of t_index is b itself
        b_internal_idx = 4
        if vertices_ID[t_index, 0] == b:
            b_internal_idx = 0
        elif vertices_ID[t_index, 1] == b:
            b_internal_idx = 1
        elif vertices_ID[t_index, 2] == b:
            b_internal_idx = 2

        if b_internal_idx != 4:
            v1 = (b_internal_idx + 1) % 3
            v2 = (b_internal_idx + 2) % 3

            if which_side[vertices_ID[t_index, v1]] == 0.0:
                v1_x = points[vertices_ID[t_index, v1], 0]
                v1_y = points[vertices_ID[t_index, v1], 1]
                det = orient2d(
                    a_x, a_y, b_x, b_y, v1_x, v1_y, res_arr, global_arr)
                which_side[vertices_ID[t_index, v1]] = det

            if which_side[vertices_ID[t_index, v2]] == 0.0:
                v2_x = points[vertices_ID[t_index, v2], 0]
                v2_y = points[vertices_ID[t_index, v2], 1]
                det = orient2d(
                    a_x, a_y, b_x, b_y, v2_x, v2_y, res_arr, global_arr)
                which_side[vertices_ID[t_index, v2]] = det

            if boundary_end + 1 >= boundary_len:
                temp2 = np.empty(shape=2*boundary_len, dtype=np.int64)
                for i in range(boundary_len):
                    temp2[i] = boundary_tri[i]
                boundary_tri = temp2
                boundary_len *= 2
            boundary_tri[boundary_end] = neighbour_ID[t_index, v1]
            boundary_end += 1
            boundary_tri[boundary_end] = neighbour_ID[t_index, v2]
            boundary_end += 1

            break

        else:
            v1 = (v0 + 1) % 3
            v2 = (v0 + 2) % 3

            # if vertices_ID[t_index, v0] != a:
            if which_side[vertices_ID[t_index, v0]] == 0.0:
                v0_x = points[vertices_ID[t_index, v0], 0]
                v0_y = points[vertices_ID[t_index, v0], 1]

                det0 = orient2d(
                    a_x, a_y, b_x, b_y, v0_x, v0_y, res_arr, global_arr)

                which_side[vertices_ID[t_index, v0]] = det0
            else:
                det0 = which_side[vertices_ID[t_index, v0]]

            if which_side[vertices_ID[t_index, v1]] == 0.0:
                v1_x = points[vertices_ID[t_index, v1], 0]
                v1_y = points[vertices_ID[t_index, v1], 1]

                det1 = orient2d(
                    a_x, a_y, b_x, b_y, v1_x, v1_y, res_arr, global_arr)

                which_side[vertices_ID[t_index, v1]] = det1
            else:
                det1 = which_side[vertices_ID[t_index, v1]]

            if which_side[vertices_ID[t_index, v2]] == 0.0:
                v2_x = points[vertices_ID[t_index, v2], 0]
                v2_y = points[vertices_ID[t_index, v2], 1]

                det2 = orient2d(
                    a_x, a_y, b_x, b_y, v2_x, v2_y, res_arr, global_arr)

                which_side[vertices_ID[t_index, v2]] = det2
            else:
                det2 = which_side[vertices_ID[t_index, v2]]

            if (det0 > 0 and det1 < 0) or (det0 < 0 and det1 > 0):
                vtx = v2
                vtx_for_nbr = v1
            elif (det0 > 0 and det2 < 0) or (det0 < 0 and det2 > 0):
                vtx = v1
                vtx_for_nbr = v2

            if boundary_end >= boundary_len:
                temp2 = np.empty(shape=2*boundary_len, dtype=np.int64)
                for i in range(boundary_len):
                    temp2[i] = boundary_tri[i]
                boundary_tri = temp2
                boundary_len *= 2
            boundary_tri[boundary_end] = neighbour_ID[t_index, vtx_for_nbr]
            boundary_end += 1

            v0 = neighbour_ID[t_index, vtx] % 3
            t_index = neighbour_ID[t_index, vtx]//3

            ### for verbosity ###
            plot_seg_cav(seg_cav_tri, sct_end, points, vertices_ID, a, b)
            print('-------')
            print('t_index : {}'.format(t_index))
            print('det0 : {}, det1 : {}, det2 : {}'.format(det0, det1, det2))
            print('v0 : {}, v1 : {}, v2 : {}'.format(
                vertices_ID[t_index, v0],
                vertices_ID[t_index, v1],
                vertices_ID[t_index, v2]))
            #####################

    boundary_end = fix_boundary_tri(
        boundary_tri, boundary_end, seg_cav_tri, sct_end)

    ### for verbosity ###
    plot_seg_cav(seg_cav_tri, sct_end, points, vertices_ID, a, b)
    print('-------')
    print('seg_cav_tri : {}'.format(seg_cav_tri[0:sct_end]))
    #####################

    return seg_cav_tri, sct_end, boundary_tri, boundary_end


def plot_seg_cav(seg_cav_tri, sct_end, points, vertices_ID, a, b):
    plt.triplot(
        points[:, 0],
        points[:, 1],
        vertices_ID[seg_cav_tri[0:sct_end], :]
    )
    plt.plot(
                points[[a, b], 0],
                points[[a, b], 1],
                '--',
                color='k',
            )
    for i in range(sct_end):
        t = seg_cav_tri[i]
        for j in range(3):
            plt.annotate(
                vertices_ID[t, j],
                (points[vertices_ID[t, j], 0], points[vertices_ID[t, j], 1])
            )
    plt.show()


@njit
def binary_search(t1, boundary_tri, start, end):
    arr_size = end - start
    if arr_size == 0:
        return -1
    elif arr_size == 1:
        if boundary_tri[start] // 3 == t1:
            return start
        else:
            return -1
    else:
        mid = arr_size // 2
        mid_val = boundary_tri[start + mid] // 3
        if mid_val == t1:
            return start + mid
        elif mid_val < t1:
            return binary_search(t1, boundary_tri, start + mid + 1, end)
        else:
            # midval > t1
            return binary_search(t1, boundary_tri, start, start + mid)


@njit
def qsort(boundary_tri, start, end):
    array_size = end - start
    if array_size == 2:
        # recursive base case
        if boundary_tri[start] // 3 > boundary_tri[start + 1] // 3:
            temp = boundary_tri[start]
            boundary_tri[start] = boundary_tri[start + 1]
            boundary_tri[start + 1] = temp
    else:
        pivot_idx = array_size // 2
        pivot_point = boundary_tri[start + pivot_idx] // 3

        left = -1
        right = array_size

        while left < right:
            while True:
                left += 1
                if left <= right and \
                        boundary_tri[start + left] // 3 < pivot_point:
                    pass
                else:
                    break

            while True:
                right -= 1
                if left <= right and \
                        boundary_tri[start + right] // 3 > pivot_point:
                    pass
                else:
                    break

            if left < right:
                temp = boundary_tri[start + left]
                boundary_tri[start + left] = boundary_tri[start + right]
                boundary_tri[start + right] = temp

        if left > 1:
            qsort(boundary_tri, start, start + left)
        if right < array_size - 2:
            qsort(boundary_tri, start + right + 1, end)
    return


@njit
def fix_boundary_tri(boundary_tri, boundary_end, seg_cav_tri, sct_end):
    qsort(boundary_tri, 0, boundary_end)

    for i in range(sct_end):
        t1 = seg_cav_tri[i]
        idx = binary_search(t1, boundary_tri, 0, boundary_end)
        if idx != -1:
            for j in range(idx, boundary_end - 1):
                boundary_tri[j] = boundary_tri[j + 1]
            boundary_end -= 1

    return boundary_end



@njit    
def _find_subpolygon(
    a, b, vertices_ID, neighbour_ID, seg_cav_tri, sct_end, which_side, sign,
    polygon_vertices):
    '''
    Finds the desired sub-polygon (with `ab` as one of its sides) that will be
    re-triangulated to make the valid CDT.

    sign : -1 will find the 'lower' polygon
           +1 will find the 'upper' polygon
    '''
    ### for verbosity ###
    print('\n    _find_subpolygon')
    print('sign : {} ==> {}'.format(
        sign,
        'upper polygon' if sign == 1 else 'lower polygon')
    )
    #####################

    pv_len = polygon_vertices.shape[0]
    pv_end = 0
    if sign == 1:
        polygon_vertices[pv_end] = b
        pv_end += 1
        next_tri = sct_end - 1
    else:
        polygon_vertices[pv_end] = a
        pv_end += 1
        next_tri = 0
    t_index = seg_cav_tri[next_tri]
    next_tri -= sign

    vtx = polygon_vertices[0]
    while True:
        ### for verbosity ###
        print('vtx : {}, t_index : {}, vertices_ID : {}'.format(
            vtx,
            t_index,
            vertices_ID[t_index])
        )
        #####################

        if vertices_ID[t_index, 0] == vtx:
            v0 = 0
        elif vertices_ID[t_index, 1] == vtx:
            v0 = 1
        elif vertices_ID[t_index, 2] == vtx:
            v0 = 2

        v1 = (v0 + 1) % 3
        # # check if v1 is a segment endpoint
        # if sign == 1:
        #     # check for a
        #     if vertices_ID[t_index, v1] == a:
        #         break
        # else:
        #     # check for b
        #     if vertices_ID[t_index, v1] == b:
        #         break
        if which_side[vertices_ID[t_index, v1]]*sign > 0:
            if pv_end >= pv_len:
                temp1 = np.empty(shape=2*pv_len, dtype=np.int64)
                for i in range(pv_len):
                    temp1[i] = polygon_vertices[i]
                pv_len *= 2
            polygon_vertices[pv_end] = vertices_ID[t_index, v1]
            pv_end += 1
            # t_index = neighbour_ID[t_index, vtx]//3
            vtx = vertices_ID[t_index, v1]

        if next_tri == 0 or next_tri == sct_end - 1:
            ### for verbosity ###
            print('end tri reached')
            #####################
            break

        t_index = seg_cav_tri[next_tri]
        next_tri -= sign
        print('next_tri : {}'.format(next_tri))

    if pv_end >= pv_len:
        temp1 = np.empty(shape=2*pv_len, dtype=np.int64)
        for i in range(pv_len):
            temp1[i] = polygon_vertices[i]
        pv_len *= 2
    if sign == 1:
        polygon_vertices[pv_end] = a
    else:
        polygon_vertices[pv_end] = b
    pv_end += 1

    ### for verbosity ###
    print('polygon_vertices : {}'.format(polygon_vertices[0:pv_end]))
    #####################

    return polygon_vertices, pv_end


@njit
def _triangulate_subpolygon(
        points, polygon_vertices, pv_end, new_tri, nt_end, res_arr,
        global_arr):
    '''
    Re-triangulates the sub-polygon specified by `sign` so as to respect the
    edge `ab`.
    '''
    print('\n    _triangulate_subpolygon')
    a = polygon_vertices[0]
    b = polygon_vertices[pv_end - 1]
    c = polygon_vertices[1]
    nt_len = new_tri.shape[0]
    print('polygon_vertices : {}'.format(polygon_vertices[0:pv_end]))

    if pv_end < 3:
        pass
    elif pv_end == 3:
        # recursive base case
        if nt_end >= nt_len:
            temp = np.empty(shape=(2*nt_len, 3), dtype=np.int64)
            for i in range(nt_len):
                temp[i, 0] = new_tri[i, 0]
                temp[i, 1] = new_tri[i, 1]
                temp[i, 2] = new_tri[i, 2]
            nt_len *= 2
            new_tri = temp
        new_tri[nt_end, 0] = a
        new_tri[nt_end, 1] = c
        new_tri[nt_end, 2] = b
        nt_end += 1

    else:
        a_x = points[a, 0]
        a_y = points[a, 1]
        b_x = points[b, 0]
        b_y = points[b, 1]
        print('...')
        v_pv_idx = 1
        ### for verbosity ###
        print('v_pv_idx : {}'.format(v_pv_idx))
        #####################
        for i in range(2, pv_end - 1):
            c_x = points[c, 0]
            c_y = points[c, 1]
            v = polygon_vertices[i]
            v_x = points[v, 0]
            v_y = points[v, 1]
            det_ic = incircle(
                a_x, a_y, c_x, c_y, b_x, b_y, v_x, v_y, res_arr, global_arr)
            if det_ic > 0:
                c = v
                v_pv_idx = i
                ### for verbosity ###
                print('v_pv_idx : {}'.format(v_pv_idx))
                #####################

        PE = polygon_vertices[0:v_pv_idx + 1]
        PE_end = v_pv_idx + 1
        PD = polygon_vertices[v_pv_idx:pv_end]
        PD_end = pv_end - v_pv_idx

        new_tri, nt_end = _triangulate_subpolygon(
            points, PE, PE_end, new_tri, nt_end, res_arr, global_arr)

        new_tri, nt_end = _triangulate_subpolygon(
            points, PD, PD_end, new_tri, nt_end, res_arr, global_arr)

        new_tri[nt_end, 0] = a
        new_tri[nt_end, 1] = c
        new_tri[nt_end, 2] = b
        nt_end += 1

    return new_tri, nt_end


def plot_triangulated_subcav(new_tri, nt_end, points, a, b):
    plt.triplot(
        points[:, 0],
        points[:, 1],
        new_tri[0:nt_end]
    )
    plt.plot(
                points[[a, b], 0],
                points[[a, b], 1],
                '--',
                color='k',
            )
    for i in range(nt_end):
        for j in range(3):
            plt.annotate(
                new_tri[i, j],
                (points[new_tri[i, j], 0], points[new_tri[i, j], 1])
            )
    plt.show()


@njit
def _insert_segment(
        points, vertices_ID, neighbour_ID, a, b, gv, res_arr, global_arr,
        which_side, seg_cav_tri, boundary_tri, polygon_vertices, new_tri,
        new_nbr, num_tri):

    ### for verbosity ###
    print('\n---------------')
    print('_insert_segment')
    print('segment : {} <--> {}'.format(a, b))
    #####################

    t_index = 0
    t_index = _walk_to_tri_with_vtx(
        a, t_index, vertices_ID, neighbour_ID, points, gv, res_arr,
        global_arr)
    t_index = _find_itri0(
        a, b, t_index, vertices_ID, neighbour_ID, points, res_arr, global_arr,
        which_side, gv)

    if t_index != -1:
        seg_cav_tri, sct_end, boundary_tri, boundary_end = _find_seg_cavity(
            a, b, t_index, vertices_ID, neighbour_ID, points, res_arr,
            global_arr, which_side, seg_cav_tri, boundary_tri)

        # find the 'upper' sub-polygon and triangulate it
        sign = 1
        polygon_vertices, pv_end = _find_subpolygon(
            a, b, vertices_ID, neighbour_ID, seg_cav_tri, sct_end, which_side,
            sign, polygon_vertices)
        nt_end = 0
        new_tri, nt_end = _triangulate_subpolygon(
            points, polygon_vertices, pv_end, new_tri, nt_end, res_arr,
            global_arr)

        ### for verbosity ###
        plot_triangulated_subcav(new_tri, nt_end, points, a, b)
        #####################

        # find the 'lower' sub-polygon and triangulate it
        sign = -1
        polygon_vertices, pv_end = _find_subpolygon(
            a, b, vertices_ID, neighbour_ID, seg_cav_tri, sct_end, which_side,
            sign, polygon_vertices)
        new_tri, nt_end = _triangulate_subpolygon(
            points, polygon_vertices, pv_end, new_tri, nt_end, res_arr,
            global_arr)

        ### for verbosity ###
        plot_triangulated_subcav(new_tri, nt_end, points, a, b)
        print('new_tri')
        for i in range(nt_end):
            tn = seg_cav_tri[i] if i < sct_end else num_tri + (i - sct_end)
            print('{} : {}'.format(tn, new_tri[i, :]))
        #####################


        # copy the triangles into the vertices_ID array
        vID_len = vertices_ID.shape[0]
        for i in range(nt_end):
            if i < sct_end:
                t = seg_cav_tri[i]
            else:
                t = num_tri
                num_tri += 1

            vertices_ID[t, 0] = new_tri[i, 0]
            vertices_ID[t, 1] = new_tri[i, 1]
            vertices_ID[t, 2] = new_tri[i, 2]

        # recompute the adjacencies for the newly created triangles
        nn_len = new_nbr.shape[0]
        if nn_len < nt_end:
            nn_len = new_tri.shape[0]
            temp3 = -1*np.ones(shape=(nn_len, 3), dtype=np.int64)
            new_nbr = temp3
        else:
            for i in range(nn_len):
                new_nbr[i, 0] = -1
                new_nbr[i, 1] = -1
                new_nbr[i, 2] = -1

        # computing internal adjacencies
        for i in range(nt_end):
            if i < sct_end:
                t1 = seg_cav_tri[i]
            else:
                t1 = num_tri - (nt_end - i)
            u1 = new_tri[i, 0]
            v1 = new_tri[i, 1]
            w1 = new_tri[i, 2]
            nbr1 = new_nbr[i, 0] // 3
            nbr2 = new_nbr[i, 1] // 3
            nbr3 = new_nbr[i, 2] // 3
            for j in range(nt_end):
                if i != j:
                    if j < sct_end:
                        t2 = seg_cav_tri[j]
                    else:
                        t2 = num_tri - (nt_end - j)
                    # checking if t1 and t2 are already neighbors, proceed only
                    # if they aren't
                    if nbr1 != t2 and nbr2 != t2 and nbr3 != t2:
                        u2 = new_tri[j, 0]
                        v2 = new_tri[j, 1]
                        w2 = new_tri[j, 2]
                        
                        u1_in_j = 3
                        if u1 == u2:
                            u1_in_j = 0
                        elif u1 == v2:
                            u1_in_j = 1
                        elif u1 == w2:
                            u1_in_j = 2

                        v1_in_j = 3
                        if v1 == u2:
                            v1_in_j = 0
                        elif v1 == v2:
                            v1_in_j = 1
                        elif v1 == w2:
                            v1_in_j = 2

                        w1_in_j = 3
                        if w1 == u2:
                            w1_in_j = 0
                        elif w1 == v2:
                            w1_in_j = 1
                        elif w1 == w2:
                            w1_in_j = 2

                        if u1_in_j != 3 and v1_in_j != 3:
                            if (u1_in_j + 1) % 3 == v1_in_j:
                                opp_vtx = (v1_in_j + 1) % 3
                            else:
                                opp_vtx = (u1_in_j + 1) % 3

                            new_nbr[i, 2] = 3*t2 + opp_vtx
                            new_nbr[j, opp_vtx] = 3*t1 + 2

                        elif v1_in_j != 3 and w1_in_j != 3:
                            if (v1_in_j + 1) % 3 == w1_in_j:
                                opp_vtx = (w1_in_j + 1) % 3
                            else:
                                opp_vtx = (v1_in_j + 1) % 3

                            new_nbr[i, 0] = 3*t2 + opp_vtx
                            new_nbr[j, opp_vtx] = 3*t1 + 0

                        elif w1_in_j != 3 and u1_in_j != 3:
                            if (w1_in_j + 1) % 3 == u1_in_j:
                                opp_vtx = (u1_in_j + 1) % 3
                            else:
                                opp_vtx = (w1_in_j + 1) % 3

                            new_nbr[i, 1] = 3*t2 + opp_vtx
                            new_nbr[j, opp_vtx] = 3*t1 + 1

        # computing external adjacencies
        for i in range(boundary_end):
            t1 = boundary_tri[i] // 3
            u1_idx = boundary_tri[i] % 3
            # u1 = vertices_id[t1, u1_idx]
            v1 = vertices_ID[t1, (u1_idx + 1) % 3]
            w1 = vertices_ID[t1, (u1_idx + 2) % 3]
            for j in range(nt_end):
                u2 = new_tri[j, 0]
                v2 = new_tri[j, 1]
                w2 = new_tri[j, 2]

                v1_in_j = 3
                if v1 == u2:
                    v1_in_j = 0
                elif v1 == v2:
                    v1_in_j = 1
                elif v1 == w2:
                    v1_in_j = 2

                w1_in_j = 3
                if w1 == u2:
                    w1_in_j = 0
                elif w1 == v2:
                    w1_in_j = 1
                elif w1 == w2:
                    w1_in_j = 2

                if v1_in_j != 3 and w1_in_j != 3:
                    if j < sct_end:
                        t2 = seg_cav_tri[j]
                    else:
                        t2 = num_tri - (nt_end - j)
                    if (v1_in_j + 1) % 3 == w1_in_j:
                        opp_vtx = (w1_in_j + 1) % 3
                    else:
                        opp_vtx = (v1_in_j + 1) % 3
                    neighbour_ID[t1, u1_idx] = 3*t2 + opp_vtx
                    new_nbr[j, opp_vtx] = 3*t1 + u1_idx

        # copying the adjacencies into neighbour_ID
        for i in range(nt_end):
            if i < sct_end:
                t1 = seg_cav_tri[i]
            else:
                t1 = num_tri - (nt_end - i)
            neighbour_ID[t1, 0] = new_nbr[i, 0]
            neighbour_ID[t1, 1] = new_nbr[i, 1]
            neighbour_ID[t1, 2] = new_nbr[i, 2]

    for i in range(which_side.shape[0]):
        if which_side[i] != 0.0:
            which_side[i] = 0.0
    truth_arr = check_nbrs(neighbour_ID)
    nbr_check = np.all(truth_arr.ravel())
    print('\n***nbr_check : {}'.format(nbr_check))
    if nbr_check == False:
        bad_idx = np.where(np.apply_along_axis(np.all, 1, truth_arr) == False)[0]
        print('bad_idx : {}'.format(bad_idx))
        print('boundary_tri : {}'.format(boundary_tri[0:boundary_end] // 3))
        print('new_tri : {}'.format(np.sort(seg_cav_tri[0:sct_end])))
        print('nbrs of bad_idx : \n{}'.format(neighbour_ID[bad_idx, :] // 3))
        print('opp_vtx of nbrs of bad_idx : \n{}'.format(neighbour_ID[bad_idx, :] % 3))

        # oarr = check_orientation(points, vertices_ID, res_arr, global_arr)
        # ocheck = np.all(oarr)
        # if ocheck == False:
        #     bad_idx = np.where(oarr == False)
        #     print('bad_idx (oarr) : {}'.format(bad_idx))

        import sys
        sys.exit()
    print('---------------')
    return


@njit
def check_orientation(points, vertices_ID, res_arr, global_arr):
    num_tri = vertices_ID.shape[0]
    gv = int(num_tri / 2 + 1)
    truth_arr = np.ones(shape=num_tri, dtype=np.bool_)
    for i in range(num_tri):
        u = vertices_ID[i, 0]
        v = vertices_ID[i, 1]
        w = vertices_ID[i, 2]
        if u != gv and v != gv and w != gv:
            u_x = points[u, 0]
            u_y = points[u, 1]
            v_x = points[v, 0]
            v_y = points[v, 1]
            w_x = points[w, 0]
            w_y = points[w, 1]
            det = orient2d(u_x, u_y, v_x, v_y, w_x, w_y, res_arr, global_arr)
            if det < 0:
                truth_arr[i] = False
    return truth_arr


@njit
def check_nbrs(neighbour_ID):
    num_tri = neighbour_ID.shape[0]
    truth_arr = np.ones(shape=(num_tri, 3), dtype=np.bool_)
    for t in range(num_tri):
        for j in range(3):
            nbr = neighbour_ID[t, j] // 3
            opp_vtx = neighbour_ID[t, j] % 3
            if neighbour_ID[nbr, opp_vtx] != 3*t + j:
                truth_arr[t, j] = False
    # if np.all(truth_arr.ravel()) != True:
    #     return False, 
    # else:
    #     return True
    return truth_arr


@njit
def assembly(
        points, vertices_ID, neighbour_ID, insertion_seq, segments, gv,
        ic_bad_tri, ic_boundary_tri, ic_boundary_vtx, bad_tri_indicator_arr,
        global_arr, res_arr, rev_insertion_seq, polygon_vertices, new_tri,
        new_nbr, which_side):

    exactinit2d(points, res_arr)
    num_tri = initialize(points, vertices_ID, neighbour_ID, insertion_seq)

    # building rev_insertion_seq
    for i in range(points.shape[0]):
        rev_insertion_seq[insertion_seq[i]] = i

    # assigning the segment vertices new indices based on the insertion
    # sequence determined by BRIO
    num_segs = segments.shape[0]
    for i in range(num_segs):
        segments[i, 0] = rev_insertion_seq[segments[i, 0]]
        segments[i, 1] = rev_insertion_seq[segments[i, 1]]

    # building the strictly Delaunay triangulation of the input points
    old_tri = np.int64(0)
    for point_id in range(3, gv):
        enclosing_tri = _walk(
            point_id, old_tri, vertices_ID, neighbour_ID, points, gv, res_arr,
            global_arr)

        ic_bad_tri, ic_bad_tri_end, ic_boundary_tri, ic_boundary_tri_end, \
        ic_boundary_vtx = _identify_cavity(
            points, point_id, enclosing_tri, neighbour_ID, vertices_ID,
            ic_bad_tri, ic_boundary_tri, ic_boundary_vtx, gv,
            bad_tri_indicator_arr, res_arr, global_arr)

        num_tri, old_tri = _make_Delaunay_ball(
            point_id, ic_bad_tri, ic_bad_tri_end, ic_boundary_tri,
            ic_boundary_tri_end, ic_boundary_vtx, points, neighbour_ID,
            vertices_ID, num_tri, gv)

        for i in range(ic_bad_tri_end):
            t = ic_bad_tri[i]
            bad_tri_indicator_arr[t] = False

    # building the constrained Delaunay triangulation by inserting the segments
    seg_cav_tri = ic_bad_tri
    boundary_tri = ic_boundary_tri
    for i in range(num_segs):
        a = segments[i, 0]
        b = segments[i, 1]
        _insert_segment(
            points, vertices_ID, neighbour_ID, a, b, gv, res_arr, global_arr,
            which_side, seg_cav_tri, boundary_tri, polygon_vertices, new_tri,
            new_nbr, num_tri)

    return


class CDT_2D:

    def __init__(self, points, segments):
        '''
        points : N x 2 array/list of points
        '''
        N = len(points)
        self._gv = N
        self._vertices_ID = np.empty(shape=(2*N-2, 3), dtype=np.int64)
        self._neighbour_ID = np.empty(shape=(2*N-2, 3), dtype=np.int64)
        self._insertion_seq, self._points = BRIO.make_BRIO(
            np.asarray(points, dtype=np.float64))
        self._segments = np.asarray(segments.copy(), dtype=np.int64)

        ### MAKING THE TRIANGULATION ###
        # Arrays that will be passed into the jit-ed functions so that they
        # don't have to get their hands dirty with object creation.
        ic_bad_tri = np.empty(50, dtype=np.int64)
        ic_boundary_tri = np.empty(50, dtype=np.int64)
        ic_boundary_vtx = np.empty(shape=(50, 2), dtype=np.int64)
        bad_tri_indicator_arr = np.zeros(shape=2*N-2, dtype=np.bool_)
        global_arr = np.empty(shape=3236, dtype=np.float64)
        res_arr = np.empty(shape=10, dtype=np.float64)
        polygon_vertices = np.empty(shape=50, dtype=np.int64)
        new_tri = np.empty(shape=(50, 3), dtype=np.int64)
        new_nbr = np.empty(shape=(50, 3), dtype=np.int64)
        which_side = np.zeros(shape=N, dtype=np.float64)
        rev_insertion_seq = np.empty(shape=N, dtype=np.int64)

        assembly(
            self._points, self._vertices_ID, self._neighbour_ID,
            self._insertion_seq, self._segments, self._gv, ic_bad_tri,
            ic_boundary_tri, ic_boundary_vtx, bad_tri_indicator_arr,
            global_arr, res_arr, rev_insertion_seq, polygon_vertices, new_tri,
            new_nbr, which_side)

        self.simplices = None
        self.neighbours = None

    def exportDT(self):
        N = self._gv
        num_tri = 2*N - 2
        ghost_tri = np.empty(shape=num_tri, dtype=np.int64)
        rectified_vertices = np.empty(shape=(num_tri, 3), dtype=np.int64)
        rectified_nbrs = np.empty(shape=(num_tri, 3), dtype=np.int64)
        num_rt = exportDT_njit(
            self._vertices_ID, self._neighbour_ID, self._insertion_seq,
            num_tri, ghost_tri, rectified_vertices, rectified_nbrs, self._gv)
        self.simplices = rectified_vertices[0:num_rt]
        self.neighbours = rectified_nbrs[0:num_rt]

        return self.simplices, self.neighbours


def perf(N):
    from DT.data import make_data
    import matplotlib.pyplot as plt

    points, segments = make_data()
    CDT = CDT_2D(points, segments)
    simplices, nbrs = CDT.exportDT()

    plt.triplot(points[:, 0], points[:, 1], simplices)
    for i in range(segments.shape[0]):
        a = segments[i, 0]
        b = segments[i, 1]
        plt.plot(
            [points[a, 0], points[b, 0]],
            [points[a, 1], points[b, 1]],
            '--',
            color='k',
            alpha=0.7
        )
    plt.show()



if __name__ == "__main__":
    import sys
    N = int(sys.argv[1])
    perf(N)