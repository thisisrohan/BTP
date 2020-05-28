import numpy as np
from numba import njit


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

    return t_index


@njit
def _find_itri0(
        a, b, t_index, vertices_ID, neighbour_ID, points, res_arr, global_arr,
        which_side):

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

        v1_x = points[vertices_ID[t_index, v1], 0]
        v1_y = points[vertices_ID[t_index, v1], 1]
        v2_x = points[vertices_ID[t_index, v2], 0]
        v2_y = points[vertices_ID[t_index, v2], 1]

        det1 = orient2d(a_x, a_y, b_x, b_y, v1_x, v1_y, res_arr, global_arr)
        det2 = orient2d(a_x, a_y, b_x, b_y, v2_x, v2_y, res_arr, global_arr)

        if (det1 < 0 and det2 < 0) or (det1 > 0 and det2 > 0):
            t_index = neighbour_ID[t_index, v2]//3
        else:
            which_side[vertices_ID[t_index, v1]] = det1
            which_side[vertices_ID[t_index, v2]] = det2
            # if det1 > 0:
            #     which_side[vertices_ID[t_index, v1]] = 1
            # else:
            #     which_side[vertices_ID[t_index, v1]] = -1
            # if det2 > 0:
            #     which_side[vertices_ID[t_index, v2]] = 1
            # else:
            #     which_side[vertices_ID[t_index, v2]] = -1
            break

    return t_index


@njit
def _find_seg_cavity(
        a, b, t_index, vertices_ID, neighbour_ID, points, res_arr, global_arr,
        which_side, seg_cav_tri):

    a_x = points[a, 0]
    a_y = points[a, 1]
    b_x = points[b, 0]
    b_y = points[b, 1]

    sct_len = len(seg_cav_tri)
    sct_end = 0
    seg_cav_tri[sct_end] = t_index
    sct_end += 1

    if vertices_ID[t_index, 0] == a:
        v0 = 0
    elif vertices_ID[t_index, 1] == a:
        v0 = 1
    elif vertices_ID[t_index, 2] == a:
        v0 = 2

    v0 = neighbour_ID[t_index, v0] % 3
    t_index = neighbour_ID[t_index, v0]//3
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
        if vertices[t_index, 0] == b:
            b_internal_idx = 0
        elif vertices[t_index, 1] == b:
            b_internal_idx = 1
        elif vertices[t_index, 2] == b:
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
                # if det > 0:
                #     which_side[vertices_ID[t_index, v1]] = 1
                # else:
                #     which_side[vertices_ID[t_index, v1]] = -1

            if which_side[vertices_ID[t_index, v2]] == 0.0:
                v2_x = points[vertices_ID[t_index, v2], 0]
                v2_y = points[vertices_ID[t_index, v2], 1]
                det = orient2d(
                    a_x, a_y, b_x, b_y, v1_x, v1_y, res_arr, global_arr)
                which_side[vertices_ID[t_index, v2]] = det
                # if det > 0:
                #     which_side[vertices_ID[t_index, v2]] = 1
                # else:
                #     which_side[vertices_ID[t_index, v2]] = -1

            break

        else:
            v1 = (v0 + 1) % 3
            v2 = (v0 + 2) % 3

            if which_side[vertices_ID[t_index, v0]] == 0.0:
                v0_x = points[vertices_ID[t_index, v0], 0]
                v0_y = points[vertices_ID[t_index, v0], 1]

                det0 = orient2d(
                    a_x, a_y, b_x, b_y, v0_x, v0_y, res_arr, global_arr)

                which_side[vertices_ID[t_index, v0]] = det0
                # if det0 > 0:
                #     which_side[vertices_ID[t_index, v0]] = 1
                # else:
                #     which_side[vertices_ID[t_index, v0]] = -1
            else:
                det0 = which_side[vertices_ID[t_index, v0]]

            if which_side[vertices_ID[t_index, v1]] == 0.0:
                v1_x = points[vertices_ID[t_index, v1], 0]
                v1_y = points[vertices_ID[t_index, v1], 1]

                det1 = orient2d(
                    a_x, a_y, b_x, b_y, v1_x, v1_y, res_arr, global_arr)

                which_side[vertices_ID[t_index, v1]] = det1
                # if det1 > 0:
                #     which_side[vertices_ID[t_index, v1]] = 1
                # else:
                #     which_side[vertices_ID[t_index, v1]] = -1
            else:
                det1 = which_side[vertices_ID[t_index, v1]]

            if which_side[vertices_ID[t_index, v2]] == 0.0:
                v2_x = points[vertices_ID[t_index, v2], 0]
                v2_y = points[vertices_ID[t_index, v2], 1]

                det2 = orient2d(
                    a_x, a_y, b_x, b_y, v2_x, v2_y, res_arr, global_arr)

                which_side[vertices_ID[t_index, v2]] = det2
                # if det2 > 0:
                #     which_side[vertices_ID[t_index, v2]] = 1
                # else:
                #     which_side[vertices_ID[t_index, v2]] = -1
            else:
                det2 = which_side[vertices_ID[t_index, v2]]

            if (det0 > 0 and det1 < 0) or (det0 < 0 and det1 > 0):
                vtx = v2
            elif (det0 > 0 and det2 < 0) or (det0 < 0 and det2 > 0):
                vtx = v1

            v0 = neighbour_ID[t_index, vtx] % 3
            t_index = neighbour_ID[t_index, vtx]//3

    return seg_cav_tri, sct_end


@njit    
def _find_polygon(
    a, b, vertices_ID, neighbour_ID, seg_cav_tri, sct_end, which_side, sign,
    polygon_vertices):
    '''
    sign : -1 will find the lower polygon
           +1 will find the upper polygon
    '''
    pv_len = len(polygon_vertices)
    pv_end = 0
    if sign == 1:
        polygon_vertices[pv_end] = b
        pv_end += 1
        next_tri = sct_end-1
    else:
        polygon_vertices[pv_end] = a
        pv_end += 1
        next_tri = 0
    t_index = seg_cav_tri[next_tri]
    next_tri -= sign

    vtx = polygon_vertices[0]
    while True:
        if vertices_ID[t_index, 0] == vtx:
            v0 = 0
        elif vertices_ID[t_index, 1] == vtx:
            v0 = 1
        elif vertices_ID[t_index, 2] == vtx:
            v0 = 2

        v1 = (v0 + 1) % 3
        # check if v1 is a segment endpoint
        if sign == 1:
            # check for a
            if vertices_ID[t_index, v1] == a:
                break
        else:
            # check for b
            if vertices_ID[t_index, v1] == b:
                break

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

        t_index = seg_cav_tri[next_tri]
        next_tri -= sign

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

    return polygon_vertices, pv_end


@njit
def _insert_vertex(
        u_idx_in_pv, v_idx_in_pv, w_idx_in_pv, polygon_vertices, pv_end,
        new_tri, nt_end, points, marked_pv, res_arr, global_arr):

    nt_len = new_tri.shape[0]

    u = polygon_vertices[u_idx_in_pv]
    v = polygon_vertices[v_idx_in_pv]
    w = polygon_vertices[w_idx_in_pv]

    X_idx_in_pv = -1
    X = -1
    t_wvX = -1
    for i in range(nt_end):
        v_idx = 3
        if polygon_vertices[new_tri[i, 0]] == v:
            v_idx = 0
        elif polygon_vertices[new_tri[i, 1]] == v:
            v_idx = 1
        elif polygon_vertices[new_tri[i, 2]] == v:
            v_idx = 2

        if v_idx != 3:
            w_idx = 3
            if polygon_vertices[new_tri[i, (v_idx + 1) % 3]] == w:
                w_idx = 0
            elif polygon_vertices[new_tri[i, (v_idx + 2) % 3]] == w:
                w_idx = 1

            if w_idx != 3:
                if w_idx == 0:
                    X_idx_in_pv = new_tri[i, (v_idx + 2) % 3]
                else:
                    X_idx_in_pv = new_tri[i, (v_idx + 1) % 3]
                X = polygon_vertices[X_idx_in_pv]
                t_wvX = i
                break

    u_x = points[u, 0]
    u_y = points[u, 1]
    v_x = points[v, 0]
    v_y = points[v, 1]
    w_x = points[w, 0]
    w_y = points[w, 1]

    det_o2d = orient2d(u_x, u_y, v_x, v_y, w_x, w_y, res_arr, global_arr)

    det_ic = 1
    if X != -1:
        X_x = points[X, 0]
        X_y = points[X, 1]
        det_ic = incircle(
            w_x, w_y, v_x, v_y, X_x, X_y, u_x, u_y, res_arr, global_arr)

        for i in range(0, pv_end):
            if X == polygon_vertices[i]:
                X_idx_in_pv = i
                break

    if X == -1 or (det_ic <= 0 and det_o2d > 0):
        # create triangle u-v-w
        if nt_end >= nt_len:
            temp1 = np.empty(shape=(2*nt_len, 3), dtype=np.int64)
            for i in range(nt_len):
                temp1[i, 0] = new_tri[i, 0]
                temp1[i, 1] = new_tri[i, 1]
                temp1[i, 2] = new_tri[i, 2]
            nt_len *= 2
            new_tri = temp1
        new_tri[nt_end, 0] = u_idx_in_pv
        new_tri[nt_end, 1] = v_idx_in_pv
        new_tri[nt_end, 2] = w_idx_in_pv
        nt_end += 1
    else:
        # delete triangle w-v-X
        for i in range(t_wvX, nt_end - 1):
            new_tri[i, 0] = new_tri[i + 1, 0]
            new_tri[i, 1] = new_tri[i + 1, 1]
            new_tri[i, 2] = new_tri[i + 1, 2]
        nt_end -= 1

        # insert_vertex(u, v, X)
        new_tri, nt_end = _insert_vertex(
            u_idx_in_pv, v_idx_in_pv, X_idx_in_pv, polygon_vertices, pv_end,
            new_tri, nt_end, points, marked_pv, res_arr, global_arr)
        # insert_vertex(u, X, w)
        new_tri, nt_end = _insert_vertex(
            u_idx_in_pv, X_idx_in_pv, w_idx_in_pv, polygon_vertices, pv_end,
            new_tri, nt_end, points, marked_pv, res_arr, global_arr)

        if det_ic <= 0:
            # 'mark' u, v, w, X to be retriangulated later
            marked_pv[u_idx_in_pv] = True
            marked_pv[v_idx_in_pv] = True
            marked_pv[w_idx_in_pv] = True
            marked_pv[X_idx_in_pv] = True

    return new_tri, nt_end


@njit
def _Chew_algo_retri():

    C_pv_end = 0
    for i in range(pv_end):
        if marked_pv[i] == True:
            C_pv[C_pv_end] = i
            C_pv_end += 1

    # C_tri_end = 0
    # for i in range(nt_end):
    #     if marked_pv[new_tri[i, 0]] == True and \
    #             marked_pv[new_tri[i, 1]] == True and \
    #             marked_pv[new_tri[i, 2]] == True:
    #         C_tri[C_tri_end] = i
    #         C_tri_end += 1

    for i in range(1, C_pv_end - 2):
        C_next_arr[i] = i + 1
        C_prev_arr[i] = i - 1
        C_pi[i] = i

    C_next_arr[0] = 1
    C_prev_arr[0] = C_pv_end - 1
    C_pv[0] = 0

    C_next_arr[C_pv_end - 1] = 0
    C_prev_arr[C_pv_end - 1] = C_pv_end - 2
    C_pv[C_pv_end - 1] = C_pv_end - 1

    for i in range(C_pv_end - 2, 1, -1):
        j = np.random.randint(low=0, high=i + 1)
        C_next_arr[C_prev_arr[C_pi[j]]] = C_next_arr[C_pi[j]]
        C_prev_arr[C_next_arr[C_pi[j]]] = C_prev_arr[C_pi[j]]

        temp = C_pi[j]
        C_pi[j] = C_pi[i]
        C_pi[i] = temp

    # C_tri_idx = 0
    # new_tri[C_tri[C_tri_idx - 1], 0] = C_pv[0]
    # new_tri[C_tri[C_tri_idx - 1], 1] = C_pv[C_pi[1]]
    # new_tri[C_tri[C_tri_idx - 1], 2] = C_pv[C_pv_end - 1]
    # new_nbr[C_tri[C_tri_idx - 1], 0] = -1
    # new_nbr[C_tri[C_tri_idx - 1], 0] = -1
    # new_nbr[C_tri[C_tri_idx - 1], 0] = -1
    # C_tri_idx += 1

    C_tri_end = 0
    C_tri[C_tri_end, 0] = C_pv[0]
    C_tri[C_tri_end, 1] = C_pv[C_pi[1]]
    C_tri[C_tri_end, 2] = C_pv[C_pv_end - 1]
    C_nbr[C_tri_end, 0] = -1
    C_nbr[C_tri_end, 1] = -1
    C_nbr[C_tri_end, 2] = -1
    C_tri_end += 1

    for i in range(2, C_pv_end-1):
        # inserting u into the triangulation, which is opposite to edge v-w
        u_idx_in_pv = C_pv[C_pi[i]]
        v_idx_in_pv = C_pv[C_next_arr[C_pi[i]]]
        w_idx_in_pv = C_pv[C_prev_arr[C_pi[i]]]

        u = polygon_vertices[u_idx_in_pv]
        v = polygon_vertices[v_idx_in_pv]
        w = polygon_vertices[w_idx_in_pv]

        t_opp_vw_found = False
        # find triangle with edge v-w
        for j in range(C_tri_end):
            t = C_tri[j]
            if t_opp_vw_found == False:
                v_idx = 3
                if polygon_vertices[new_tri[t, 0]] == v:
                    v_idx = 0
                elif polygon_vertices[new_tri[t, 1]] == v:
                    v_idx = 1
                elif polygon_vertices[new_tri[t, 2]] == v:
                    v_idx = 2
                if v_idx != 3:
                    w_idx = 3
                    if polygon_vertices[new_tri[t, (v_idx + 1) % 3]] == w:
                        w_idx = (v_idx + 1) % 3
                    elif polygon_vertices[new_tri[t, (v_idx + 2) % 3]] == w:
                        w_idx = (v_idx + 2) % 3
                    if w_idx != 3:
                        t_opp_vw_found = True
                        t_opp_vw = t
                        break

            a = polygon_vertices[]

        # find all the bad triangles





@njit
def _triangulate_seg_cavity(
    a, b, vertices_ID, neighbour_ID, seg_cav_tri, sct_end, which_side, sign,
    polygon_vertices, pv_end, pi, next_arr, prev_arr, new_tri
):

    nt_len = len(new_tri)
    nt_end = 0

    for i in range(1, pv_end - 2):
        next_arr[i] = i + 1
        prev_arr[i] = i - 1
        distance[i] = sign*which_side[polygon_vertices[i]]
        pi[i] = i
    distance[0] = 0.0
    distance[pv_end - 1] = 0.0

    for i in range(pv_end - 2, 1, -1):
        while True:
            j = np.random.randint(low=1, high=i + 1)
            if distance[pi[j]] < distance[prev_arr[pi[j]]] and \
                    distance[pi[j]] < distance[next_arr[pi[j]]]:
                pass
            else:
                break
        next_arr[prev_arr[pi[j]]] = next_arr[pi[j]]
        prev_arr[next_arr[pi[j]]] = prev_arr[pi[j]]

        temp = pi[j]
        pi[j] = pi[i]
        pi[i] = temp

    # create triangle v_{0}, v_{pi[1]}, v_{pv_end-1}
    new_tri[nt_end, 0] = 0
    new_tri[nt_end, 1] = pi[1]
    new_tri[nt_end, 2] = pv_end - 1

    for i in range(2, pv_end - 2):
        new_tri, nt_end = insert_vertex(
            pi[i], next_arr[pi[i]], prev_arr[pi[i]], polygon_vertices, pv_end,
            new_tri, nt_end, points, marked_pv, res_arr, global_arr)
        # nt_len = new_tri.shape[0] #--> is this needed?

        if marked_pv[polygon_vertices[pi[i]]] == True:
            # call Chew's algorithm to retriangulate the fan of triangles with
            # all 3 vertices marked
