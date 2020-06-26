import numpy as np
from numba import njit


def _walk(
        point_x, point_y, t_index, vertices_ID, neighbour_ID, points, gv,
        exactinit_arr, global_arr):
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
        t_index = neighbour_ID[t_index, gv_idx] // 3

    while True:
        # i.e. t_index is a real tri

        t_op_index_in_t = 4

        a_x = points[vertices_ID[t_index, 0], 0]
        a_y = points[vertices_ID[t_index, 0], 1]
        b_x = points[vertices_ID[t_index, 1], 0]
        b_y = points[vertices_ID[t_index, 1], 1]
        c_x = points[vertices_ID[t_index, 2], 0]
        c_y = points[vertices_ID[t_index, 2], 1]

        
        det = orient2d(
            point_x, point_y, c_x, c_y, b_x, b_y, exactinit_arr, global_arr)
        if det > 0:
            t_op_index_in_t = 0
        else:
            det = orient2d(
                point_x, point_y, a_x, a_y, c_x, c_y, exactinit_arr,
                global_arr)
            if det > 0:
                t_op_index_in_t = 1
            else:
                det = orient2d(
                    point_x, point_y, b_x, b_y, a_x, a_y, exactinit_arr,
                    global_arr)
                if det > 0:
                    t_op_index_in_t = 2

        if t_op_index_in_t != 4:
            t_index = neighbour_ID[t_index, t_op_index_in_t] // 3
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
def reset_tri(
        insertion_points, vertices_ID, num_tri, neighbour_ID, points,
        num_points, segments, num_segments, tri_to_be_deleted, exactinit_arr,
        global_arr, seg_ht_cap, seg_ht_arr):
    '''
    insertion_points : k x 2 array (virus introduced at these k points)
    '''
    # print(insertion_points)
    num_viral_points = insertion_points.shape[0]
    num_tri_to_be_deleted = 0
    old_tri = 0
    gv = num_points

    for k in range(num_viral_points):
        tri_iter = num_tri_to_be_deleted

        insertion_point_x = insertion_points[k, 0]
        insertion_point_y = insertion_points[k, 1]

        enclosing_tri = _walk(
            insertion_point_x, insertion_point_y, old_tri, vertices_ID,
            neighbour_ID, points, gv, exactinit_arr, global_arr)
        # old_tri = enclosing_tri

        if num_tri_to_be_deleted >= len(tri_to_be_deleted):
            # checking if the array has space for another element
            temp_arr_del_tri = np.empty(
                2*num_tri_to_be_deleted, dtype=np.int64)
            for l in range(num_tri_to_be_deleted):
                temp_arr_del_tri[l] = tri_to_be_deleted[l]
            tri_to_be_deleted = temp_arr_del_tri

        tri_to_be_deleted[num_tri_to_be_deleted] = enclosing_tri
        num_tri_to_be_deleted += 1

        # last_tri = enclosing_tri

        while tri_iter < num_tri_to_be_deleted:

            tri_idx = tri_to_be_deleted[tri_iter]

            a_idx = vertices_ID[tri_idx, 0]
            b_idx = vertices_ID[tri_idx, 1]
            c_idx = vertices_ID[tri_idx, 2]

            if a_idx == gv:
                vertices_ID[tri_idx, 0] = -1
            elif b_idx == gv:
                vertices_ID[tri_idx, 1] = -1
            elif c_idx == gv:
                vertices_ID[tri_idx, 2] = -1

            nbr_a = neighbour_ID[tri_idx, 0] // 3
            del_nbr_a = True
            nbr_b = neighbour_ID[tri_idx, 1] // 3
            del_nbr_b = True
            nbr_c = neighbour_ID[tri_idx, 2] // 3
            del_nbr_c = True

            seg_idx1 = get_seg_idx(
                a_idx, b_idx, seg_ht_cap, seg_ht_arr, segments)
            seg_idx2 = get_seg_idx(
                b_idx, c_idx, seg_ht_cap, seg_ht_arr, segments)
            seg_idx3 = get_seg_idx(
                c_idx, a_idx, seg_ht_cap, seg_ht_arr, segments)

            if seg_idx1 != -1:
                del_nbr_c = False
            if seg_idx2 != -1:
                del_nbr_a = False
            if seg_idx3 != -1:
                del_nbr_b = False

            for i in range(num_tri_to_be_deleted):
                temp_tri = tri_to_be_deleted[i]
                if nbr_a == temp_tri:
                    del_nbr_a = False
                if nbr_b == temp_tri:
                    del_nbr_b = False
                if nbr_c == temp_tri:
                    del_nbr_c = False

            if del_nbr_a == True:
                if num_tri_to_be_deleted >= len(tri_to_be_deleted):
                    # checking if the array has space for another element
                    temp_arr_del_tri = np.empty(
                        2*num_tri_to_be_deleted, dtype=np.int64)
                    for l in range(num_tri_to_be_deleted):
                        temp_arr_del_tri[l] = tri_to_be_deleted[l]
                    tri_to_be_deleted = temp_arr_del_tri

                tri_to_be_deleted[num_tri_to_be_deleted] = nbr_a
                num_tri_to_be_deleted += 1

            if del_nbr_b == True:
                if num_tri_to_be_deleted >= len(tri_to_be_deleted):
                    # checking if the array has space for another element
                    temp_arr_del_tri = np.empty(
                        2*num_tri_to_be_deleted, dtype=np.int64)
                    for l in range(num_tri_to_be_deleted):
                        temp_arr_del_tri[l] = tri_to_be_deleted[l]
                    tri_to_be_deleted = temp_arr_del_tri

                tri_to_be_deleted[num_tri_to_be_deleted] = nbr_b
                num_tri_to_be_deleted += 1

            if del_nbr_c == True:
                if num_tri_to_be_deleted >= len(tri_to_be_deleted):
                    # checking if the array has space for another element
                    temp_arr_del_tri = np.empty(
                        2*num_tri_to_be_deleted, dtype=np.int64)
                    for l in range(num_tri_to_be_deleted):
                        temp_arr_del_tri[l] = tri_to_be_deleted[l]
                    tri_to_be_deleted = temp_arr_del_tri

                tri_to_be_deleted[num_tri_to_be_deleted] = nbr_c
                num_tri_to_be_deleted += 1

            # print("num_tri_to_be_deleted : {}".format(num_tri_to_be_deleted))
            tri_iter += 1

    quicksort(tri_to_be_deleted, 0, num_tri_to_be_deleted)

    for i in range(num_tri_to_be_deleted):
        tri = tri_to_be_deleted[i]
        idx = i
        if tri != i:
            tri_j = 4
            idx_j = 4
            for j in range(3):
                if neighbour_ID[tri, j] // 3 == idx:
                    tri_j = j
                if neighbour_ID[idx, j] // 3 == tri:
                    idx_j = j

            if tri_j == 4 and idx_j == 4:
                for j in range(3):
                    temp_tri = neighbour_ID[tri, j] // 3
                    temp_j = neighbour_ID[tri, j] % 3
                    neighbour_ID[temp_tri, temp_j] = 3*idx + j
                for j in range(3):
                    temp_tri = neighbour_ID[idx, j] // 3
                    temp_j = neighbour_ID[idx, j] % 3
                    neighbour_ID[temp_tri, temp_j] = 3*tri + j
                for j in range(3):
                    temp = vertices_ID[tri, j]
                    vertices_ID[tri, j] = vertices_ID[idx, j]
                    vertices_ID[idx, j] = temp

                    temp = neighbour_ID[tri, j]
                    neighbour_ID[tri, j] = neighbour_ID[idx, j]
                    neighbour_ID[idx, j] = temp
            else:
                for j in range(3):
                    if j != tri_j:
                        temp_tri = neighbour_ID[tri, j] // 3
                        temp_j = neighbour_ID[tri, j] % 3
                        neighbour_ID[temp_tri, temp_j] = 3*idx + j
                for j in range(3):
                    if j != idx_j:
                        temp_tri = neighbour_ID[idx, j] // 3
                        temp_j = neighbour_ID[idx, j] % 3
                        neighbour_ID[temp_tri, temp_j] = 3*tri + j
                for j in range(3):
                    temp = vertices_ID[tri, j]
                    vertices_ID[tri, j] = vertices_ID[idx, j]
                    vertices_ID[idx, j] = temp

                    temp = neighbour_ID[tri, j]
                    neighbour_ID[tri, j] = neighbour_ID[idx, j]
                    neighbour_ID[idx, j] = temp                

                neighbour_ID[tri, idx_j] = 3*idx + tri_j
                neighbour_ID[idx, tri_j] = 3*tri + idx_j

            tri_to_be_deleted[i] = idx
            for j in range(i + 1, num_tri_to_be_deleted):
                if tri_to_be_deleted[j] == idx:
                    tri_to_be_deleted[j] = tri

    return num_tri_to_be_deleted


@njit
def reset_tri_2(
        insertion_points, vertices_ID, num_tri, neighbour_ID, points,
        num_points, segments, num_segments, exactinit_arr, global_arr,
        seg_ht_cap, seg_ht_arr):
    '''
    insertion_points : k x 2 array (virus introduced at these k points)
    '''
    # print(insertion_points)
    num_viral_points = insertion_points.shape[0]
    num_tri_to_be_deleted = 0
    old_tri = 0
    gv = num_points

    for i in range(num_tri):
        vertices_ID[i, 3] = 0
        if vertices_ID[i, 0] == gv:
            vertices_ID[i, 0] = -1
            vertices_ID[i, 3] = -1
        elif vertices_ID[i, 1] == gv:
            vertices_ID[i, 1] = -1
            vertices_ID[i, 3] = -1
        elif vertices_ID[i, 1] == gv:
            vertices_ID[i, 2] = -1
            vertices_ID[i, 3] = -1
    gv = -1

    for k in range(num_viral_points):
        tri_iter = num_tri_to_be_deleted

        insertion_point_x = insertion_points[k, 0]
        insertion_point_y = insertion_points[k, 1]

        enclosing_tri = _walk(
            insertion_point_x, insertion_point_y, old_tri, vertices_ID,
            neighbour_ID, points, gv, exactinit_arr, global_arr)
        # old_tri = enclosing_tri

        if num_tri_to_be_deleted >= len(tri_to_be_deleted):
            # checking if the array has space for another element
            temp_arr_del_tri = np.empty(
                2*num_tri_to_be_deleted, dtype=np.int64)
            for l in range(num_tri_to_be_deleted):
                temp_arr_del_tri[l] = tri_to_be_deleted[l]
            tri_to_be_deleted = temp_arr_del_tri

        tri_to_be_deleted[num_tri_to_be_deleted] = enclosing_tri
        num_tri_to_be_deleted += 1

        # last_tri = enclosing_tri

        while tri_iter < num_tri_to_be_deleted:

            tri_idx = tri_to_be_deleted[tri_iter]

            a_idx = vertices_ID[tri_idx, 0]
            b_idx = vertices_ID[tri_idx, 1]
            c_idx = vertices_ID[tri_idx, 2]

            if vertices_ID[tri_idx, 3] == -1:
                # THE USER HAS SPECIFIED INSERTION POINTS OUTSIDE THE DOMAIN OF
                # TRIANGULATION IF THE ABOVE STATEMENT EVALUATES TO TRUE
                print('THE FOLLOWING INSERTION POINT LIES OUTSIDE THE DOMAIN OF TRIANGULATION :')
                print(insertion_points[k])
                if a_idx == gv:
                    seg_idx1 = -1
                    seg_idx2 = get_seg_idx(
                        b_idx, c_idx, seg_ht_cap, seg_ht_arr, segments)
                    seg_idx3 = -1
                elif b_idx == gv:
                    seg_idx1 = -1
                    seg_idx2 = -1
                    seg_idx3 = get_seg_idx(
                        c_idx, a_idx, seg_ht_cap, seg_ht_arr, segments)
                elif c_idx == gv:
                    seg_idx1 = get_seg_idx(
                        a_idx, b_idx, seg_ht_cap, seg_ht_arr, segments)
                    seg_idx2 = -1
                    seg_idx3 = -1
            else:
                seg_idx1 = get_seg_idx(
                    a_idx, b_idx, seg_ht_cap, seg_ht_arr, segments)
                seg_idx2 = get_seg_idx(
                    b_idx, c_idx, seg_ht_cap, seg_ht_arr, segments)
                seg_idx3 = get_seg_idx(
                    c_idx, a_idx, seg_ht_cap, seg_ht_arr, segments)

            nbr_a = neighbour_ID[tri_idx, 0] // 3
            del_nbr_a = True
            if vertices_ID[nbr_a, 3] == -1
                del_nbr_a = False

            nbr_b = neighbour_ID[tri_idx, 1] // 3
            del_nbr_b = True
            if vertices_ID[nbr_b, 3] == -1
                del_nbr_a = False

            nbr_c = neighbour_ID[tri_idx, 2] // 3
            del_nbr_c = True
            if vertices_ID[nbr_c, 3] == -1
                del_nbr_a = False

            if seg_idx1 != -1:
                del_nbr_c = False
            if seg_idx2 != -1:
                del_nbr_a = False
            if seg_idx3 != -1:
                del_nbr_b = False

            if del_nbr_a == True:
                vertices_ID[nbr_a, 3] = -1
            if del_nbr_b == True:
                vertices_ID[nbr_b, 3] = -1
            if del_nbr_c == True:
                vertices_ID[nbr_c, 3] = -1
    return