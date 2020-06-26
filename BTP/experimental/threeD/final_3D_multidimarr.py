import numpy as np
import BTP.experimental.threeD.tools.BRIO_3D as BRIO
from BTP.experimental.threeD.tools.adaptive_predicates import exactinit3d, orient3d, insphere


def njit(f):
    print("numba NOT used")
    return f
from numba import njit


@njit
def cross_pdt(a_x, a_y, a_z, b_x, b_y, b_z):
    return a_y*b_z-a_z*b_y, -a_x*b_z+a_z*b_x, a_x*b_y-a_y*b_x


@njit
def _calculate_sub_dets(points, final_sub_dets):
    '''
    Calculates and returns the sub-determinants of the given tet.

            points : Co-ordinates of the vertices of the tet.
    final_sub_dets : The array to be returned, it contains all the
                     final values.
    '''

    abx_ = points[1, 0] - points[0, 0]  # b_x-a_x
    aby_ = points[1, 1] - points[0, 1]  # b_y-a_y
    abz_ = points[1, 2] - points[0, 2]  # b_z-a_z
    acx_ = points[2, 0] - points[0, 0]  # c_x-a_x
    acy_ = points[2, 1] - points[0, 1]  # c_y-a_y
    acz_ = points[2, 2] - points[0, 2]  # c_z-a_z
    adx_ = points[3, 0] - points[0, 0]  # d_x-a_x
    ady_ = points[3, 1] - points[0, 1]  # d_y-a_y
    adz_ = points[3, 2] - points[0, 2]  # d_z-a_z
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

    final_sub_dets[0] = A
    final_sub_dets[1] = B
    final_sub_dets[2] = C
    final_sub_dets[3] = D


@njit
def _walk(
        point_id, t_index, vertices_ID, neighbour_ID, points, gv, global_arr,
        exactinit_arr):
    '''
    Walks from the given tet to the tet enclosing the given point.

        point_id : The index (corresponding to the points array) of the
                   point to be inserted into the triangulation.
         t_index : The index of the tet to start the walk from.
     vertices_ID : The global array storing all the indices (corresponding
                   to the points array) of the vertices of all the tets.
    neighbour_ID : The global array storing the indices of the neighbouring
                   tets.
          points : The global array storing the co-ordinates of all the
                   points to be triangulated.
              gv : Index assigned to the ghost vertex.
    '''

    gv_idx = 4
    if vertices_ID[t_index, 0] == gv:
        gv_idx = 0
    elif vertices_ID[t_index, 1] == gv:
        gv_idx = 1
    elif vertices_ID[t_index, 2] == gv:
        gv_idx = 2
    elif vertices_ID[t_index, 3] == gv:
        gv_idx = 3

    if gv_idx != 4:
        # t_index is a ghost tet, in this case simply step into the adjacent
        # real tet.
        t_index = neighbour_ID[t_index, gv_idx]//4

    point_x = points[point_id, 0]
    point_y = points[point_id, 1]
    point_z = points[point_id, 2]

    while True:
        # i.e. t_index is a real tetrahedron
        t_op_index_in_t = 5

        a = vertices_ID[t_index, 0]
        b = vertices_ID[t_index, 1]
        c = vertices_ID[t_index, 2]
        d = vertices_ID[t_index, 3]

        a_x = points[a, 0]
        a_y = points[a, 1]
        a_z = points[a, 2]
        b_x = points[b, 0]
        b_y = points[b, 1]
        b_z = points[b, 2]
        c_x = points[c, 0]
        c_y = points[c, 1]
        c_z = points[c, 2]
        d_x = points[d, 0]
        d_y = points[d, 1]
        d_z = points[d, 2]

        det = orient3d(
            point_x, point_y, point_z, b_x, b_y, b_z, c_x, c_y, c_z, d_x, d_y,
            d_z, global_arr, exactinit_arr)
        if det < 0:
            t_op_index_in_t = 0
        else:
            det = orient3d(
                a_x, a_y, a_z, point_x, point_y, point_z, c_x, c_y, c_z, d_x,
                d_y, d_z, global_arr, exactinit_arr)
            if det < 0:
                t_op_index_in_t = 1
            else:
                det = orient3d(
                    a_x, a_y, a_z, b_x, b_y, b_z, point_x, point_y, point_z,
                    d_x, d_y, d_z, global_arr, exactinit_arr)
                if det < 0:
                    t_op_index_in_t = 2
                else:
                    det = orient3d(
                        a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z, point_x,
                        point_y, point_z, global_arr, exactinit_arr)
                    if det < 0:
                        t_op_index_in_t = 3

        if t_op_index_in_t != 5:
            t_index = neighbour_ID[t_index, t_op_index_in_t]//4
        else:
            # point_id lies inside t_index
            break

        if vertices_ID[t_index, 0] == gv:
            break
        elif vertices_ID[t_index, 1] == gv:
            break
        elif vertices_ID[t_index, 2] == gv:
            break
        elif vertices_ID[t_index, 3] == gv:
            break

    return t_index


@njit
def _cavity_helper(
        point_id, t_index, points, vertices_ID, sub_determinants, gv,
        global_arr, exactinit_arr):
    '''
    Checks whether the given point lies inside the circumsphere the given tet.
    Returns True if it does.

            point_id : The index (corresponding to the points array) of the
                       point to be inserted into the triangulation.
             t_index : The index of the tet to check.
              points : The global array storing the co-ordinates of all the
                       points to be triangulated.
         vertices_ID : The global array storing all the indices (corresponding
                       to the points array) of the vertices of all the tets.
    sub_determinants : The global array storing the sub-determinants of all the
                       tets.
                  gv : Index assigned to the ghost vertex.
    '''

    gv_idx = 4
    if vertices_ID[t_index, 0] == gv:
        gv_idx = 0
    elif vertices_ID[t_index, 1] == gv:
        gv_idx = 1
    elif vertices_ID[t_index, 2] == gv:
        gv_idx = 2
    elif vertices_ID[t_index, 3] == gv:
        gv_idx = 3

    point_x = points[point_id, 0]
    point_y = points[point_id, 1]
    point_z = points[point_id, 2]

    if gv_idx != 4:
        # i.e. t_index is a ghost triangle
        if gv_idx == 0:
            b_x = points[vertices_ID[t_index, 1], 0]
            b_y = points[vertices_ID[t_index, 1], 1]
            b_z = points[vertices_ID[t_index, 1], 2]
            c_x = points[vertices_ID[t_index, 2], 0]
            c_y = points[vertices_ID[t_index, 2], 1]
            c_z = points[vertices_ID[t_index, 2], 2]
            d_x = points[vertices_ID[t_index, 3], 0]
            d_y = points[vertices_ID[t_index, 3], 1]
            d_z = points[vertices_ID[t_index, 3], 2]
        elif gv_idx == 1:
            b_x = points[vertices_ID[t_index, 2], 0]
            b_y = points[vertices_ID[t_index, 2], 1]
            b_z = points[vertices_ID[t_index, 2], 2]
            c_x = points[vertices_ID[t_index, 0], 0]
            c_y = points[vertices_ID[t_index, 0], 1]
            c_z = points[vertices_ID[t_index, 0], 2]
            d_x = points[vertices_ID[t_index, 3], 0]
            d_y = points[vertices_ID[t_index, 3], 1]
            d_z = points[vertices_ID[t_index, 3], 2]
        elif gv_idx == 2:
            b_x = points[vertices_ID[t_index, 0], 0]
            b_y = points[vertices_ID[t_index, 0], 1]
            b_z = points[vertices_ID[t_index, 0], 2]
            c_x = points[vertices_ID[t_index, 1], 0]
            c_y = points[vertices_ID[t_index, 1], 1]
            c_z = points[vertices_ID[t_index, 1], 2]
            d_x = points[vertices_ID[t_index, 3], 0]
            d_y = points[vertices_ID[t_index, 3], 1]
            d_z = points[vertices_ID[t_index, 3], 2]
        elif gv_idx == 3:
            b_x = points[vertices_ID[t_index, 0], 0]
            b_y = points[vertices_ID[t_index, 0], 1]
            b_z = points[vertices_ID[t_index, 0], 2]
            c_x = points[vertices_ID[t_index, 2], 0]
            c_y = points[vertices_ID[t_index, 2], 1]
            c_z = points[vertices_ID[t_index, 2], 2]
            d_x = points[vertices_ID[t_index, 1], 0]
            d_y = points[vertices_ID[t_index, 1], 1]
            d_z = points[vertices_ID[t_index, 1], 2]

        vol_t = orient3d(
            point_x, point_y, point_z, b_x, b_y, b_z, c_x, c_y, c_z, d_x, d_y,
            d_z, global_arr, exactinit_arr)

        if vol_t > 0:
            return True
        elif vol_t == 0:
            adx_ = point_x - d_x
            ady_ = point_y - d_y
            adz_ = point_z - d_z

            bdx_ = b_x - d_x
            bdy_ = b_y - d_y
            bdz_ = b_z - d_z

            cdx_ = c_x - d_x
            cdy_ = c_y - d_y
            cdz_ = c_z - d_z

            cbx_ = c_x - b_x
            cby_ = c_y - b_y
            cbz_ = c_z - b_z

            # t0x, t0y, t0z = cross_pdt(cbx_, cby_, cbz_, -bdx_, -bdy_, -bdz_)
            t0x, t0y, t0z = -cby_*bdz_+cbz_*bdy_, cbx_*bdz_-cbz_*bdx_, -cbx_*bdy_+cby_*bdx_
            # t1x, t1y, t1z = cross_pdt(t0x, t0y, t0z, cbx_, cby_, cbz_)
            t1x, t1y, t1z = t0y*cbz_-t0z*cby_, -t0x*cbz_+t0z*cbx_, t0x*cby_-t0y*cbx_
            # t2x, t2y, t2z = cross_pdt(-t0x, -t0y, -t0z, -bdx_, -bdy_, -bdz_)
            t2x, t2y, t2z = t0y*bdz_-t0z*bdy_, -t0x*bdz_+t0z*bdx_, t0x*bdy_-t0y*bdx_

            normsq_db = bdx_**2 + bdy_**2 + bdz_**2
            normsq_cb = cbx_**2 + cby_**2 + cbz_**2
            normsq_t0 = t0x**2 + t0y**2 + t0z**2

            center_x = b_x + 0.5*(normsq_db*t1x + normsq_cb*t2x)/normsq_t0
            center_y = b_y + 0.5*(normsq_db*t1y + normsq_cb*t2y)/normsq_t0
            center_z = b_z + 0.5*(normsq_db*t1z + normsq_cb*t2z)/normsq_t0

            r_sq = (center_x-b_x)**2+(center_y-b_y)**2+(center_z-b_z)**2

            dist = (point_x-center_x)**2 + \
                   (point_y-center_y)**2 + \
                   (point_z-center_z)**2
            if dist <= r_sq:
                return True
            else:
                return False
        else:
            return False
    else:
        a_x = points[vertices_ID[t_index, 0], 0]
        a_y = points[vertices_ID[t_index, 0], 1]
        a_z = points[vertices_ID[t_index, 0], 2]

        det = (point_x - a_x)*sub_determinants[t_index, 0] + \
              (point_y - a_y)*sub_determinants[t_index, 1] + \
              (point_z - a_z)*sub_determinants[t_index, 2] + \
              (
                  (point_x-a_x)**2 + (point_y-a_y)**2 + (point_z-a_z)**2
              )*sub_determinants[t_index, 3]

        static_filter_i3d = exactinit_arr[9]

        if np.abs(det) <= static_filter_i3d:
            b_x = points[vertices_ID[t_index, 1], 0]
            b_y = points[vertices_ID[t_index, 1], 1]
            b_z = points[vertices_ID[t_index, 1], 2]
            c_x = points[vertices_ID[t_index, 2], 0]
            c_y = points[vertices_ID[t_index, 2], 1]
            c_z = points[vertices_ID[t_index, 2], 2]
            d_x = points[vertices_ID[t_index, 3], 0]
            d_y = points[vertices_ID[t_index, 3], 1]
            d_z = points[vertices_ID[t_index, 3], 2]
            det = insphere(
                a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z, d_x, d_y, d_z,
                point_x, point_y, point_z, global_arr, exactinit_arr)


        if det >= 0:
            return True
        else:
            return False


@njit
def _identify_cavity(
        points, point_id, t_index, neighbour_ID, vertices_ID, sub_determinants,
        bad_tet, boundary_tet, boundary_vtx, boundary_index, gv, global_arr,
        exactinit_arr, array_sizes):
    '''
    Identifies all the 'bad' triangles, i.e. the triangles whose circumcircles
    enclose the given point. Returns a list of the indices of the bad triangles
    and a list of the triangles bordering the cavity.

              points : The global array containing the co-ordinates of all the
                       points to be triangulated.
            point_id : The index (corresponding to the points array) of the
                       point to be inserted into the triangulation.
             t_index : The index of the tet enclosing point_id.
        neighbour_ID : The global array containing the indices of the
                       neighbours of all the tets.
         vertices_ID : The global array containing the indices (corresponding
                       to the points array) of the vertices of all the tets.
    sub_determinants : The global array containing the sub-determinants of all
                       the tets.
             bad_tet : Helper array, used to store the indices of the 'bad'
                       tets, i.e. those whose circumspheres containt point_id.
        boundary_tet : Helper array, used to store the tets on the boundary of
                       the cavity.
        boundary_vtx : Helper array, used to store the points on the boundary
                       of the cavity.
      boundary_index : Helper array, assigns a unique secondary index to all
                       the points on the boundary of the caivty. This is used
                       later on to reconstruct adjacencies.
                  gv : Index assigned to the ghost vertex.
    '''

    bt_len = array_sizes[0]
    bt_end = 0

    boundary_len = array_sizes[1]
    boundary_end = 0

    # Adding the first bad triangle, i.e. the enclosing triangle
    bad_tet[bt_end] = t_index
    sub_determinants[t_index, 3] = 10
    bt_end += 1

    boundary_vtx_counter = 0
    bt_iter = 0
    while True:
        t_index = bad_tet[bt_iter]

        for j in range(4):
            jth_nbr_idx = neighbour_ID[t_index, j]//4

            if sub_determinants[jth_nbr_idx, 3] <= 0:
                # jth_nbr_idx has not been stored in the bad_tet array yet.
                inside_tet_flag = _cavity_helper(
                    point_id, jth_nbr_idx, points, vertices_ID,
                    sub_determinants, gv, global_arr, exactinit_arr)
                if inside_tet_flag == True:
                    # i.e. the j'th neighbour is a bad tet.
                    if bt_end >= bt_len:
                        temp_arr1 = np.empty(2*bt_len, dtype=np.int64)
                        for l in range(bt_end):
                            temp_arr1[l] = bad_tet[l]
                        bt_len = 2*bt_len
                        bad_tet = temp_arr1

                    bad_tet[bt_end] = jth_nbr_idx
                    bt_end += 1

                    sub_determinants[jth_nbr_idx, 3] = 10
                else:
                    # i.e. the j'th neighbour is a boundary tet.
                    if boundary_end >= boundary_len:
                        temp_arr2 = np.empty(2*boundary_len, dtype=np.int64)
                        temp_arr3 = np.empty(
                            shape=(2*boundary_len, 3), dtype=np.int64)
                        for l in range(boundary_end):
                            temp_arr2[l] = boundary_tet[l]
                            temp_arr3[l, 0] = boundary_vtx[l, 0]
                            temp_arr3[l, 1] = boundary_vtx[l, 1]
                            temp_arr3[l, 2] = boundary_vtx[l, 2]
                        boundary_len = 2*boundary_len
                        boundary_tet = temp_arr2
                        boundary_vtx = temp_arr3

                    boundary_tet[boundary_end] = neighbour_ID[t_index, j]

                    # Storing the vertices of t_index that lie on the boundary

                    if j % 2 == 0:
                        boundary_vtx[boundary_end, 0] = vertices_ID[t_index, (j+1) % 4]
                        boundary_vtx[boundary_end, 1] = vertices_ID[t_index, (j+2) % 4]
                        boundary_vtx[boundary_end, 2] = vertices_ID[t_index, (j+3) % 4]
                    else:
                        boundary_vtx[boundary_end, 0] = vertices_ID[t_index, (j+3) % 4]
                        boundary_vtx[boundary_end, 1] = vertices_ID[t_index, (j+2) % 4]
                        boundary_vtx[boundary_end, 2] = vertices_ID[t_index, (j+1) % 4]

                    if boundary_index[boundary_vtx[boundary_end, 0]] == -1:
                        boundary_index[boundary_vtx[boundary_end, 0]] = boundary_vtx_counter
                        boundary_vtx_counter += 1
                    if boundary_index[boundary_vtx[boundary_end, 1]] == -1:
                        boundary_index[boundary_vtx[boundary_end, 1]] = boundary_vtx_counter
                        boundary_vtx_counter += 1
                    if boundary_index[boundary_vtx[boundary_end, 2]] == -1:
                        boundary_index[boundary_vtx[boundary_end, 2]] = boundary_vtx_counter
                        boundary_vtx_counter += 1

                    boundary_end += 1

        bt_iter += 1
        if bt_iter == bt_end:
            break

    array_sizes[0] = bt_len
    array_sizes[1] = boundary_len
    array_sizes[2] = bt_end
    array_sizes[3] = boundary_end
    array_sizes[4] = boundary_vtx_counter

    return bad_tet, boundary_tet, boundary_vtx


@njit
def _make_Delaunay_ball(
        point_id, bad_tets, boundary_tets, boundary_vtx, points, csd_final,
        csd_points, neighbour_ID, vertices_ID, sub_determinants,
        boundary_index, adjacency_array, gv, array_sizes, num_entities,
        available_tet, q_params):
    '''
    Joins all the vertices on the boundary to the new point, and forms
    the corresponding triangles along with their adjacencies. Returns the index
    of a new triangle, to be used as the starting point of the next walk.

         point_id : The index corresponding to the points array of the point to
                    be inserted into the triangulation.
         bad_tets : The list of tets whose circumcircle contains point_id.
    boundary_tets : The list of triangles lying on the boundary of the cavity
                    formed by the bad triangles.
     boundary_vtx : The vertices lying on the boundary of the cavity formed by
                    all the bad triangles.
           points : The global array storing the co-ordinates of all the points
                    to be triangulated.
    '''

    num_tet = num_entities[1]
    mesh_cap = num_entities[2]

    bt_end = array_sizes[2]
    boundary_end = array_sizes[3]
    boundary_vtx_counter = array_sizes[4]

    q_head = q_params[0]
    q_tail = q_params[1]
    q_cap = q_params[2]
    q_num_items = q_params[3]

    # populating the cavity with new tets
    for i in range(boundary_end):
        if i < bt_end:
            t_index = bad_tets[i]
        else:
            if q_num_items > 0:
                # print("assigning from available_tet")
                t_index = available_tet[q_head]
                # print("assigned from available_tet")
                q_head += 1
                if q_head >= q_cap:
                    q_head -= q_cap
                q_num_items -= 1
            else:
                if num_tet > mesh_cap:
                    temp1 = np.empty(shape=(2*mesh_cap, 4), dtype=np.int64)
                    for l in range(mesh_cap):
                        temp1[l, 0] = neighbour_ID[l, 0]
                        temp1[l, 1] = neighbour_ID[l, 1]
                        temp1[l, 2] = neighbour_ID[l, 2]
                        temp1[l, 3] = neighbour_ID[l, 3]
                    neighbour_ID = temp1

                    temp2 = np.empty(shape=(2*mesh_cap, 4), dtype=np.int64)
                    for l in range(mesh_cap):
                        temp2[l, 0] = vertices_ID[l, 0]
                        temp2[l, 1] = vertices_ID[l, 1]
                        temp2[l, 2] = vertices_ID[l, 2]
                        temp2[l, 3] = vertices_ID[l, 3]
                    vertices_ID = temp2

                    temp3 = np.empty(shape=(2*mesh_cap, 4), dtype=np.float64)
                    for l in range(mesh_cap):
                        temp3[l, 0] = sub_determinants[l, 0]
                        temp3[l, 1] = sub_determinants[l, 1]
                        temp3[l, 2] = sub_determinants[l, 2]
                        temp3[l, 3] = sub_determinants[l, 3]
                    sub_determinants = temp3
                    mesh_cap *= 2

                t_index = num_tet
                num_tet += 1

        vertices_ID[t_index, 0] = point_id
        vertices_ID[t_index, 1] = boundary_vtx[i, 0]
        vertices_ID[t_index, 2] = boundary_vtx[i, 1]
        vertices_ID[t_index, 3] = boundary_vtx[i, 2]

        b_tet = boundary_tets[i]
        neighbour_ID[t_index, 0] = b_tet
        neighbour_ID[b_tet//4, b_tet % 4] = 4*t_index

        is_ghost_tet = False
        if vertices_ID[t_index, 1] == gv:
            is_ghost_tet = True
        elif vertices_ID[t_index, 2] == gv:
            is_ghost_tet = True
        elif vertices_ID[t_index, 3] == gv:
            is_ghost_tet = True

        if is_ghost_tet is False:
            for j in range(4):
                csd_points[j, 0] = points[vertices_ID[t_index, j], 0]
                csd_points[j, 1] = points[vertices_ID[t_index, j], 1]
                csd_points[j, 2] = points[vertices_ID[t_index, j], 2]
            _calculate_sub_dets(csd_points, csd_final)
            sub_determinants[t_index, 0] = csd_final[0]
            sub_determinants[t_index, 1] = csd_final[1]
            sub_determinants[t_index, 2] = csd_final[2]
            sub_determinants[t_index, 3] = csd_final[3]
        else:
            sub_determinants[t_index, 0] = 0
            sub_determinants[t_index, 1] = 0
            sub_determinants[t_index, 2] = 0
            sub_determinants[t_index, 3] = 0

    # computing internal adjacencies
    q_head = q_params[0]
    q_num_items = q_params[3]
    for i in range(boundary_end):
        if i < bt_end:
            t = bad_tets[i]
        else:
            if q_num_items > 0:
                t = available_tet[q_head]
                q_head += 1
                if q_head >= q_cap:
                    q_head -= q_cap
                q_num_items -= 1
            else:
                t = num_tet - (boundary_end - i)
        i1 = boundary_index[vertices_ID[t, 1]]
        i2 = boundary_index[vertices_ID[t, 2]]
        i3 = boundary_index[vertices_ID[t, 3]]
        adjacency_array[boundary_vtx_counter*i1+i2] = 4*t+3
        adjacency_array[boundary_vtx_counter*i2+i3] = 4*t+1
        adjacency_array[boundary_vtx_counter*i3+i1] = 4*t+2

    q_head = q_params[0]
    q_cap = q_params[2]
    q_num_items = q_params[3]
    for i in range(boundary_end):
        if i < bt_end:
            t = bad_tets[i]
        else:
            if q_num_items > 0:
                t = available_tet[q_head]
                q_head += 1
                if q_head >= q_cap:
                    q_head -= q_cap
                q_num_items -= 1
            else:
                t = num_tet - (boundary_end - i)
        i1 = boundary_index[vertices_ID[t, 1]]
        i2 = boundary_index[vertices_ID[t, 2]]
        i3 = boundary_index[vertices_ID[t, 3]]
        neighbour_ID[t, 1] = adjacency_array[boundary_vtx_counter*i3+i2]
        neighbour_ID[t, 2] = adjacency_array[boundary_vtx_counter*i1+i3]
        neighbour_ID[t, 3] = adjacency_array[boundary_vtx_counter*i2+i1]

    q_head = q_params[0]
    q_num_items = q_params[3]
    for i in range(boundary_end):
        if i < bt_end:
            t = bad_tets[i]
        else:
            if q_num_items > 0:
                t = available_tet[q_head]
                q_head += 1
                if q_head >= q_cap:
                    q_head -= q_cap
                q_num_items -= 1
            else:
                t = num_tet - (boundary_end - i)
        boundary_index[vertices_ID[t, 1]] = -1
        boundary_index[vertices_ID[t, 2]] = -1
        boundary_index[vertices_ID[t, 3]] = -1

    q_params[0] = q_head
    q_params[3] = q_num_items

    num_entities[1] = num_tet
    num_entities[2] = mesh_cap

    return neighbour_ID, vertices_ID, sub_determinants


@njit
def assembly(
        csd_final, csd_points, bad_tet, boundary_tet, boundary_vtx, points,
        vertices_ID, neighbour_ID, sub_determinants, boundary_index,
        adjacency_array, gv, global_arr, exactinit_arr, array_sizes,
        num_entities, available_tet, q_params):

    path_cases_counter = 0
    exactinit3d(points, exactinit_arr)

    initialize(
        points, vertices_ID, gv, neighbour_ID, csd_final, csd_points,
        sub_determinants, global_arr, exactinit_arr)
    # print("initialized")

    num_points = num_entities[0]
    num_tet = 5
    num_entities[1] = num_tet

    old_tet = 0
    for point_id in np.arange(4, num_points):
        # print("point : ")
        # print(point_id)

        enclosing_tet = _walk(
            point_id, old_tet, vertices_ID, neighbour_ID, points, gv,
            global_arr, exactinit_arr)
        # print("walk done")

        cavity_tuple = _identify_cavity(
            points, point_id, enclosing_tet, neighbour_ID, vertices_ID,
            sub_determinants, bad_tet, boundary_tet, boundary_vtx,
            boundary_index, gv, global_arr, exactinit_arr, array_sizes)
        # print("cavity done")

        bad_tet = cavity_tuple[0]
        boundary_tet = cavity_tuple[1]
        boundary_vtx = cavity_tuple[2]

        bt_end = array_sizes[2]
        boundary_end = array_sizes[3]
        boundary_vtx_counter = array_sizes[4]

        if len(adjacency_array) < boundary_vtx_counter*boundary_vtx_counter:
            adjacency_array = np.zeros(
                shape=boundary_vtx_counter**2, dtype=np.int64)
        # print("adjacency_array adjusted if it was needed")

        # print(boundary_end)
        # print(bt_end)
        # print(q_params[3])
        mDb_tuple = _make_Delaunay_ball(
            point_id, bad_tet, boundary_tet, boundary_vtx, points, csd_final,
            csd_points, neighbour_ID, vertices_ID, sub_determinants,
            boundary_index, adjacency_array, gv, array_sizes, num_entities,
            available_tet, q_params)
        # print("mDb done")

        neighbour_ID = mDb_tuple[0]
        vertices_ID = mDb_tuple[1]
        sub_determinants = mDb_tuple[2]

        if bt_end < boundary_end:
            old_tet = bad_tet[bt_end - 1]
        else:
            old_tet = bad_tet[boundary_end-1]

        if boundary_end < bt_end:
            # print("path case")
            path_cases_counter += 1

            ntet_to_add = bt_end - boundary_end
            q_head = q_params[0]
            q_tail = q_params[1]
            q_cap = q_params[2]
            q_num_items = q_params[3]

            if ntet_to_add + q_num_items >= q_cap:
                temp_arr4 = np.empty(
                    shape=2*(ntet_to_add + q_num_items), dtype=np.int64)
                for k in range(q_cap):
                    temp_arr4[k] = available_tet[q_head]
                    q_head += 1
                    if q_head >= q_cap:
                        q_head -= q_cap
                available_tet = temp_arr4
                q_params[0] = 0
                q_params[1] = q_num_items
                q_params[2] = 2*(ntet_to_add + q_num_items)
                q_head = q_params[0]
                q_tail = q_params[1]
                q_cap = q_params[2]

            for k in range(boundary_end, bt_end):
                available_tet[q_tail] = bad_tet[k]
                q_tail += 1
                if q_tail >= q_cap:
                    q_tail -= q_cap
                q_num_items += 1

            q_params[1] = q_tail
            q_params[3] = q_num_items


    if q_params[3] > 0:
        q_head = q_params[0]
        q_cap = q_params[2]
        q_num_items = q_params[3]

        num_tet = num_entities[1]

        for k in range(q_params[3]):
            tet = available_tet[q_head]
            q_head += 1
            if q_head >= q_cap:
                q_head -= q_cap
            q_num_items -= 1

            for t in range(tet, num_tet-1):
                vertices_ID[t, 0] = vertices_ID[t+1, 0]
                vertices_ID[t, 1] = vertices_ID[t+1, 1]
                vertices_ID[t, 2] = vertices_ID[t+1, 2]
                vertices_ID[t, 3] = vertices_ID[t+1, 3]

                neighbour_ID[t, 0] = neighbour_ID[t+1, 0]
                neighbour_ID[t, 1] = neighbour_ID[t+1, 1]
                neighbour_ID[t, 2] = neighbour_ID[t+1, 2]
                neighbour_ID[t, 3] = neighbour_ID[t+1, 3]

                sub_determinants[t, 0] = sub_determinants[t+1, 0]
                sub_determinants[t, 1] = sub_determinants[t+1, 1]
                sub_determinants[t, 2] = sub_determinants[t+1, 2]
                sub_determinants[t, 3] = sub_determinants[t+1, 3]

            num_tet -= 1

            for i in range(num_tet):
                for j in range(4):
                    temp = neighbour_ID[i, j]
                    if temp//4 > tet:
                        neighbour_ID[i, j] = 4*(temp//4-1) + temp % 4

            for i in range(q_head, q_head + q_num_items):
                if i >= q_cap:
                    i -= q_cap
                if available_tet[i] > tet:
                    available_tet[i] -= 1

        num_entities[1] = num_tet
        q_params[0] = q_head
        q_params[3] = q_num_items

    return vertices_ID, neighbour_ID, sub_determinants, path_cases_counter


@njit
def TestNeighbours_njit(neighbour_ID, truth_array, truth_array_2):

    num_tet = int(len(neighbour_ID)*0.25)

    for t in range(num_tet):
        for j in range(4):
            neighbour = neighbour_ID[4*t+j]//4
            op_idx = neighbour_ID[4*t+j] % 4
            temp = neighbour_ID[4*neighbour+op_idx]
            if temp != 4*t+j:
                truth_array[4*t+j] = 0
                if j == 0:
                    truth_array_2[t] = 0


@njit
def TestNeighbours(neighbour_ID):

    num_tet = int(len(neighbour_ID)*0.25)
    truth_array = np.ones(shape=4*num_tet, dtype=np.bool_)
    truth_array_2 = np.ones(shape=num_tet, dtype=np.bool_)

    TestNeighbours_njit(neighbour_ID, truth_array, truth_array_2)

    num_prob_tets = 0
    for i in range(num_tet):
        if not truth_array[4*i+0]:
            num_prob_tets += 1
        elif not truth_array[4*i+1]:
            num_prob_tets += 1
        elif not truth_array[4*i+2]:
            num_prob_tets += 1
        elif not truth_array[4*i+3]:
            num_prob_tets += 1

    if num_prob_tets > 0:
        flag1 = False
    else:
        flag1 = True

    num_prob_tets2 = len(np.where(truth_array_2 == np.bool_(False))[0])
    # for i in range(num_tet):
    #     if truth_array_2[i] == False:
    #         num_prob_tets2 += 1

    flag2 = False
    if num_prob_tets2 > 0:
        flag2 = True

    return num_prob_tets, flag2, truth_array, truth_array_2


def plot_tets(vertices_ID, num_tet, gv, points):

    import pyvista as pv
    import vtk

    indices = np.zeros(shape=len(vertices_ID), dtype=np.bool_)
    counter = 0
    for i in np.arange(0, num_tet):
        idx = np.where(vertices_ID[4*i:4*i+4] == gv)[0]
        if len(idx) == 0:
            # i.e. i is not a ghost triangle
            indices[4*i+0] = 1
            indices[4*i+1] = 1
            indices[4*i+2] = 1
            indices[4*i+3] = 1
            counter += 1
    vertices = vertices_ID[indices]
    print("vertices : \n" + str(vertices.reshape((counter, 4))))
    print("counter : " + str(counter))

    cells = 4*np.ones(shape=5*counter, dtype=np.int64)
    for i in range(4):
        cells[i+1::5] = vertices[i::4]

    points = points.reshape(
        (
            int(len(points)/3),
            3
        )
    )

    # print("points : \n" + str(points))

    offset = np.arange(0, 5*counter, 5)

    cell_type = np.array([vtk.VTK_TETRA for i in range(counter)])

    grid = pv.UnstructuredGrid(offset, cells, cell_type, points)

    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True)

    plotter.add_point_labels(
        points,
        np.arange(len(points)),
        point_size=20,
        font_size=36
    )

    plotter.show()


@njit
def initialize(
        points, vertices_ID, gv, neighbour_ID, csd_final, csd_points,
        sub_determinants, global_arr, exactinit_arr):

    idx = 3
    a_x = points[0, 0]
    a_y = points[0, 1]
    a_z = points[0, 2]
    b_x = points[1, 0]
    b_y = points[1, 1]
    b_z = points[1, 2]
    c_x = points[2, 0]
    c_y = points[2, 1]
    c_z = points[2, 2]
    while True:
        d_x = points[idx, 0]
        d_y = points[idx, 1]
        d_z = points[idx, 2]
        vol_t = orient3d(
            a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z, d_x, d_y,
            d_z, global_arr, exactinit_arr)
        if vol_t > 0:
            for i in range(3):
                points[3, i], points[idx, i] = points[idx, i], points[3, i]
            break
        elif vol_t < 0:
            for i in range(3):
                points[3, i], points[idx, i] = points[idx, i], points[3, i]
                points[0, i], points[1, i] = points[1, i], points[0, i]
            break
        else:
            idx += 1

    vertices_ID[0, 0] = 0
    vertices_ID[0, 1] = 1
    vertices_ID[0, 2] = 2
    vertices_ID[0, 3] = 3

    idx_0 = vertices_ID[0, 0]
    idx_1 = vertices_ID[0, 1]
    idx_2 = vertices_ID[0, 2]
    idx_3 = vertices_ID[0, 3]

    vertices_ID[1, 0] = gv
    vertices_ID[1, 1] = idx_0
    vertices_ID[1, 2] = idx_1
    vertices_ID[1, 3] = idx_2

    vertices_ID[2, 0] = gv
    vertices_ID[2, 1] = idx_0
    vertices_ID[2, 2] = idx_3
    vertices_ID[2, 3] = idx_1

    vertices_ID[3, 0] = gv
    vertices_ID[3, 1] = idx_3
    vertices_ID[3, 2] = idx_2
    vertices_ID[3, 3] = idx_1

    vertices_ID[4, 0] = gv
    vertices_ID[4, 1] = idx_0
    vertices_ID[4, 2] = idx_2
    vertices_ID[4, 3] = idx_3

    neighbour_ID[0, 0] = 4*3+0
    neighbour_ID[0, 1] = 4*4+0
    neighbour_ID[0, 2] = 4*2+0
    neighbour_ID[0, 3] = 4*1+0

    neighbour_ID[1, 0] = 4*0+3
    neighbour_ID[1, 1] = 4*3+1
    neighbour_ID[1, 2] = 4*4+3
    neighbour_ID[1, 3] = 4*2+2

    neighbour_ID[2, 0] = 4*0+2
    neighbour_ID[2, 1] = 4*3+2
    neighbour_ID[2, 2] = 4*1+3
    neighbour_ID[2, 3] = 4*4+2

    neighbour_ID[3, 0] = 4*0+0
    neighbour_ID[3, 1] = 4*1+1
    neighbour_ID[3, 2] = 4*2+1
    neighbour_ID[3, 3] = 4*4+1

    neighbour_ID[4, 0] = 4*0+1
    neighbour_ID[4, 1] = 4*3+3
    neighbour_ID[4, 2] = 4*2+3
    neighbour_ID[4, 3] = 4*1+2

    for i in range(4):
        for j in range(3):
            csd_points[i, j] = points[vertices_ID[0, i], j]
    _calculate_sub_dets(csd_points, csd_final)

    sub_determinants[0, 0] = csd_final[0]
    sub_determinants[0, 1] = csd_final[1]
    sub_determinants[0, 2] = csd_final[2]
    sub_determinants[0, 3] = csd_final[3]

    sub_determinants[1, 0] = 0
    sub_determinants[1, 1] = 0
    sub_determinants[1, 2] = 0
    sub_determinants[1, 3] = 0

    sub_determinants[2, 0] = 0
    sub_determinants[2, 1] = 0
    sub_determinants[2, 2] = 0
    sub_determinants[2, 3] = 0

    sub_determinants[3, 0] = 0
    sub_determinants[3, 1] = 0
    sub_determinants[3, 2] = 0
    sub_determinants[3, 3] = 0

    sub_determinants[4, 0] = 0
    sub_determinants[4, 1] = 0
    sub_determinants[4, 2] = 0
    sub_determinants[4, 3] = 0

    return


class Delaunay3D:

    def __init__(self, points):
        # points: list of points to be triangulated

        self.points = BRIO.make_BRIO(points)
        # self.points = points
        N = len(self.points)
        self.gv = N  # index of the ghost vertex
        self.boundary_index = -1*np.ones(shape=N+1, dtype=np.int64)
        self.vertices_ID = np.empty(shape=(7*N, 4), dtype=np.int64)
        self.neighbour_ID = np.empty(shape=(7*N, 4), dtype=np.int64)
        self.sub_determinants = np.empty(shape=(7*N, 4), dtype=np.float64)

        # Arrays that will be passed into the jit-ed functions
        # so that they don't have to get their hands dirty with
        # object creation.
        csd_final = np.empty(4, dtype=np.float64)
        csd_points = np.empty(shape=(4, 3), dtype=np.float64)
        bad_tet = np.empty(100, dtype=np.int64)
        boundary_tet = np.empty(100, dtype=np.int64)
        boundary_vtx = np.empty(shape=(100, 3), dtype=np.int64)
        adjacency_array = np.zeros(shape=40*40, dtype=np.int64)
        exactinit_arr = np.empty(shape=10, dtype=np.float64)
        global_arr = np.empty(shape=22592, dtype=np.float64)
        array_sizes = np.empty(shape=5, dtype=np.int64)
        available_tet = np.empty(shape=256, dtype=np.int64)
        q_params = np.empty(shape=4, dtype=np.int64)
        num_entities = np.empty(shape=3, dtype=np.int64)

        array_sizes[0] = 100
        array_sizes[1] = 100

        q_params[0] = 0
        q_params[1] = 0
        q_params[2] = 256
        q_params[3] = 0

        num_entities[0] = N
        num_entities[1] = 0
        num_entities[2] = 7*N

        res = assembly(
            csd_final, csd_points, bad_tet, boundary_tet, boundary_vtx,
            self.points, self.vertices_ID, self.neighbour_ID,
            self.sub_determinants, self.boundary_index, adjacency_array,
            self.gv, exactinit_arr, global_arr, array_sizes, num_entities,
            available_tet, q_params)

        self.vertices_ID = res[0]
        self.neighbour_ID = res[1]
        self.sub_determinants = res[2]
        self.num_tet = num_entities[1]
        path_cases_counter = res[3]

        return

    def exportDT(self):
        # Export the present Delaunay triangulation

        points = self.points.copy()
        indices = np.zeros(shape=len(self.vertices_ID), dtype=np.bool_)
        # counter = 0
        for i in np.arange(0, self.num_tet):
            idx = np.where(self.vertices_ID[i, :] == self.gv)[0]
            if len(idx) == 0:
                # i.e. i is not a ghost triangle
                indices[i] = 1
                # counter += 1
        vertices = self.vertices_ID[indices, :]
        sub_determinants = self.sub_determinants[indices, :]
        # neighbors = self.neighbour_ID[indices]//4

        # ghost_tets = np.where(indices[0:4*self.num_tet:4] == False)[0]

        # temp_idx = np.isin(neighbors, ghost_tets)
        # neighbors[temp_idx] = -1

        # for gt in ghost_tets[::-1]:
        #     temp_idx = np.where(neighbors > gt)
        #     neighbors[temp_idx] -= 1

        return points, vertices, sub_determinants  # , neighbors

    def export_VTK(self):

        import pyvista as pv
        import vtk

        indices = np.zeros(shape=len(self.vertices_ID), dtype=np.bool_)
        counter = 0

        for i in np.arange(0, self.num_tet):
            idx = np.where(self.vertices_ID[4*i:4*i+4] == self.gv)[0]
            if len(idx) == 0:
                # i.e. i is not a ghost triangle
                indices[4*i+0] = 1
                indices[4*i+1] = 1
                indices[4*i+2] = 1
                indices[4*i+3] = 1
                counter += 1
        vertices = self.vertices_ID[indices]
        # print("vertices : \n" + str(vertices.reshape((counter, 4))))
        print("counter : " + str(counter))

        cells = 4*np.ones(shape=5*counter, dtype=np.int64)
        for i in range(4):
            cells[i+1::5] = vertices[i::4]

        points = self.points.reshape(
            (
                int(len(self.points)/3),
                3
            )
        )

        # print("points : \n" + str(points[0:-1]))

        offset = np.arange(0, 5*counter, 5)

        cell_type = np.array([vtk.VTK_TETRA for i in range(counter)])

        grid = pv.UnstructuredGrid(offset, cells, cell_type, points)

        plotter = pv.Plotter()
        plotter.add_mesh(grid, show_edges=True)

        plotter.add_point_labels(
            points,
            np.arange(len(points)),
            point_size=20,
            font_size=36
        )

        plotter.show()
        # grid.save("mesh.vtk")


def perf(N):
    import time

    np.random.seed(seed=10)

    print("priming numba")
    temp_pts = np.random.rand(3*10).reshape((10, 3))
    tempDT = Delaunay3D(temp_pts)
    print("numba primed \n")

    del temp_pts
    del tempDT

    num_runs = 5
    times = np.empty(shape=num_runs, dtype=np.float64)
    for i in range(num_runs):
        np.random.seed(seed=i**2)
        points = np.random.rand(3*N).reshape((N, 3))
        start = time.time()
        DT = Delaunay3D(points)
        end = time.time()
        times[i] = end - start
        print("RUN {} : {} s".format(i+1, times[i]))
        del DT
        del points

    return np.min(times)

if __name__ == "__main__":
    import sys
    N = int(sys.argv[1])
    time = perf(N)
    print("   Time taken to make the tet-mesh : {} s".format(time))

    # points = np.array([
    #     [0, 0, 1.0],
    #     [1., 0., 0.],
    #     [0., 1.0, 0.],
    #     [0., 0., 0.],
    #     [10.0, 10.0, 10.0]
    # ])

    # a_x = 0.
    # a_y = 0.
    # a_z = 1.
    # a_x = 0.0
    # a_y = 0.0
    # a_z = 1.0
    # b_x = 1.0
    # b_y = 0.0
    # b_z = 0.0
    # c_x = 0.0
    # c_y = 1.0
    # c_z = 0.0
    # d_x = 0.0
    # d_y = 0.0
    # d_z = 0.0
    # e_x = 10.0
    # e_y = 10.0
    # e_z = 10.0

    # exactinit_arr = np.empty(shape=10, dtype=np.float64)
    # global_arr = np.empty(shape=26600, dtype=np.float64)
    # exactinit3d(points, exactinit_arr)
    # res = orient3d(
    #     a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z, d_x, d_y, d_z,
    #     global_arr, exactinit_arr)
    # print(res)

    # res = insphere(
    #     a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z, d_x, d_y, d_z, e_x, e_y,
    #     e_z, global_arr, exactinit_arr)
    # print(res)
