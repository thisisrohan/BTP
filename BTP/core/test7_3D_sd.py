import numpy as np
import BTP.tools.BRIO_3D as BRIO


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

    abx_ = points[3]-points[0]  # b_x-a_x
    aby_ = points[4]-points[1]  # b_y-a_y
    abz_ = points[5]-points[2]  # b_z-a_z
    acx_ = points[6]-points[0]  # c_x-a_x
    acy_ = points[7]-points[1]  # c_y-a_y
    acz_ = points[8]-points[2]  # c_z-a_z
    adx_ = points[9]-points[0]  # d_x-a_x
    ady_ = points[10]-points[1]  # d_y-a_y
    adz_ = points[11]-points[2]  # d_z-a_z
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
    point_id,
    t_index,
    vertices_ID,
    neighbour_ID,
    points,
    gv,
):
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
    if vertices_ID[4*t_index+0] == gv:
        gv_idx = 0
    elif vertices_ID[4*t_index+1] == gv:
        gv_idx = 1
    elif vertices_ID[4*t_index+2] == gv:
        gv_idx = 2
    elif vertices_ID[4*t_index+3] == gv:
        gv_idx = 3

    if gv_idx != 4:
        # t_index is a ghost tet, in this case simply step into the adjacent
        # real tet.
        t_index = neighbour_ID[4*t_index+gv_idx]//4

    point_x = points[3*point_id+0]
    point_y = points[3*point_id+1]
    point_z = points[3*point_id+2]

    while True:
        # i.e. t_index is a real tetrahedron
        t_op_index_in_t = 5

        a = vertices_ID[4*t_index+0]
        b = vertices_ID[4*t_index+1]
        c = vertices_ID[4*t_index+2]
        d = vertices_ID[4*t_index+3]

        pdx_ = point_x - points[3*d+0]
        pdy_ = point_y - points[3*d+1]
        pdz_ = point_z - points[3*d+2]

        bdx_ = points[3*b+0] - points[3*d+0]
        bdy_ = points[3*b+1] - points[3*d+1]
        bdz_ = points[3*b+2] - points[3*d+2]

        cdx_ = points[3*c+0] - points[3*d+0]
        cdy_ = points[3*c+1] - points[3*d+1]
        cdz_ = points[3*c+2] - points[3*d+2]

        temp = pdx_*(bdy_*cdz_-bdz_*cdy_) - \
            pdy_*(bdx_*cdz_-bdz_*cdx_) + \
            pdz_*(bdx_*cdy_-bdy_*cdx_)
        if temp < 0:
            t_op_index_in_t = 0
        else:
            adx_ = points[3*a+0] - points[3*d+0]
            ady_ = points[3*a+1] - points[3*d+1]
            adz_ = points[3*a+2] - points[3*d+2]

            temp = pdx_*(cdy_*adz_-cdz_*ady_) - \
                pdy_*(cdx_*adz_-cdz_*adx_) + \
                pdz_*(cdx_*ady_-cdy_*adx_)
            if temp < 0:
                t_op_index_in_t = 1
            else:
                temp = pdx_*(ady_*bdz_-adz_*bdy_) - \
                    pdy_*(adx_*bdz_-adz_*bdx_) + \
                    pdz_*(adx_*bdy_-ady_*bdx_)
                if temp < 0:
                    t_op_index_in_t = 2
                else:
                    acx_ = points[3*a+0] - points[3*c+0]
                    acy_ = points[3*a+1] - points[3*c+1]
                    acz_ = points[3*a+2] - points[3*c+2]

                    bcx_ = points[3*b+0] - points[3*c+0]
                    bcy_ = points[3*b+1] - points[3*c+1]
                    bcz_ = points[3*b+2] - points[3*c+2]

                    temp = (point_x-points[3*c+0])*(bcy_*acz_-bcz_*acy_) - \
                           (point_y-points[3*c+1])*(bcx_*acz_-bcz_*acx_) + \
                           (point_z-points[3*c+2])*(bcx_*acy_-bcy_*acx_)
                    if temp < 0:
                        t_op_index_in_t = 3

        if t_op_index_in_t != 5:
            t_index = neighbour_ID[4*t_index+t_op_index_in_t]//4
        else:
            # point_id lies inside t_index
            break

        if vertices_ID[4*t_index+0] == gv:
            break
        elif vertices_ID[4*t_index+1] == gv:
            break
        elif vertices_ID[4*t_index+2] == gv:
            break
        elif vertices_ID[4*t_index+3] == gv:
            break

    return t_index


@njit
def _cavity_helper(
    point_id,
    t_index,
    points,
    vertices_ID,
    sub_determinants,
    gv,
):
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
    if vertices_ID[4*t_index+0] == gv:
        gv_idx = 0
    elif vertices_ID[4*t_index+1] == gv:
        gv_idx = 1
    elif vertices_ID[4*t_index+2] == gv:
        gv_idx = 2
    elif vertices_ID[4*t_index+3] == gv:
        gv_idx = 3

    point_x = points[3*point_id+0]
    point_y = points[3*point_id+1]
    point_z = points[3*point_id+2]

    if gv_idx != 4:
        # i.e. t_index is a ghost triangle
        if gv_idx == 0:
            b_x = points[3*vertices_ID[4*t_index+1]+0]
            b_y = points[3*vertices_ID[4*t_index+1]+1]
            b_z = points[3*vertices_ID[4*t_index+1]+2]
            c_x = points[3*vertices_ID[4*t_index+2]+0]
            c_y = points[3*vertices_ID[4*t_index+2]+1]
            c_z = points[3*vertices_ID[4*t_index+2]+2]
            d_x = points[3*vertices_ID[4*t_index+3]+0]
            d_y = points[3*vertices_ID[4*t_index+3]+1]
            d_z = points[3*vertices_ID[4*t_index+3]+2]
        elif gv_idx == 1:
            b_x = points[3*vertices_ID[4*t_index+2]+0]
            b_y = points[3*vertices_ID[4*t_index+2]+1]
            b_z = points[3*vertices_ID[4*t_index+2]+2]
            c_x = points[3*vertices_ID[4*t_index+0]+0]
            c_y = points[3*vertices_ID[4*t_index+0]+1]
            c_z = points[3*vertices_ID[4*t_index+0]+2]
            d_x = points[3*vertices_ID[4*t_index+3]+0]
            d_y = points[3*vertices_ID[4*t_index+3]+1]
            d_z = points[3*vertices_ID[4*t_index+3]+2]
        elif gv_idx == 2:
            b_x = points[3*vertices_ID[4*t_index+0]+0]
            b_y = points[3*vertices_ID[4*t_index+0]+1]
            b_z = points[3*vertices_ID[4*t_index+0]+2]
            c_x = points[3*vertices_ID[4*t_index+1]+0]
            c_y = points[3*vertices_ID[4*t_index+1]+1]
            c_z = points[3*vertices_ID[4*t_index+1]+2]
            d_x = points[3*vertices_ID[4*t_index+3]+0]
            d_y = points[3*vertices_ID[4*t_index+3]+1]
            d_z = points[3*vertices_ID[4*t_index+3]+2]
        elif gv_idx == 3:
            b_x = points[3*vertices_ID[4*t_index+0]+0]
            b_y = points[3*vertices_ID[4*t_index+0]+1]
            b_z = points[3*vertices_ID[4*t_index+0]+2]
            c_x = points[3*vertices_ID[4*t_index+2]+0]
            c_y = points[3*vertices_ID[4*t_index+2]+1]
            c_z = points[3*vertices_ID[4*t_index+2]+2]
            d_x = points[3*vertices_ID[4*t_index+1]+0]
            d_y = points[3*vertices_ID[4*t_index+1]+1]
            d_z = points[3*vertices_ID[4*t_index+1]+2]

        adx_ = point_x - d_x
        ady_ = point_y - d_y
        adz_ = point_z - d_z

        bdx_ = b_x - d_x
        bdy_ = b_y - d_y
        bdz_ = b_z - d_z

        cdx_ = c_x - d_x
        cdy_ = c_y - d_y
        cdz_ = c_z - d_z

        vol_t = adx_*(bdy_*cdz_-bdz_*cdy_) - \
            ady_*(bdx_*cdz_-bdz_*cdx_) + \
            adz_*(bdx_*cdy_-bdy_*cdx_)

        if vol_t > 0:
            return True
        elif vol_t == 0:
            cbx_ = c_x - b_x
            cby_ = c_y - b_y
            cbz_ = c_z - b_z

            t0x, t0y, t0z = cross_pdt(cbx_, cby_, cbz_, -bdx_, -bdy_, -bdz_)
            t1x, t1y, t1z = cross_pdt(t0x, t0y, t0z, cbx_, cby_, cbz_)
            t2x, t2y, t2z = cross_pdt(-t0x, -t0y, -t0z, -bdx_, -bdy_, -bdz_)

            normsq_db = bdx_**2+bdy_**2+bdz_**2
            normsq_cb = cbx_**2+cby_**2+cbz_**2
            normsq_t0 = t0x**2+t0y**2+t0z**2

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
        a_x = points[3*vertices_ID[4*t_index+0]+0]
        a_y = points[3*vertices_ID[4*t_index+0]+1]
        a_z = points[3*vertices_ID[4*t_index+0]+2]

        temp = (point_x - a_x)*sub_determinants[4*t_index+0] + \
            (point_y - a_y)*sub_determinants[4*t_index+1] + \
            (point_z - a_z)*sub_determinants[4*t_index+2] + \
            (
                (point_x-a_x)**2 + (point_y-a_y)**2 + (point_z-a_z)**2
            )*sub_determinants[4*t_index+3]

        if temp >= 0:
            return True
        else:
            return False


@njit
def _identify_cavity(
    points,
    point_id,
    t_index,
    neighbour_ID,
    vertices_ID,
    sub_determinants,
    ic_bad_tet,
    ic_boundary_tet,
    ic_boundary_vtx,
    boundary_index,
    gv,
):
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
          ic_bad_tet : Helper array, used to store the indices of the 'bad'
                       tets, i.e. those whose circumspheres containt point_id.
     ic_boundary_tet : Helper array, used to store the tets on the boundary of
                       the cavity.
     ic_boundary_vtx : Helper array, used to store the points on the boundary
                       of the cavity.
      boundary_index : Helper array, assigns a unique secondary index to all
                       the points on the boundary of the caivty. This is used
                       later on to reconstruct adjacencies.
                  gv : Index assigned to the ghost vertex.
    '''

    ic_len_bad_tet = len(ic_bad_tet)
    ic_bad_tet_end = 0

    ic_len_boundary_tet = len(ic_boundary_tet)
    ic_boundary_tet_end = 0

    ic_len_boundary_vtx = len(ic_boundary_vtx)
    ic_boundary_vtx_end = 0

    # Adding the first bad triangle, i.e. the enclosing triangle
    ic_bad_tet[ic_bad_tet_end] = t_index
    sub_determinants[4*t_index+3] = 10
    ic_bad_tet_end += 1

    boundary_vtx_counter = 0
    ic_idx = 0
    while True:
        t_index = ic_bad_tet[ic_idx]

        for j in range(4):
            jth_nbr_idx = neighbour_ID[4*t_index+j]//4

            if sub_determinants[4*jth_nbr_idx+3] <= 0:
                # jth_nbr_idx has not been stored in the ic_bad_tet array yet.
                inside_tet_flag = _cavity_helper(
                    point_id,
                    jth_nbr_idx,
                    points,
                    vertices_ID,
                    sub_determinants,
                    gv,
                )
                if inside_tet_flag is True:
                    # i.e. the j'th neighbour is a bad tet.
                    if ic_bad_tet_end >= ic_len_bad_tet:
                        temp_arr1 = np.empty(2*ic_len_bad_tet, dtype=np.int64)
                        for l in range(ic_bad_tet_end):
                            temp_arr1[l] = ic_bad_tet[l]
                        ic_len_bad_tet = 2*ic_len_bad_tet
                        ic_bad_tet = temp_arr1

                    ic_bad_tet[ic_bad_tet_end] = jth_nbr_idx
                    ic_bad_tet_end += 1

                    sub_determinants[4*jth_nbr_idx+3] = 10
                else:
                    # i.e. the j'th neighbour is a boundary tet.
                    if ic_boundary_tet_end >= ic_len_boundary_tet:
                        temp_arr2 = np.empty(
                            2*ic_len_boundary_tet,
                            dtype=np.int64
                        )
                        for l in range(ic_boundary_tet_end):
                            temp_arr2[l] = ic_boundary_tet[l]
                        ic_len_boundary_tet = 2*ic_len_boundary_tet
                        ic_boundary_tet = temp_arr2

                    ic_boundary_tet[ic_boundary_tet_end] = neighbour_ID[
                        4*t_index + j
                    ]
                    ic_boundary_tet_end += 1

                    # Storing the vertices of t_index that lie on the boundary
                    if ic_boundary_vtx_end >= ic_len_boundary_vtx:
                        temp_arr3 = np.empty(
                            2*ic_len_boundary_vtx,
                            dtype=np.int64
                        )
                        for l in range(ic_boundary_vtx_end):
                            temp_arr3[l] = ic_boundary_vtx[l]
                        ic_len_boundary_vtx = 2*ic_len_boundary_vtx
                        ic_boundary_vtx = temp_arr3

                    if j % 2 == 0:
                        ic_boundary_vtx[ic_boundary_vtx_end+0] = vertices_ID[
                            4*t_index + (j+1) % 4
                        ]
                        ic_boundary_vtx[ic_boundary_vtx_end+1] = vertices_ID[
                            4*t_index + (j+2) % 4
                        ]
                        ic_boundary_vtx[ic_boundary_vtx_end+2] = vertices_ID[
                            4*t_index + (j+3) % 4
                        ]
                    else:
                        ic_boundary_vtx[ic_boundary_vtx_end+0] = vertices_ID[
                            4*t_index + (j+3) % 4
                        ]
                        ic_boundary_vtx[ic_boundary_vtx_end+1] = vertices_ID[
                            4*t_index + (j+2) % 4
                        ]
                        ic_boundary_vtx[ic_boundary_vtx_end+2] = vertices_ID[
                            4*t_index + (j+1) % 4
                        ]

                    if boundary_index[
                        ic_boundary_vtx[ic_boundary_vtx_end+0]
                    ] == -1:
                        boundary_index[
                            ic_boundary_vtx[ic_boundary_vtx_end+0]
                        ] = boundary_vtx_counter
                        boundary_vtx_counter += 1
                    if boundary_index[
                        ic_boundary_vtx[ic_boundary_vtx_end+1]
                    ] == -1:
                        boundary_index[
                            ic_boundary_vtx[ic_boundary_vtx_end+1]
                        ] = boundary_vtx_counter
                        boundary_vtx_counter += 1
                    if boundary_index[
                        ic_boundary_vtx[ic_boundary_vtx_end+2]
                    ] == -1:
                        boundary_index[
                            ic_boundary_vtx[ic_boundary_vtx_end+2]
                        ] = boundary_vtx_counter
                        boundary_vtx_counter += 1

                    ic_boundary_vtx_end += 3

        ic_idx += 1
        if ic_idx == ic_bad_tet_end:
            break

    return (ic_bad_tet, ic_bad_tet_end, ic_boundary_tet,
            ic_boundary_tet_end, ic_boundary_vtx, boundary_vtx_counter)


@njit
def _make_Delaunay_ball(
    point_id,
    bad_tets,
    bad_tets_end,
    boundary_tets,
    boundary_tets_end,
    boundary_vtx,
    points,
    csd_final,
    csd_points,
    neighbour_ID,
    vertices_ID,
    sub_determinants,
    num_tet,
    boundary_index,
    boundary_vtx_counter,
    adjacency_array,
    gv,
):
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

    # populating the cavity with new tets
    for i in range(boundary_tets_end):
        if i < bad_tets_end:
            t_index = bad_tets[i]
        else:
            if 4*num_tet >= len(neighbour_ID):
                temp1 = np.empty(shape=2*len(neighbour_ID), dtype=np.int64)
                for l in range(num_tet):
                    temp1[4*l+0] = neighbour_ID[4*l+0]
                    temp1[4*l+1] = neighbour_ID[4*l+1]
                    temp1[4*l+2] = neighbour_ID[4*l+2]
                    temp1[4*l+3] = neighbour_ID[4*l+3]
                neighbour_ID = temp1

                temp2 = np.empty(shape=2*len(vertices_ID), dtype=np.int64)
                for l in range(num_tet):
                    temp2[4*l+0] = vertices_ID[4*l+0]
                    temp2[4*l+1] = vertices_ID[4*l+1]
                    temp2[4*l+2] = vertices_ID[4*l+2]
                    temp2[4*l+3] = vertices_ID[4*l+3]
                vertices_ID = temp2

                temp3 = np.empty(shape=2*len(sub_determinants),
                                 dtype=np.float64)
                for l in range(num_tet):
                    temp3[4*l+0] = sub_determinants[4*l+0]
                    temp3[4*l+1] = sub_determinants[4*l+1]
                    temp3[4*l+2] = sub_determinants[4*l+2]
                    temp3[4*l+3] = sub_determinants[4*l+3]
                sub_determinants = temp3

            t_index = num_tet
            num_tet += 1

        vertices_ID[4*t_index+0] = point_id
        vertices_ID[4*t_index+1] = boundary_vtx[3*i+0]
        vertices_ID[4*t_index+2] = boundary_vtx[3*i+1]
        vertices_ID[4*t_index+3] = boundary_vtx[3*i+2]

        neighbour_ID[4*t_index+0] = boundary_tets[i]
        neighbour_ID[boundary_tets[i]] = 4*t_index+0

        is_ghost_tet = False
        if vertices_ID[4*t_index+1] == gv:
            is_ghost_tet = True
        elif vertices_ID[4*t_index+2] == gv:
            is_ghost_tet = True
        elif vertices_ID[4*t_index+3] == gv:
            is_ghost_tet = True

        if is_ghost_tet is False:
            for j in range(4):
                csd_points[3*j+0] = points[3*vertices_ID[4*t_index+j]+0]
                csd_points[3*j+1] = points[3*vertices_ID[4*t_index+j]+1]
                csd_points[3*j+2] = points[3*vertices_ID[4*t_index+j]+2]
            _calculate_sub_dets(csd_points, csd_final)
            sub_determinants[4*t_index+0] = csd_final[0]
            sub_determinants[4*t_index+1] = csd_final[1]
            sub_determinants[4*t_index+2] = csd_final[2]
            sub_determinants[4*t_index+3] = csd_final[3]
        else:
            sub_determinants[4*t_index+0] = 0
            sub_determinants[4*t_index+1] = 0
            sub_determinants[4*t_index+2] = 0
            sub_determinants[4*t_index+3] = 0

    # computing internal adjacencies
    for i in range(boundary_tets_end):
        if i < bad_tets_end:
            t = bad_tets[i]
        else:
            t = num_tet - (boundary_tets_end-i)
        i1 = boundary_index[vertices_ID[4*t+1]]
        i2 = boundary_index[vertices_ID[4*t+2]]
        i3 = boundary_index[vertices_ID[4*t+3]]
        adjacency_array[boundary_vtx_counter*i1+i2] = 4*t+3
        adjacency_array[boundary_vtx_counter*i2+i3] = 4*t+1
        adjacency_array[boundary_vtx_counter*i3+i1] = 4*t+2

    for i in range(boundary_tets_end):
        if i < bad_tets_end:
            t = bad_tets[i]
        else:
            t = num_tet - (boundary_tets_end-i)
        i1 = boundary_index[vertices_ID[4*t+1]]
        i2 = boundary_index[vertices_ID[4*t+2]]
        i3 = boundary_index[vertices_ID[4*t+3]]
        neighbour_ID[4*t+1] = adjacency_array[boundary_vtx_counter*i3+i2]
        neighbour_ID[4*t+2] = adjacency_array[boundary_vtx_counter*i1+i3]
        neighbour_ID[4*t+3] = adjacency_array[boundary_vtx_counter*i2+i1]

    for i in range(boundary_tets_end):
        if i < bad_tets_end:
            t = bad_tets[i]
        else:
            t = num_tet - (boundary_tets_end-i)
        boundary_index[vertices_ID[4*t+1]] = -1
        boundary_index[vertices_ID[4*t+2]] = -1
        boundary_index[vertices_ID[4*t+3]] = -1

    return (neighbour_ID, vertices_ID, sub_determinants, num_tet)


@njit
def assembly(
    old_tet,
    csd_final,
    csd_points,
    ic_bad_tet,
    ic_boundary_tet,
    ic_boundary_vtx,
    points,
    vertices_ID,
    neighbour_ID,
    sub_determinants,
    num_tet,
    boundary_index,
    adjacency_array,
    gv,
):
    path_cases_counter = 0
    for point_id in np.arange(4, int(len(points)/3)):

        enclosing_tet = _walk(
            point_id,             # point_id
            old_tet,              # t_index
            vertices_ID,          # vertices_ID
            neighbour_ID,         # neighbour_ID
            points,               # points
            gv,                   # ghost vertex index
        )

        cavity_tuple = _identify_cavity(
            points,
            point_id,
            enclosing_tet,
            neighbour_ID,
            vertices_ID,
            sub_determinants,
            ic_bad_tet,
            ic_boundary_tet,
            ic_boundary_vtx,
            boundary_index,
            gv,
        )

        ic_bad_tet = cavity_tuple[0]
        ic_bad_tet_end = cavity_tuple[1]
        ic_boundary_tet = cavity_tuple[2]
        ic_boundary_tet_end = cavity_tuple[3]
        ic_boundary_vtx = cavity_tuple[4]
        boundary_vtx_counter = cavity_tuple[5]

        if len(adjacency_array) < boundary_vtx_counter**2:
            adjacency_array = np.zeros(shape=boundary_vtx_counter**2,
                                       dtype=np.int64)

        mDb_tuple = _make_Delaunay_ball(
            point_id,
            ic_bad_tet,
            ic_bad_tet_end,
            ic_boundary_tet,
            ic_boundary_tet_end,
            ic_boundary_vtx,
            points,
            csd_final,
            csd_points,
            neighbour_ID,
            vertices_ID,
            sub_determinants,
            num_tet,
            boundary_index,
            boundary_vtx_counter,
            adjacency_array,
            gv,
        )

        neighbour_ID = mDb_tuple[0]
        vertices_ID = mDb_tuple[1]
        sub_determinants = mDb_tuple[2]
        num_tet = mDb_tuple[3]

        if ic_bad_tet_end < ic_boundary_tet_end:
            old_tet = num_tet-1
        else:
            old_tet = ic_bad_tet[ic_boundary_tet_end-1]

        if ic_boundary_tet_end < ic_bad_tet_end:
            print("path case")
            path_cases_counter += 1
            for k in range(ic_boundary_tet_end, ic_bad_tet_end):
                tet = ic_bad_tet[k]
                for t in range(tet, num_tet-1):
                    vertices_ID[4*t+0] = vertices_ID[4*(t+1)+0]
                    vertices_ID[4*t+1] = vertices_ID[4*(t+1)+1]
                    vertices_ID[4*t+2] = vertices_ID[4*(t+1)+2]
                    vertices_ID[4*t+3] = vertices_ID[4*(t+1)+3]

                    neighbour_ID[4*t+0] = neighbour_ID[4*(t+1)+0]
                    neighbour_ID[4*t+1] = neighbour_ID[4*(t+1)+1]
                    neighbour_ID[4*t+2] = neighbour_ID[4*(t+1)+2]
                    neighbour_ID[4*t+3] = neighbour_ID[4*(t+1)+3]

                    sub_determinants[4*t+0] = sub_determinants[4*(t+1)+0]
                    sub_determinants[4*t+1] = sub_determinants[4*(t+1)+1]
                    sub_determinants[4*t+2] = sub_determinants[4*(t+1)+2]
                    sub_determinants[4*t+3] = sub_determinants[4*(t+1)+3]

                num_tet -= 1

                for i in range(num_tet):
                    for j in range(4):
                        temp = neighbour_ID[4*i+j]
                        if temp//4 > tet:
                            neighbour_ID[4*i+j] = 4*(temp//4-1) + temp % 4

                for i in range(k+1, ic_bad_tet_end):
                    if ic_bad_tet[i] > tet:
                        ic_bad_tet[i] -= 1

    return (vertices_ID, neighbour_ID, sub_determinants,
            num_tet, path_cases_counter)


@njit
def TestNeighbours_njit(neighbour_ID, truth_array, truth_array_2):

    num_tets = int(len(neighbour_ID)*0.25)

    for t in range(num_tets):
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

    num_tets = int(len(neighbour_ID)*0.25)
    truth_array = np.ones(shape=4*num_tets, dtype=np.bool_)
    truth_array_2 = np.ones(shape=num_tets, dtype=np.bool_)

    TestNeighbours_njit(neighbour_ID, truth_array, truth_array_2)

    num_prob_tets = 0
    for i in range(num_tets):
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
    # for i in range(num_tets):
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
def intialize(
    a, b, c, d,
    points,
    vertices_ID,
    gv,
    neighbour_ID,
    csd_final,
    csd_points,
    sub_determinants,
):

    idx = 3
    while True:
        d[0] = points[3*idx+0]
        d[1] = points[3*idx+1]
        d[2] = points[3*idx+2]
        vol_t = a[0]*(
            b[1]*(c[2]-d[2]) -
            b[2]*(c[1]-d[1]) +
            (c[1]*d[2]-c[2]*d[1])
        ) - \
        a[1]*(
            b[0]*(c[2]-d[2]) -
            b[2]*(c[0]-d[0]) +
            (c[0]*d[2]-c[2]*d[0])
        ) + \
        a[2]*(
            b[0]*(c[1]-d[1]) -
            b[1]*(c[0]-d[0]) +
            (c[0]*d[1]-c[1]*d[0])
        ) - \
        (
            b[0]*(c[1]*d[2]-c[2]*d[1]) -
            b[1]*(c[0]*d[2]-c[2]*d[0]) +
            b[2]*(c[0]*d[1]-c[1]*d[0])
        )
        if vol_t > 0:
            for i in range(3):
                points[3*3+i], points[3*idx+i] = points[3*idx+i], points[3*3+i]
            break
        elif vol_t < 0:
            for i in range(3):
                points[3*3+i], points[3*idx+i] = points[3*idx+i], points[3*3+i]
                points[3*0+i], points[3*1+i] = points[3*1+i], points[3*0+i]
            break
        else:
            idx += 1

    vertices_ID[0] = 0
    vertices_ID[1] = 1
    vertices_ID[2] = 2
    vertices_ID[3] = 3

    idx_0 = vertices_ID[0]
    idx_1 = vertices_ID[1]
    idx_2 = vertices_ID[2]
    idx_3 = vertices_ID[3]

    vertices_ID[4] = gv
    vertices_ID[5] = idx_0
    vertices_ID[6] = idx_1
    vertices_ID[7] = idx_2

    vertices_ID[8] = gv
    vertices_ID[9] = idx_0
    vertices_ID[10] = idx_3
    vertices_ID[11] = idx_1

    vertices_ID[12] = gv
    vertices_ID[13] = idx_3
    vertices_ID[14] = idx_2
    vertices_ID[15] = idx_1

    vertices_ID[16] = gv
    vertices_ID[17] = idx_0
    vertices_ID[18] = idx_2
    vertices_ID[19] = idx_3

    neighbour_ID[0] = 4*3+0
    neighbour_ID[1] = 4*4+0
    neighbour_ID[2] = 4*2+0
    neighbour_ID[3] = 4*1+0

    neighbour_ID[4] = 4*0+3
    neighbour_ID[5] = 4*3+1
    neighbour_ID[6] = 4*4+3
    neighbour_ID[7] = 4*2+2

    neighbour_ID[8] = 4*0+2
    neighbour_ID[9] = 4*3+2
    neighbour_ID[10] = 4*1+3
    neighbour_ID[11] = 4*4+2

    neighbour_ID[12] = 4*0+0
    neighbour_ID[13] = 4*1+1
    neighbour_ID[14] = 4*2+1
    neighbour_ID[15] = 4*4+1

    neighbour_ID[16] = 4*0+1
    neighbour_ID[17] = 4*3+3
    neighbour_ID[18] = 4*2+3
    neighbour_ID[19] = 4*1+2

    for i in range(4):
        for j in range(3):
            csd_points[3*i+j] = points[3*vertices_ID[i]+j]
    _calculate_sub_dets(csd_points, csd_final)

    sub_determinants[0] = csd_final[0]
    sub_determinants[1] = csd_final[1]
    sub_determinants[2] = csd_final[2]
    sub_determinants[3] = csd_final[3]

    sub_determinants[4] = 0
    sub_determinants[5] = 0
    sub_determinants[6] = 0
    sub_determinants[7] = 0

    sub_determinants[8] = 0
    sub_determinants[9] = 0
    sub_determinants[10] = 0
    sub_determinants[11] = 0

    sub_determinants[12] = 0
    sub_determinants[13] = 0
    sub_determinants[14] = 0
    sub_determinants[15] = 0

    sub_determinants[16] = 0
    sub_determinants[17] = 0
    sub_determinants[18] = 0
    sub_determinants[19] = 0

    return


class Delaunay3D:

    def __init__(self, points):
        # points: list of points to be triangulated

        self.points = BRIO.make_BRIO(points)
        # self.points = points
        N = int(len(self.points)/3)
        self.gv = N  # index of the ghost vertex
        self.boundary_index = -1*np.ones(shape=N+1, dtype=np.int64)

        a = self.points[0:3]
        b = self.points[3:6]
        c = self.points[6:9]
        d = np.empty(shape=3)

        self.vertices_ID = np.empty(4*(7*N), dtype=np.int64)
        self.neighbour_ID = np.empty(4*(7*N), dtype=np.int64)
        self.sub_determinants = np.empty(4*(7*N), dtype=np.float64)

        csd_points = np.empty(shape=12, dtype=np.float64)
        csd_final = np.empty(4, dtype=np.float64)

        intialize(a, b, c, d, self.points, self.vertices_ID,
                  self.gv, self.neighbour_ID, csd_final,
                  csd_points, self.sub_determinants)

        self.num_tet = 5

        # self.export_VTK()

    def makeDT(self, printTime=False, returnPathCases=False):
        # makes the Delaunay traingulation of the given point list

        time_taken = None
        old_tet = np.int64(0)

        # Arrays that will be passed into the jit-ed functions
        # so that they don't have to get their hands dirty with
        # object creation.
        csd_final = np.empty(4, dtype=np.float64)
        csd_points = np.empty(12, dtype=np.float64)
        ic_bad_tet = np.empty(100, dtype=np.int64)
        ic_boundary_tet = np.empty(100, dtype=np.int64)
        ic_boundary_vtx = np.empty(3*100, dtype=np.int64)
        adjacency_array = np.zeros(shape=40*40, dtype=np.int64)

        if printTime is True:
            import time
            start = time.time()

        res = assembly(
            old_tet,
            csd_final,
            csd_points,
            ic_bad_tet,
            ic_boundary_tet,
            ic_boundary_vtx,
            self.points,
            self.vertices_ID,
            self.neighbour_ID,
            self.sub_determinants,
            self.num_tet,
            self.boundary_index,
            adjacency_array,
            self.gv,
        )

        self.vertices_ID = res[0]
        self.neighbour_ID = res[1]
        self.sub_determinants = res[2]
        self.num_tet = res[3]
        path_cases_counter = res[4]

        if printTime is True:
            end = time.time()
            t_time = end-start
            print("Time taken to make the triangulation: {} s.".format(t_time))
        if returnPathCases is True:
            return path_cases_counter

        return

    def exportDT(self):
        # Export the present Delaunay triangulation

        points = self.points.copy()
        indices = np.zeros(shape=len(self.vertices_ID), dtype=np.bool_)
        # counter = 0
        for i in np.arange(0, self.num_tet):
            idx = np.where(self.vertices_ID[4*i:4*i+4] == self.gv)[0]
            if len(idx) == 0:
                # i.e. i is not a ghost triangle
                indices[4*i+0] = 1
                indices[4*i+1] = 1
                indices[4*i+2] = 1
                indices[4*i+3] = 1
                # counter += 1
        vertices = self.vertices_ID[indices]
        sub_determinants = self.sub_determinants[indices]
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

    # print("priming numba")
    temp_pts = np.random.rand(30)
    tempDT = Delaunay3D(temp_pts)
    # print("DT initialized")
    tempDT.makeDT()
    # print("numba primed \n")

    del temp_pts
    del tempDT

    np.random.seed(seed=20)

    for i in range(5):
        points = np.random.rand(3*N)
        start = time.time()
        DT = Delaunay3D(points)
        DT.makeDT()
        end = time.time()
        if i == 0:
            running_time = end - start
        else:
            running_time = min(running_time, end-start)
        del DT
        del points

    return running_time

if __name__ == "__main__":
    import sys
    N = int(sys.argv[1])
    time = perf(N)
    print("   Time taken to make the tet-mesh : {} s".format(time))
