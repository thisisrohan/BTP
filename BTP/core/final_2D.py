import numpy as np
import BTP.tools.BRIO_2D as BRIO
from numba import njit


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
    if vertices_ID[3*t_index+0] == gv:
        gv_idx = 0
    elif vertices_ID[3*t_index+1] == gv:
        gv_idx = 1
    elif vertices_ID[3*t_index+2] == gv:
        gv_idx = 2

    if gv_idx != 3:
        # t_index is a ghost tri, in this case simply step into the adjacent
        # real triangle.
        t_index = neighbour_ID[3*t_index+gv_idx]//3

    point_x = points[2*point_id+0]
    point_y = points[2*point_id+1]

    while True:
        # i.e. t_index is a real tri
        t_op_index_in_t = 4

        ax = points[2*vertices_ID[3*t_index+0]+0]
        ay = points[2*vertices_ID[3*t_index+0]+1]
        bx = points[2*vertices_ID[3*t_index+1]+0]
        by = points[2*vertices_ID[3*t_index+1]+1]
        cx = points[2*vertices_ID[3*t_index+2]+0]
        cy = points[2*vertices_ID[3*t_index+2]+1]

        temp = (point_x-ax)*(by-ay) - (point_y-ay)*(bx-ax)

        if temp > 0:
            t_op_index_in_t = 2
        else:
            temp = (point_x-bx)*(cy-by) - (point_y-by)*(cx-bx)
            if temp > 0:
                t_op_index_in_t = 0
            else:
                temp = (point_x-cx)*(ay-cy) - (point_y-cy)*(ax-cx)
                if temp > 0:
                    t_op_index_in_t = 1

        if t_op_index_in_t != 4:
            t_index = neighbour_ID[3*t_index+t_op_index_in_t]//3
        else:
            # point_id lies inside t_index
            break

        if vertices_ID[3*t_index+0] == gv:
            break
        elif vertices_ID[3*t_index+1] == gv:
            break
        elif vertices_ID[3*t_index+2] == gv:
            break

    return t_index


@njit
def _cavity_helper(
    point_id,
    t_index,
    points,
    vertices_ID,
    gv,
):
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
    if vertices_ID[3*t_index+0] == gv:
        gv_idx = 0
    elif vertices_ID[3*t_index+1] == gv:
        gv_idx = 1
    elif vertices_ID[3*t_index+2] == gv:
        gv_idx = 2

    point_x = points[2*point_id+0]
    point_y = points[2*point_id+1]

    if gv_idx != 3:
        # t_index is a ghost triangle
        if gv_idx == 0:
            b_x = points[2*vertices_ID[3*t_index+1]+0]
            b_y = points[2*vertices_ID[3*t_index+1]+1]
            c_x = points[2*vertices_ID[3*t_index+2]+0]
            c_y = points[2*vertices_ID[3*t_index+2]+1]
        elif gv_idx == 1:
            b_x = points[2*vertices_ID[3*t_index+2]+0]
            b_y = points[2*vertices_ID[3*t_index+2]+1]
            c_x = points[2*vertices_ID[3*t_index+0]+0]
            c_y = points[2*vertices_ID[3*t_index+0]+1]
        elif gv_idx == 2:
            b_x = points[2*vertices_ID[3*t_index+0]+0]
            b_y = points[2*vertices_ID[3*t_index+0]+1]
            c_x = points[2*vertices_ID[3*t_index+1]+0]
            c_y = points[2*vertices_ID[3*t_index+1]+1]

        area_t = (point_x-c_x)*(b_y-c_y) - \
                 (point_y-c_y)*(b_x-c_x)

        if area_t > 0:
            return True
        elif area_t == 0:
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
        ax = points[2*vertices_ID[3*t_index+0]+0]
        ay = points[2*vertices_ID[3*t_index+0]+1]
        bx = points[2*vertices_ID[3*t_index+1]+0]
        by = points[2*vertices_ID[3*t_index+1]+1]
        cx = points[2*vertices_ID[3*t_index+2]+0]
        cy = points[2*vertices_ID[3*t_index+2]+1]

        bax_ = bx - ax
        bay_ = by - ay
        cax_ = cx - ax
        cay_ = cy - ay
        normsq_ba = (bx**2 + by**2) - (ax**2 + ay**2)
        normsq_ca = (cx**2 + cy**2) - (ax**2 + ay**2)

        det = (-bay_*normsq_ca + cay_*normsq_ba)*(point_x-ax)
        det += (bax_*normsq_ca - cax_*normsq_ba)*(point_y-ay)
        det += (-bax_*cay_ + cax_*bay_)*(
            (point_x**2 + point_y**2) - (ax**2 + ay**2)
        )

        if det >= 0:
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
    ic_bad_tri,
    ic_boundary_tri,
    ic_boundary_vtx,
    gv,
    bad_tri_indicator_arr,
):
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
            jth_nbr_idx = neighbour_ID[3*t_index+j]//3

            if bad_tri_indicator_arr[jth_nbr_idx]:
                # i.e. jth_nbr_idx has not been stored in the ic_bad_tri
                # array yet.
                inside_tri_flag = _cavity_helper(
                    point_id,
                    jth_nbr_idx,
                    points,
                    vertices_ID,
                    gv,
                )
                if inside_tri_flag:
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
                        3*t_index + j
                    ]
                    ic_boundary_tri_end += 1

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

                    ic_boundary_vtx[ic_boundary_vtx_end+0] = vertices_ID[
                        3*t_index + (j+1) % 3
                    ]
                    ic_boundary_vtx[ic_boundary_vtx_end+1] = vertices_ID[
                        3*t_index + (j+2) % 3
                    ]

                    ic_boundary_vtx_end += 2

        ic_idx += 1

        if ic_idx == ic_bad_tri_end:
            break

    return (ic_bad_tri, ic_bad_tri_end, ic_boundary_tri,
            ic_boundary_tri_end, ic_boundary_vtx)


@njit
def _make_Delaunay_ball(
    point_id,
    bad_tri,
    bad_tri_end,
    boundary_tri,
    boundary_tri_end,
    boundary_vtx,
    points,
    neighbour_ID,
    vertices_ID,
    num_tri,
    gv,
):
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

        neighbour_ID[3*t_index] = boundary_tri[i]
        vertices_ID[3*t_index+0] = point_id
        vertices_ID[3*t_index+1] = boundary_vtx[2*i+0]
        vertices_ID[3*t_index+2] = boundary_vtx[2*i+1]
        neighbour_ID[boundary_tri[i]] = 3*t_index

    for i in range(boundary_tri_end):
            if i < bad_tri_end:
                t1 = bad_tri[i]
            else:
                t1 = num_tri - (boundary_tri_end-1-i) - 1
            for j in range(boundary_tri_end):
                if j < bad_tri_end:
                    t2 = bad_tri[j]
                else:
                    t2 = num_tri - (boundary_tri_end-1-j) - 1
                if vertices_ID[3*t1+1] == vertices_ID[3*t2+2]:
                    neighbour_ID[3*t1+2] = 3*t2+1
                    neighbour_ID[3*t2+1] = 3*t1+2
                    break

    old_tri = bad_tri[bad_tri_end-1]

    if boundary_tri_end < bad_tri_end:
        old_tri = bad_tri[boundary_tri_end-1]
        for k in range(boundary_tri_end, bad_tri_end):
            tri = bad_tri[k]
            for t in range(tri, num_tri):
                vertices_ID[3*t+0] = vertices_ID[3*(t+1)+0]
                vertices_ID[3*t+1] = vertices_ID[3*(t+1)+1]
                vertices_ID[3*t+2] = vertices_ID[3*(t+1)+2]

                neighbour_ID[3*t+0] = neighbour_ID[3*(t+1)+0]
                neighbour_ID[3*t+1] = neighbour_ID[3*(t+1)+1]
                neighbour_ID[3*t+2] = neighbour_ID[3*(t+1)+2]

            num_tri -= 1

            for i in range(num_tri):
                for j in range(3):
                    if neighbour_ID[3*i+j]//3 > tri:
                        neighbour_ID[3*i+j] = 3*(neighbour_ID[3*i+j]//3-1) + \
                                              neighbour_ID[3*i+j] % 3

            for i in range(k+1, bad_tri_end):
                if bad_tri[i] > tri:
                    bad_tri[i] -= 1

    return num_tri, old_tri


@njit
def assembly(
    old_tri,
    ic_bad_tri,
    ic_boundary_tri,
    ic_boundary_vtx,
    points,
    vertices_ID,
    neighbour_ID,
    num_tri,
    gv,
    bad_tri_indicator_arr,
):
    for point_id in np.arange(3, gv):

        enclosing_tri = _walk(
            point_id,
            old_tri,
            vertices_ID,
            neighbour_ID,
            points,
            gv,
        )

        cavity_results = _identify_cavity(
            points,
            point_id,
            enclosing_tri,
            neighbour_ID,
            vertices_ID,
            ic_bad_tri,
            ic_boundary_tri,
            ic_boundary_vtx,
            gv,
            bad_tri_indicator_arr,
        )

        ic_bad_tri = cavity_results[0]
        ic_bad_tri_end = cavity_results[1]
        ic_boundary_tri = cavity_results[2]
        ic_boundary_tri_end = cavity_results[3]
        ic_boundary_vtx = cavity_results[4]

        num_tri, old_tri = _make_Delaunay_ball(
            point_id,
            ic_bad_tri,
            ic_bad_tri_end,
            ic_boundary_tri,
            ic_boundary_tri_end,
            ic_boundary_vtx,
            points,
            neighbour_ID,
            vertices_ID,
            num_tri,
            gv,
        )

        for i in range(ic_bad_tri_end):
            t = ic_bad_tri[i]
            bad_tri_indicator_arr[t] = False

    return num_tri


@njit
def initialize(
    points,
    vertices_ID,
    neighbour_ID,
):
    N = int(len(points)/2)

    a_x = points[0]
    a_y = points[1]
    b_x = points[2]
    b_y = points[3]

    num_tri = np.int64(0)

    idx = 2
    while True:
        p_x = points[2*idx]
        p_y = points[2*idx+1]
        signed_area = (b_x-a_x)*(p_y-a_y)-(p_x-a_x)*(b_y-a_y)
        if signed_area > 0:
            points[4], points[2*idx] = points[2*idx], points[4]
            points[4+1], points[2*idx+1] = points[2*idx+1], points[4+1]
            break
        elif signed_area < 0:
            points[4], points[2*idx] = points[2*idx], points[4]
            points[4+1], points[2*idx+1] = points[2*idx+1], points[4+1]
            points[0], points[2] = points[2], points[0]
            points[1], points[3] = points[3], points[1]
            break
        else:
            idx += 1

    vertices_ID[0] = 0      #
    vertices_ID[1] = 1      # ---> 0th triangle [real]
    vertices_ID[2] = 2      #

    vertices_ID[3] = 0      #
    vertices_ID[4] = N      # ---> 1st triangle [ghost]
    vertices_ID[5] = 1      #

    vertices_ID[6] = 1      #
    vertices_ID[7] = N      # ---> 2nd triangle [ghost]
    vertices_ID[8] = 2      #

    vertices_ID[9] = 2      #
    vertices_ID[10] = N     # ---> 3rd triangle [ghost]
    vertices_ID[11] = 0     #

    neighbour_ID[0] = 3*2+1     #
    neighbour_ID[1] = 3*3+1     # ---> 0th triangle [real]
    neighbour_ID[2] = 3*1+1     #

    neighbour_ID[3] = 3*2+2     #
    neighbour_ID[4] = 3*0+2     # ---> 1st triangle [ghost]
    neighbour_ID[5] = 3*3+0     #

    neighbour_ID[6] = 3*3+2     #
    neighbour_ID[7] = 3*0+0     # ---> 2nd triangle [ghost]
    neighbour_ID[8] = 3*1+0     #

    neighbour_ID[9] = 3*1+2     #
    neighbour_ID[10] = 3*0+1    # ---> 3rd triangle [ghost]
    neighbour_ID[11] = 3*2+0    #

    num_tri += 4

    return num_tri


class Delaunay2D:

    def __init__(self, points):

        N = int(len(points)/2)

        self.gv = N

        self.vertices_ID = N*np.ones(3*(2*N-2), dtype=np.int64)
        self.neighbour_ID = np.empty(3*(2*N-2), dtype=np.int64)

        self.points = BRIO.make_BRIO(points)

        self.num_tri = initialize(
            self.points,
            self.vertices_ID,
            self.neighbour_ID,
        )

    def makeDT(self):

        old_tri = np.int64(0)

        # Arrays that will be passed into the jit-ed functions so that they
        # don't have to get their hands dirty with object creation.
        ic_bad_tri = np.empty(50, dtype=np.int64)
        ic_boundary_tri = np.empty(50, dtype=np.int64)
        ic_boundary_vtx = np.empty(2*50, dtype=np.int64)
        bad_tri_indicator_arr = np.zeros(shape=2*self.gv-2, dtype=np.bool_)

        self.num_tri = assembly(
            old_tri,
            ic_bad_tri,
            ic_boundary_tri,
            ic_boundary_vtx,
            self.points,
            self.vertices_ID,
            self.neighbour_ID,
            self.num_tri,
            self.gv,
            bad_tri_indicator_arr,
        )


def perf(N):
    import time

    np.random.seed(seed=10)

    print("\npriming numba")
    temp_pts = np.random.rand(20)
    tempDT = Delaunay2D(temp_pts)
    print("DT initialized")
    tempDT.makeDT()
    print("numba primed \n")

    del temp_pts
    del tempDT

    np.random.seed(seed=20)

    for i in range(5):
        points = np.random.rand(2*N)
        start = time.time()
        DT = Delaunay2D(points)
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
    print("   Time taken to make the triangulation : {} s".format(time))
