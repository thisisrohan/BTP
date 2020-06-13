import numpy as np
import tools.BRIO_2D_multidimarr as BRIO
from tools.adaptive_predicates import incircle, orient2d, exactinit2d
import time
from compyle.types import annotate, int_, KnownType, declare, _get_type
from compyle.low_level import Cython
from compyle.extern import Extern
from math import *

def njit(f):
    return f
from numba import njit


_bool = KnownType('bint')
_boolp = KnownType('bint*', 'bint')


class print_import(Extern):
    def code(self, backend):
        code = 'from libc.stdio cimport printf'
        return code

    def __call__(self, *args):
        pass
printf = print_import()


class _mem_alloc(Extern):
    def code(self, backend):
        if backend != 'cython':
            raise NotImplementedError('currently only supported on Cython')

        code = 'from libc.stdlib cimport malloc\n' + \
               'cdef inline int* mem_alloc (int* arr, int num_elements):\n' + \
               '    arr = <int*> malloc(num_elements * sizeof(int))\n' + \
               '    return arr\n'

        return code

    def __call__(self, *args):
        num_elements = args[1]
        return np.empty(shape=num_elements, dtype=np.int32)
mem_alloc = _mem_alloc()


class _mem_alloc_bint(Extern):
    def code(self, backend):
        if backend != 'cython':
            raise NotImplementedError('currently only supported on Cython')

        code = 'cdef inline bint* mem_alloc_bint (bint* arr, int num_elements):\n' + \
               '    arr = <bint*> malloc(num_elements * sizeof(bint))\n' + \
               '    return arr\n'

        return code

    def __call__(self, *args):
        num_elements = args[1]
        return np.empty(shape=num_elements, dtype=np.bool_)
mem_alloc_bint = _mem_alloc_bint()


class _free_mem(Extern):
    def code(self, backend):
        if backend != 'cython':
            raise NotImplementedError('currently only supported on Cython')

        code = 'from libc.stdlib cimport free\n' + \
               'cdef inline void free_mem (int* arr):\n' + \
               '    free(arr)\n' + \
               '    return\n'

        return code

    def __call__(self, *args):
        pass
free_mem = _free_mem()


class _free_mem_bint(Extern):
    def code(self, backend):
        if backend != 'cython':
            raise NotImplementedError('currently only supported on Cython')

        code = 'cdef inline void free_mem_bint (bint* arr):\n' + \
               '    free(arr)\n' + \
               '    return\n'

        return code

    def __call__(self, *args):
        pass
free_mem_bint = _free_mem_bint()


class _mem_realloc(Extern):
    def code(self, backend):
        if backend != 'cython':
            raise NotImplementedError('currently only supported on Cython')

        code = 'from libc.stdlib cimport realloc\n' + \
               'cdef inline int* mem_realloc (int* arr, int num_elements):\n' + \
               '    arr = <int*> realloc(arr, num_elements * sizeof(int))\n' + \
               '    return arr\n'

        return code

    def __call__(self, *args):
        pass
mem_realloc = _mem_realloc()


# class _max_(Extern):
#     def code(self, backend):
#         if backend != 'cython':
#             raise NotImplementedError('currently only supported on Cython')
#         code = 'cdef inline int max_(int* arr, int arr_size):\n' + \
#                '    cdef int i, res\n' + \
#                '    res = arr[0]\n' + \
#                '    for i in range(1, arr_size):\n' + \
#                '        if res < arr[i]:\n' + \
#                '            res = arr[i]\n' + \
#                '    return res\n'
#         return code

#     def __call__(self, *args):
#         return np.max(args[0][0:args[1]])
# max_ = _max_()


@annotate(array='intp', int='array_size, return_')
def max_(array, array_size):
    i, res = declare('int', 2)
    res = array[0]
    for i in range(1, array_size):
        if res < array[i]:
            res = array[i]
    return res


@annotate(
    int='point_id, t_index, gv, return_',
    doublep='points, exactinit_arr',
    intp='vertices_ID, neighbour_ID')
def _walk(
        point_id, t_index, vertices_ID, neighbour_ID, points, gv,
        exactinit_arr):
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
    exactinit_arr : Contains values relevant to the robust geometric predicates
    '''
    printf("walk entered -- ")
    gv_idx, t_op_index_in_t = declare('int', 2)
    point_x, point_y, a_x, a_y, b_x, b_y, c_x, c_y, det = declare('double', 9)

    gv_idx = 3
    # using t_op_idx_in_t to index into and store the value indexed from the
    # vertices_ID array
    t_op_index_in_t = 3*t_index
    t_op_index_in_t = vertices_ID[t_op_index_in_t]
    if t_op_index_in_t == gv:
        gv_idx = 0
    else:
        t_op_index_in_t = 3*t_index + 1
        t_op_index_in_t = vertices_ID[t_op_index_in_t]
        if t_op_index_in_t == gv:
                gv_idx = 1
        else:
            t_op_index_in_t = 3*t_index + 2
            t_op_index_in_t = vertices_ID[t_op_index_in_t]
            if t_op_index_in_t == gv:
                gv_idx = 2

    if gv_idx != 3:
        # t_index is a ghost tri, in this case simply step into the adjacent
        # real triangle.
        t_index = neighbour_ID[3*t_index + gv_idx] // 3

    gv_idx = 2*point_id  # reusing gv_idx to index into the points array
    point_x = points[gv_idx]
    gv_idx += 1
    point_y = points[gv_idx]

    while True:
        # i.e. t_index is a real tri

        t_op_index_in_t = 4

        # reusing gv_idx to index into the points array
        gv_idx = 3*t_index
        gv_idx = 2*vertices_ID[gv_idx]
        a_x = points[gv_idx]
        gv_idx += 1
        a_y = points[gv_idx]

        gv_idx = 3*t_index + 1
        gv_idx = 2*vertices_ID[gv_idx]
        b_x = points[gv_idx]
        gv_idx += 1
        b_y = points[gv_idx]

        gv_idx = 3*t_index + 2
        gv_idx = 2*vertices_ID[gv_idx]
        c_x = points[gv_idx]
        gv_idx += 1
        c_y = points[gv_idx]

        det = orient2d(point_x, point_y, c_x, c_y, b_x, b_y, exactinit_arr)
        if det > 0:
            t_op_index_in_t = 0
        else:
            det = orient2d(point_x, point_y, a_x, a_y, c_x, c_y, exactinit_arr)
            if det > 0:
                t_op_index_in_t = 1
            else:
                det = orient2d(
                    point_x, point_y, b_x, b_y, a_x, a_y, exactinit_arr)
                if det > 0:
                    t_op_index_in_t = 2

        if t_op_index_in_t != 4:
            # reusing gv_idx to index into the neighbour_ID array
            gv_idx = 3*t_index + t_op_index_in_t
            t_index = neighbour_ID[gv_idx] // 3
        else:
            # point_id lies inside t_index
            break

        # reusing gv_idx to index into the vertices_ID array
        gv_idx = 3*t_index
        # reusing t_op_index_in_t to store this value indexed from the
        # vertices_ID array
        t_op_index_in_t = vertices_ID[gv_idx]
        if t_op_index_in_t == gv:
            break
        else:
            gv_idx += 1
            t_op_index_in_t = vertices_ID[gv_idx]
            if t_op_index_in_t == gv:
                break
            else:
                gv_idx += 1
                t_op_index_in_t = vertices_ID[gv_idx]
                if t_op_index_in_t == gv:
                    break

    printf("walk exited\n")
    return t_index
# walk_ = Cython(_walk)


@annotate(
    int='point_id, t_index, gv',
    doublep='points, exactinit_arr',
    vertices_ID='intp',
    return_=_bool)
def _cavity_helper(
        point_id, t_index, points, vertices_ID, gv, exactinit_arr):
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
               gv : Index assigned to the ghost vertex.
    exactinit_arr : Contains values relevant to the robust geometric predicates                  
    '''
    gv_idx, temp_res = declare('int', 2)
    point_x, point_y, a_x, a_y, b_x, b_y, c_x, c_y = declare('double', 8)
    m1_x, m2_x, m1_y, m2_y, det = declare('double', 5)

    gv_idx = 3
    temp_res = 3*t_index
    temp_res = vertices_ID[temp_res]
    if temp_res == gv:
        gv_idx = 0
    else:
        temp_res = 3*t_index + 1
        temp_res = vertices_ID[temp_res]
        if temp_res == gv:
            gv_idx = 1
        else:
            temp_res = 3*t_index + 2
            temp_res = vertices_ID[temp_res]
            if temp_res == gv:
                gv_idx = 2

    temp_res = 2*point_id
    point_x = points[temp_res]
    temp_res += 1
    point_y = points[temp_res]

    if gv_idx != 3:
        # t_index is a ghost triangle
        temp_res = 3*t_index + (gv_idx + 1) % 3
        temp_res = vertices_ID[temp_res]
        temp_res *= 2
        b_x = points[temp_res]
        temp_res += 1
        b_y = points[temp_res]

        temp_res = 3*t_index + (gv_idx + 2) % 3
        temp_res = vertices_ID[temp_res]
        temp_res *= 2
        c_x = points[temp_res]
        temp_res += 1
        c_y = points[temp_res]

        det = orient2d(point_x, point_y, b_x, b_y, c_x, c_y, exactinit_arr)

        if det > 0:
            return True
        elif det == 0:
            m1_x = point_x - b_x
            m2_x = c_x - point_x
            m1_y = point_y - b_y
            m2_y = c_y - point_y
            # reusing b_x and b_y
            b_x = m1_x*m2_x
            b_y = m1_y*m2_y
            if b_x >= 0 and b_y >= 0:
                return True
            else:
                return False
        else:
            return False
    else:
        # t_index is a real triangle
        temp_res = 3*t_index
        temp_res = vertices_ID[temp_res]
        temp_res *= 2
        a_x = points[temp_res]
        temp_res += 1
        a_y = points[temp_res]

        temp_res = 3*t_index + 1
        temp_res = vertices_ID[temp_res]
        temp_res *= 2
        b_x = points[temp_res]
        temp_res += 1
        b_y = points[temp_res]

        temp_res = 3*t_index + 2
        temp_res = vertices_ID[temp_res]
        temp_res *= 2
        c_x = points[temp_res]
        temp_res += 1
        c_y = points[temp_res]

        det = incircle(
            a_x, a_y, b_x, b_y, c_x, c_y, point_x, point_y, exactinit_arr)

        if det >= 0.0:
            return True
        else:
            return False
# cavity_helper_ = Cython(_cavity_helper)


@annotate(
    int='point_id, t_index, gv, bad_tri_len, boundary_len',
    doublep='points, exactinit_arr',
    intp='neighbour_ID, vertices_ID, bad_tri, boundary_tri, boundary_vtx, ' + \
         'return_arr',
    bad_tri_indicator_arr=_boolp)
def _identify_cavity(
        points, point_id, t_index, neighbour_ID, vertices_ID, bad_tri,
        boundary_tri, boundary_vtx, bad_tri_len, boundary_len, gv,
        bad_tri_indicator_arr, exactinit_arr, return_arr):
    '''
    Identifies all the 'bad' triangles, i.e. the triangles whose circumcircles
    enclose the given point. Returns a list of the indices of the bad triangles
    and a list of the triangles bordering the cavity.

                   points : The global array containing the co-ordinates of all
                            the points to be triangulated.
                 point_id : The index (corresponding to the points array) of
                            the point to be inserted into the triangulation.
                  t_index : The index of the tri enclosing point_id.
             neighbour_ID : The global array containing the indices of the
                            neighbours of all the triangles.
              vertices_ID : The global array containing the indices
                            (corresponding to the points array) of the vertices
                            of all the tri.
                  bad_tri : Helper array, used to store the indices of the
                            'bad' tri, i.e. those whose circumspheres containt
                            point_id.
             boundary_tri : Helper array, used to store the tri on the boundary
                            of the cavity.
             boundary_vtx : Helper array, used to store the points on the
                            boundary of the cavity.
              bad_tri_len : Total length of the bad_tri array to use.
             boundary_len : Total lenght of the boundary_tri array. Twice of
                            this is the length of the boundary_vtx array.
                       gv : Index assigned to the ghost vertex.
    bad_tri_indicator_arr : Indicator array of bints, signals whether a
                            triangle is stored in the bad_tri array. False
                            implies that it hasn't been stored yet.
    '''
    printf("cav\n")
    bad_tri_end, boundary_end, bad_tri_iter, j, jth_nbr_idx = declare('int', 5)
    inside_tri = declare('bint')

    bad_tri_end = return_arr[0]
    boundary_end = return_arr[1]
    bad_tri_iter = return_arr[2]

    # Adding the first bad triangle, i.e. the enclosing triangle
    if bad_tri_end == 0:
        bad_tri[bad_tri_end] = t_index
        bad_tri_indicator_arr[t_index] = True
        bad_tri_end += 1

    while True:
        t_index = bad_tri[bad_tri_iter]

        for j in range(3):
            jth_nbr_idx = neighbour_ID[3*t_index + j] // 3

            if bad_tri_indicator_arr[jth_nbr_idx] == False:
                # i.e. jth_nbr_idx has not been stored in the bad_tri
                # array yet.
                inside_tri = _cavity_helper(
                    point_id, jth_nbr_idx, points, vertices_ID, gv,
                    exactinit_arr)
                if inside_tri:
                    # i.e. the j'th neighbour is a bad triangle
                    if bad_tri_end < bad_tri_len:
                        bad_tri[bad_tri_end] = jth_nbr_idx
                        bad_tri_indicator_arr[jth_nbr_idx] = True
                    elif bad_tri_end == bad_tri_len:
                        return_arr[2] = bad_tri_iter
                    else:
                        pass
                    bad_tri_end += 1
                else:
                    # i.e. the j'th neighbour is a boundary triangle
                    if boundary_end < boundary_len:
                        boundary_tri[boundary_end] = neighbour_ID[3*t_index + j]
                        boundary_vtx[2*boundary_end + 0] = vertices_ID[3*t_index + (j + 1) % 3]
                        boundary_vtx[2*boundary_end + 1] = vertices_ID[3*t_index + (j + 2) % 3]
                    elif boundary_end == boundary_end:
                        return_arr[3] = bad_tri_iter
                    else:
                        pass
                    boundary_end += 1

        bad_tri_iter += 1
        if bad_tri_iter == bad_tri_end:
            break

    return_arr[0] = bad_tri_end
    return_arr[1] = boundary_end
    return
# identify_cavity_ = Cython(_identify_cavity)


@annotate(
    int='point_id, num_tri, gv, bad_tri_end, boundary_end',
    points='doublep',
    intp='neighbour_ID, vertices_ID, bad_tri, boundary_tri, boundary_vtx, ' + \
         'return_arr'
)
def _make_Delaunay_ball(
        point_id, points, neighbour_ID, vertices_ID, num_tri, gv, bad_tri,
        bad_tri_end, boundary_tri, boundary_end, boundary_vtx, return_arr):
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
    i, j, k, t_index, t_info, t1, t2, old_tri = declare('int', 8)

    # populating the cavity with new triangles
    for i in range(boundary_end):
        if i < bad_tri_end:
            t_index = bad_tri[i]
        else:
            t_index = num_tri
            num_tri += 1

        t_info = boundary_tri[i]
        j = 3*t_index
        k = 2*i

        neighbour_ID[j] = t_info
        vertices_ID[j] = point_id
        j += 1
        vertices_ID[j] = boundary_vtx[k]
        j += 1
        k += 1
        vertices_ID[j] = boundary_vtx[k]
        neighbour_ID[t_info] = 3*t_index

    for i in range(boundary_end):
        if i < bad_tri_end:
            t1 = bad_tri[i]
        else:
            t1 = num_tri - (boundary_end - i)
        for j in range(boundary_end):
            if j < bad_tri_end:
                t2 = bad_tri[j]
            else:
                t2 = num_tri - (boundary_end - j)
            k = 3*t1 + 1
            t_info = 3*t2 + 2
            if vertices_ID[k] == vertices_ID[t_info]:
                k = 3*t1 + 2
                t_info = 3*t2 + 1
                neighbour_ID[k] = t_info
                neighbour_ID[t_info] = k
                break

    old_tri =  bad_tri[bad_tri_end - 1]

    return_arr[0] = num_tri
    return_arr[1] = old_tri

    return
# make_Delaunay_ball_ = Cython(_make_Delaunay_ball)


@annotate(
    intp='vertices_ID, neighbour_ID',
    points='doublep',
    int='num_tri, gv')
def _assembly(points, vertices_ID, neighbour_ID, num_tri, gv):

    old_tri, point_id, enclosing_tri, bad_tri_len, boundary_len = declare('int', 5)
    bad_tri_end, boundary_end, new_size, bad_tri_iter, i, t = declare('int', 6)

    bad_tri_len = 64
    boundary_len = 64
    exactinit_arr = declare('matrix(10, "double")')
    return_arr = declare('matrix(4, "int")')

    bad_tri = declare('int*')
    boundary_tri = declare('int*')
    boundary_vtx = declare('int*')
    bad_tri_indicator_arr = declare('bint*')

    bad_tri = mem_alloc(bad_tri, bad_tri_len)
    boundary_tri = mem_alloc(boundary_tri, boundary_len)
    boundary_vtx = mem_alloc(boundary_vtx, 2*boundary_len)
    bad_tri_indicator_arr = mem_alloc_bint(bad_tri_indicator_arr, 2*gv - 2)

    for i in range(2*gv - 2):
        bad_tri_indicator_arr[i] = False

    exactinit2d(points, gv, exactinit_arr)

    old_tri = 0
    for point_id in range(3, gv):
        printf("point_id : %d \n", point_id+1)
        printf("outside walk\n")
        enclosing_tri = _walk(
            point_id, old_tri, vertices_ID, neighbour_ID, points, gv,
            exactinit_arr)
        printf('walk done\n')

        return_arr[0] = 0
        return_arr[1] = 0
        return_arr[2] = 0
        return_arr[3] = 0
        printf("cavity building started -- ")
        while True:
            _identify_cavity(
                points, point_id, enclosing_tri, neighbour_ID, vertices_ID,
                bad_tri, boundary_tri, boundary_vtx, bad_tri_len, boundary_len,
                gv, bad_tri_indicator_arr, exactinit_arr, return_arr)
            bad_tri_end = return_arr[0]
            boundary_end = return_arr[1]
            if bad_tri_end >= bad_tri_len or boundary_end >= boundary_len:
                printf(" -- expanding -- ")
                bad_tri_iter = max_(return_arr, 4)

                if bad_tri_end >= bad_tri_len:
                    if bad_tri_end % 16 == 0:
                        new_size = bad_tri_end
                    else:
                        new_size = 16*(bad_tri_end // 16 + 1)
                    bad_tri = mem_realloc(bad_tri, new_size)
                    bad_tri_len = new_size
                    if return_arr[2] < bad_tri_iter:
                        bad_tri_iter = return_arr[2]

                if boundary_end >= boundary_len:
                    if boundary_end % 16 == 0:
                        new_size = boundary_end
                    else:
                        new_size = 16*(boundary_end // 16 + 1)
                    boundary_tri = mem_realloc(boundary_tri, new_size)
                    boundary_vtx = mem_realloc(boundary_vtx, 2*new_size)
                    boundary_len = new_size
                    if return_arr[3] < bad_tri_iter:
                        bad_tri_iter = return_arr[3]

                return_arr[2] = bad_tri_iter
            else:
                break
        printf("cavity building ended\n")

        printf("cavity retriangulation started\n")
        _make_Delaunay_ball(
            point_id, points, neighbour_ID, vertices_ID, num_tri, gv, bad_tri,
            bad_tri_end, boundary_tri, boundary_end, boundary_vtx, return_arr)
        printf("cavity retriangulated\n")

        num_tri = return_arr[0]
        old_tri = return_arr[1]

        for i in range(bad_tri_end):
            t = bad_tri[i]
            bad_tri_indicator_arr[t] = False

    free_mem(bad_tri)
    free_mem(boundary_tri)
    free_mem(boundary_vtx)
    free_mem_bint(bad_tri_indicator_arr)

    return
assembly = Cython(_assembly)


@njit
def exportDT_njit(
        vertices_ID, neighbour_ID, insertion_seq, num_tri, ghost_tri,
        rectified_vertices, rectified_nbrs, gv):

    gt_end = 0
    rt_end = 0
    for i in range(num_tri):
        is_gt = False
        # The first vertex of a triangle can never be a ghost vertex, since in
        # every newly created traingle the first vertex is set to the point
        # being inserted.
        if vertices_ID[3*i + 1] == gv:
            is_gt = True
        elif vertices_ID[3*i + 2] == gv:
            is_gt = True

        if is_gt == True:
            ghost_tri[gt_end] = i
            gt_end += 1
        else:
            for j in range(3):
                rectified_vertices[rt_end, j] = insertion_seq[
                    vertices_ID[3*i + j]]
                rectified_nbrs[rt_end, j] = neighbour_ID[3*i + j]//3
            rt_end += 1

    for i in range(gt_end-1, -1, -1):
        tri = ghost_tri[i]
        for j in range(rt_end):
            for k in range(3):
                nbr = rectified_nbrs[j, k]
                if nbr > tri:
                    rectified_nbrs[j, k] = nbr - 1
                elif nbr == tri:
                    rectified_nbrs[j, k] = -1

    return rt_end


@annotate(
    points='doublep',
    intp='vertices_ID, neighbour_ID, insertion_seq',
    int='num_points, return_')
def _initialize(points, vertices_ID, neighbour_ID, insertion_seq, num_points):

    a_x, a_y, b_x, b_y, p_x, p_y, signed_area, tempd = declare('double', 8)
    num_tri, idx, tempi = declare('int', 3)

    a_x = points[2*0 + 0]
    a_y = points[2*0 + 1]
    b_x = points[2*1 + 0]
    b_y = points[2*1 + 1]

    num_tri = 0

    idx = 2
    while True:
        p_x = points[2*idx + 0]
        p_y = points[2*idx + 1]
        signed_area = (b_x-a_x)*(p_y-a_y)-(p_x-a_x)*(b_y-a_y)
        if signed_area > 0:
            tempd = points[2*2 + 0]
            points[2*2 + 0] = points[2*idx + 0]
            points[2*idx + 0] = tempd

            tempd = points[2*2 + 1]
            points[2*2 + 1] = points[2*idx + 1]
            points[2*idx + 1] = tempd

            tempi = insertion_seq[2]
            insertion_seq[2] = insertion_seq[idx]
            insertion_seq[idx] = tempi

            break
        
        elif signed_area < 0:
            tempd = points[2*2 + 0]
            points[2*2 + 0] = points[2*idx + 0]
            points[2*idx + 0] = tempd

            tempd = points[2*2 + 1]
            points[2*2 + 1] = points[2*idx + 1]
            points[2*idx + 1] = tempd

            tempi = insertion_seq[2]
            insertion_seq[2] = insertion_seq[idx]
            insertion_seq[idx] = tempi

            tempd = points[2*0 + 0]
            points[2*0 + 0] = points[2*1 + 0]
            points[2*1 + 0] = tempd

            tempd = points[2*0 + 1]
            points[2*0 + 1] = points[2*1 + 1]
            points[2*1 + 1] = tempd

            tempi = insertion_seq[0]
            insertion_seq[0] = insertion_seq[1]
            insertion_seq[1] = tempi

            break
        
        else:
            idx += 1

    vertices_ID[3*0 + 0] = 0      #
    vertices_ID[3*0 + 1] = 1      # ---> 0th triangle [real]
    vertices_ID[3*0 + 2] = 2      #

    vertices_ID[3*1 + 0] = 0               #
    vertices_ID[3*1 + 1] = num_points      # ---> 1st triangle [ghost]
    vertices_ID[3*1 + 2] = 1               #

    vertices_ID[3*2 + 0] = 1               #
    vertices_ID[3*2 + 1] = num_points      # ---> 2nd triangle [ghost]
    vertices_ID[3*2 + 2] = 2               #

    vertices_ID[3*3 + 0] = 2               #
    vertices_ID[3*3 + 1] = num_points      # ---> 3rd triangle [ghost]
    vertices_ID[3*3 + 2] = 0               #

    neighbour_ID[3*0 + 0] = 3*2+1     #
    neighbour_ID[3*0 + 1] = 3*3+1     # ---> 0th triangle [real]
    neighbour_ID[3*0 + 2] = 3*1+1     #

    neighbour_ID[3*1 + 0] = 3*2+2     #
    neighbour_ID[3*1 + 1] = 3*0+2     # ---> 1st triangle [ghost]
    neighbour_ID[3*1 + 2] = 3*3+0     #

    neighbour_ID[3*2 + 0] = 3*3+2     #
    neighbour_ID[3*2 + 1] = 3*0+0     # ---> 2nd triangle [ghost]
    neighbour_ID[3*2 + 2] = 3*1+0     #

    neighbour_ID[3*3 + 0] = 3*1+2     #
    neighbour_ID[3*3 + 1] = 3*0+1     # ---> 3rd triangle [ghost]
    neighbour_ID[3*3 + 2] = 3*2+0     #

    num_tri += 4

    return num_tri
initialize = Cython(_initialize)


class Delaunay2D:

    def __init__(self, points):
        '''
        points : N x 2 array/list of points
        '''
        ### INITIALIZING THE TRIANGULATION ###
        N = len(points)
        self._gv = N
        self._vertices_ID = np.empty(shape=3*(2*N-2), dtype=np.int32)
        self._neighbour_ID = np.empty(shape=3*(2*N-2), dtype=np.int32)
        self._insertion_seq, self._points = BRIO.make_BRIO(
            np.asarray(points, dtype=np.float64).ravel())
        # print("BRIO made")
        num_tri = initialize(
            self._points, self._vertices_ID, self._neighbour_ID,
            self._insertion_seq, N)
        # print("triangulation initialized")

        ### MAKING THE TRIANGULATION ###
        # Arrays that will be passed into the jit-ed functions so that they
        # don't have to get their hands dirty with object creation.
        # bad_tri_indicator_arr = np.zeros(shape=2*N-2, dtype=np.bool_)

        assembly(
            self._points, self._vertices_ID, self._neighbour_ID, num_tri, 
            self._gv)

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
    import time

    np.random.seed(seed=10)

    print("\npriming numba")
    temp_pts = np.random.rand(10, 2)
    tempDT = Delaunay2D(temp_pts)
    print("triangulation made")
    # simplices, nbrs = tempDT.exportDT()
    # print("triangulation exported")
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
        end = time.time()
        time_arr[i] = end - start
        # start = time.time()
        # simplices, nbrs = DT.exportDT()
        # end = time.time()
        print("\nRUN {} : {} s.\n".format(i + 1, time_arr[i]))
        # print("export time : {} s. \n".format(end - start))
        del DT
        del points

    return np.min(time_arr)

if __name__ == "__main__":
    import sys
    N = int(sys.argv[1])
    time = perf(N)
    print("   Time taken to make the triangulation : {} s".format(time))
