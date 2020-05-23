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
intpp = KnownType('int**', 'int')


class _mem_alloc(Extern):
    def code(self, backend):
        if backend != 'cython':
            raise NotImplementedError('currently only supported on Cython')
        code = 'from libc.stdlib cimport malloc\n' + \
               'cdef inline void mem_alloc (int** arr, int num_elements):\n' + \
               '    *arr = malloc(num_elements * sizeof(int))\n' + \
               '    if *arr == NULL:\n' + \
               '        raise MemoryError()\n'
        return code

    def __call__(self, *args):
        # num_elements = args[1]
        # return np.empty(shape=num_elements, dtype=np.float64)
        pass
mem_alloc = _mem_alloc()


class _free_mem(Extern):
    def code(self, backend):
        if backend != 'cython':
            raise NotImplementedError('currently only supported on Cython')
        code = 'from libc.stdlib cimport free\n' + \
               'cdef inline void free_mem (int** arr):\n' + \
               '    free(*arr)\n'
        return code

    def __call__(self, *args):
        # del args[0]
        pass
free_mem = _free_mem()


class _mem_realloc(Extern):
    def code(self, backend):
        if backend != 'cython':
            raise NotImplementedError('currently only supported on Cython')
        code = 'from libc.stdlib cimport realloc\n' + \
               'cdef inline void mem_realloc (int** arr, int new_size):\n' + \
               '    *arr = realloc(*arr, new_size*sizeof(int))\n' + \
               '    if *arr == NULL:\n' + \
               '        raise MemoryError()\n'
        return code

    def __call__(self, *args):
        pass
mem_realloc = _mem_realloc()


class _set_ptr(Extern):
    def code(self, backend):
        if backend != 'cython':
            raise NotImplementedError('currently only supported on Cython')
        code = 'cdef inline void set_ptr(int** &pp, int* &p):\n' + \
               '    pp = &p'
        return code

    def __call__(self, *args):
        pass
set_ptr = _set_ptr()


class _max_(Extern):
    def code(self, backend):
        if backend != 'cython':
            raise NotImplementedError('currently only supported on Cython')
        code = 'cdef inline int max_(int* arr, int arr_size):\n' + \
               '    cdef int i, res\n' + \
               '    res = arr[0]\n' + \
               '    for i in range(1, arr_size):\n' + \
               '        if res < arr[i]:\n' + \
               '            res = arr[i]\n' + \
               '    return res'
        return code

    def __call__(self, *args):
        return np.max(args[0][0:args[1]])
max_ = _max_()


@annotate(
    double='splitter, ccwerrboundA, ccwerrboundB, ccwerrboundC, ' + \
           'resulterrbound, static_filter_o2d',
    int='point_id, t_index, gv, return_',
    points='doublep',
    intp='vertices_ID, neighbour_ID')
def _walk(
        point_id, t_index, vertices_ID, neighbour_ID, points, gv, splitter,
        ccwerrboundA, ccwerrboundB, ccwerrboundC, resulterrbound,
        static_filter_o2d):
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
    gv_idx, t_op_index_in_t = declare('int', 2)
    point_x, point_y, a_x, a_y, b_x, b_y, c_x, c_y, det_left, det_right, det, \
    detsum = declare('double', 12)

    gv_idx = 3
    if vertices_ID[3*t_index + 1] == gv:
        gv_idx = 1
    elif vertices_ID[3*t_index + 2] == gv:
        gv_idx = 2

    if gv_idx != 3:
        # t_index is a ghost tri, in this case simply step into the adjacent
        # real triangle.
        t_index = neighbour_ID[3*t_index + gv_idx]//3

    point_x = points[2*point_id + 0]
    point_y = points[2*point_id + 1]

    while True:
        # i.e. t_index is a real tri

        t_op_index_in_t = 4

        a_x = points[2*vertices_ID[3*t_index + 0] + 0]
        a_y = points[2*vertices_ID[3*t_index + 0] + 1]
        b_x = points[2*vertices_ID[3*t_index + 1] + 0]
        b_y = points[2*vertices_ID[3*t_index + 1] + 1]
        c_x = points[2*vertices_ID[3*t_index + 2] + 0]
        c_y = points[2*vertices_ID[3*t_index + 2] + 1]

        det_left = (point_x-b_x)*(c_y-b_y)
        det_right = (point_y-b_y)*(c_x-b_x)
        det = det_left - det_right
        if abs(det) < static_filter_o2d:
            detsum = abs(det_left) + abs(det_right)
            det = orient2d(
                point_x, point_y, c_x, c_y, b_x, b_y, splitter, ccwerrboundA,
                ccwerrboundB, ccwerrboundC, resulterrbound, det, detsum)
        if det > 0:
            t_op_index_in_t = 0
        else:
            det_left = (point_x-c_x)*(a_y-c_y)
            det_right = (point_y-c_y)*(a_x-c_x)
            det = det_left - det_right
            if abs(det) < static_filter_o2d:
                detsum = abs(det_left) + abs(det_right)
                det = orient2d(
                    point_x, point_y, a_x, a_y, c_x, c_y, splitter,
                    ccwerrboundA, ccwerrboundB, ccwerrboundC, resulterrbound,
                    det, detsum)
            if det > 0:
                t_op_index_in_t = 1
            else:
                det_left = (point_x-a_x)*(b_y-a_y)
                det_right =  (point_y-a_y)*(b_x-a_x)
                det = det_left - det_right
                if abs(det) < static_filter_o2d:
                    detsum = abs(det_left) + abs(det_right)
                    det = orient2d(
                        point_x, point_y, b_x, b_y, a_x, a_y, splitter,
                        ccwerrboundA, ccwerrboundB, ccwerrboundC,
                        resulterrbound, det, detsum)
                if det > 0:
                    t_op_index_in_t = 2

        if t_op_index_in_t != 4:
            t_index = neighbour_ID[3*t_index +t_op_index_in_t]//3
        else:
            # point_id lies inside t_index
            break

        if vertices_ID[3*t_index + 1] == gv:
            break
        elif vertices_ID[3*t_index + 2] == gv:
            break

    return t_index


@annotate(
    double='splitter, iccerrboundA, iccerrboundB, iccerrboundC, ' + \
           'resulterrbound, ccwerrboundA, ccwerrboundB, ccwerrboundC, ' + \
           'static_filter_o2d, static_filter_i2d',
    int='point_id, t_index, gv',
    points='doublep',
    vertices_ID='intp',
    return_=_bool)
def _cavity_helper(
        point_id, t_index, points, vertices_ID, gv, splitter, iccerrboundA,
        iccerrboundB, iccerrboundC, resulterrbound, ccwerrboundA, ccwerrboundB,
        ccwerrboundC, static_filter_o2d, static_filter_i2d):
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
    gv_idx = declare('int', 1)
    point_x, point_y, a_x, a_y, b_x, b_y, c_x, c_y, det_left, det_right, det, \
    detsum, m1_x, m2_x, m1_y, m2_y, adx, bdx, cdx, ady, bdy, cdy, bdxcdy, \
    cdxbdy, alift, cdxady, adxcdy, blift, adxbdy, bdxady, clift, \
    permanent = declare('double', 32)

    gv_idx = 3
    if vertices_ID[3*t_index + 1] == gv:
        gv_idx = 1
    elif vertices_ID[3*t_index + 2] == gv:
        gv_idx = 2

    point_x = points[2*point_id + 0]
    point_y = points[2*point_id + 1]

    if gv_idx != 3:
        # t_index is a ghost triangle
        b_x = points[2*vertices_ID[3*t_index + (gv_idx + 1) % 3] + 0]
        b_y = points[2*vertices_ID[3*t_index + (gv_idx + 1) % 3] + 1]
        c_x = points[2*vertices_ID[3*t_index + (gv_idx + 2) % 3] + 0]
        c_y = points[2*vertices_ID[3*t_index + (gv_idx + 2) % 3] + 1]

        det_left = (point_x-c_x)*(b_y-c_y)
        det_right = (point_y-c_y)*(b_x-c_x)
        det = det_left - det_right
        if abs(det) <= static_filter_o2d:
            detsum = abs(det_left) + abs(det_right)
            det = orient2d(
                point_x, point_y, b_x, b_y, c_x, c_y, splitter, ccwerrboundA,
                ccwerrboundB, ccwerrboundC, resulterrbound, det, detsum)

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
        a_x = points[2*vertices_ID[3*t_index + 0] + 0]
        a_y = points[2*vertices_ID[3*t_index + 0] + 1]
        b_x = points[2*vertices_ID[3*t_index + 1] + 0]
        b_y = points[2*vertices_ID[3*t_index + 1] + 1]
        c_x = points[2*vertices_ID[3*t_index + 2] + 0]
        c_y = points[2*vertices_ID[3*t_index + 2] + 1]

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
        if abs(det) <= static_filter_i2d:
            permanent = (abs(bdxcdy) + abs(cdxbdy)) * alift + \
                        (abs(cdxady) + abs(adxcdy)) * blift + \
                        (abs(adxbdy) + abs(bdxady)) * clift
            det = incircle(
                a_x, a_y, b_x, b_y, c_x, c_y, point_x, point_y, splitter,
                iccerrboundA, iccerrboundB, iccerrboundC, resulterrbound, det,
                permanent)

        if det >= 0.0:
            return True
        else:
            return False


@annotate(
    double='splitter, iccerrboundA, iccerrboundB, iccerrboundC, ' + \
           'resulterrbound, ccwerrboundA, ccwerrboundB, ccwerrboundC, ' + \
           'static_filter_o2d, static_filter_i2d',
    int='point_id, t_index, gv, bad_tri_len, boundary_len',
    points='doublep',
    intp='neighbour_ID, vertices_ID, ic_bad_tri, ic_boundary_tri, ' + \
         'ic_boundary_vtx, return_arr',
    bad_tri_indicator_arr=_boolp)
def _identify_cavity(
        points, point_id, t_index, neighbour_ID, vertices_ID, bad_tri,
        boundary_tri, boundary_vtx, bad_tri_len, boundary_len, gv,
        bad_tri_indicator_arr, splitter, iccerrboundA, iccerrboundB,
        iccerrboundC, resulterrbound, ccwerrboundA, ccwerrboundB, ccwerrboundC,
        static_filter_o2d, static_filter_i2d, return_arr):
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
            jth_nbr_idx = neighbour_ID[3*t_index + j]//3

            if not bad_tri_indicator_arr[jth_nbr_idx]:
                # i.e. jth_nbr_idx has not been stored in the bad_tri
                # array yet.
                inside_tri = _cavity_helper(
                    point_id, jth_nbr_idx, points, vertices_ID, gv, splitter,
                    iccerrboundA, iccerrboundB, iccerrboundC, resulterrbound,
                    ccwerrboundA, ccwerrboundB, ccwerrboundC,
                    static_filter_o2d, static_filter_i2d)
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
                        boundary_tri[boundary_end] = neighbour_ID[
                            3*t_index + j]
                        boundary_vtx[2*boundary_end + 0] = vertices_ID[
                            3*t_index + (j+1) % 3]
                        boundary_vtx[2*boundary_end + 1] = vertices_ID[
                            3*t_index + (j+2) % 3]
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


@annotate(
    int='point_id, num_tri, gv, bad_tri_end, boundary_end',
    points='doublep',
    intp='neighbour_ID, vertices_ID, bad_tri, boundary_tri, boundary_vtx' + \
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
    i, j, t_index, t_info, t1, t2, old_tri = declare('int', 7)

    # populating the cavity with new triangles
    for i in range(boundary_end):
        if i < bad_tri_end:
            t_index = bad_tri[i]
        else:
            t_index = num_tri
            num_tri += 1

        t_info = boundary_tri[i]
        neighbour_ID[3*t_index + 0] = t_info
        vertices_ID[3*t_index + 0] = point_id
        vertices_ID[3*t_index + 1] = boundary_vtx[2*i + 0]
        vertices_ID[3*t_index + 2] = boundary_vtx[2*i + 1]
        neighbour_ID[tinfo] = 3*t_index

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
                if vertices_ID[3*t1 + 1] == vertices_ID[3*t2 + 2]:
                    neighbour_ID[3*t1 + 2] = 3*t2 + 1
                    neighbour_ID[3*t2 + 1] = 3*t1 + 2
                    break

    old_tri =  bad_tri[bad_tri_end - 1]

    if boundary_end < bad_tri_end:
        k, tri, t = declare('int', 3)
        old_tri = bad_tri[boundary_end-1]
        for k in range(boundary_end, bad_tri_end):
            tri = bad_tri[k]
            for t in range(tri, num_tri):
                vertices_ID[3*t + 0] = vertices_ID[3*(t + 1) + 0]
                vertices_ID[3*t + 1] = vertices_ID[3*(t + 1) + 1]
                vertices_ID[3*t + 2] = vertices_ID[3*(t + 1) + 2]

                neighbour_ID[3*t + 0] = neighbour_ID[3*(t + 1) + 0]
                neighbour_ID[3*t + 1] = neighbour_ID[3*(t + 1) + 1]
                neighbour_ID[3*t + 2] = neighbour_ID[3*(t + 1) + 2]

            num_tri -= 1

            for i in range(num_tri):
                for j in range(3):
                    if neighbour_ID[3*i + j]//3 > tri:
                        neighbour_ID[3*i + j] = \
                            3*(neighbour_ID[3*i + j]//3 - 1) + \
                            neighbour_ID[3*i + j] % 3

            for i in range(k+1, bad_tri_end):
                if bad_tri[i] > tri:
                    bad_tri[i] -= 1

    return_arr[0] = num_tri
    return_arr[1] = old_tri
    return


@annotate(
    intp='vertices_ID, neighbour_ID',
    points='doublep',
    int='num_tri, gv',
    bad_tri_indicator_arr=_boolp)
def _assembly(
        points, vertices_ID, neighbour_ID, num_tri, gv, bad_tri_indicator_arr):

    splitter, resulterrbound, ccwerrboundA, ccwerrboundB, ccwerrboundC, \
    iccerrboundA, iccerrboundB, iccerrboundC, static_filter_o2d, \
    static_filter_i2d = declare('double', 10)
    old_tri, point_id, enclosing_tri, bad_tri_len, boundary_len, bad_tri_end, \
    boundary_end, new_size, bad_tri_iter, i, t = declare('int', 11)

    bad_tri_len = 64
    boundary_len = 64
    exactinit_arr = declare('matrix(10, "double")')
    return_arr = declare('matrix(4, "int")')

    bad_tri, boundary_tri, boundary_vtx = declare('int*', 3)
    bad_tri_p, boundary_tri_p, boundary_vtx_p = declare('int**', 3)
    
    set_ptr(bad_tri_p, bad_tri)
    mem_alloc(bad_tri_p, bad_tri_len)
    
    set_ptr(boundary_tri_p, boundary_tri)
    mem_alloc(boundary_tri_p, boundary_len)
    
    set_ptr(boundary_vtx_p, boundary_vtx)
    mem_alloc(boundary_vtx_p, 2*boundary_len)

    exactinit2d(points, gv, exactinit_arr)
    splitter = exactinit_arr[0]
    resulterrbound = exactinit_arr[1]
    ccwerrboundA = exactinit_arr[2]
    ccwerrboundB = exactinit_arr[3]
    ccwerrboundC = exactinit_arr[4]
    iccerrboundA = exactinit_arr[5]
    iccerrboundB = exactinit_arr[6]
    iccerrboundC = exactinit_arr[7]
    static_filter_o2d = exactinit_arr[8]
    static_filter_i2d = exactinit_arr[9]

    old_tri = 0
    for point_id in range(3, gv):
        enclosing_tri = _walk(
            point_id, old_tri, vertices_ID, neighbour_ID, points, gv, splitter,
            ccwerrboundA, ccwerrboundB, ccwerrboundC, resulterrbound,
            static_filter_o2d)

        return_arr[0] = 0
        return_arr[1] = 0
        return_arr[2] = 0
        return_arr[3] = 0        
        while True:
            _identify_cavity(
                points, point_id, enclosing_tri, neighbour_ID, vertices_ID,
                bad_tri, boundary_tri, boundary_vtx, bad_tri_len, boundary_len,
                gv, bad_tri_indicator_arr, splitter, iccerrboundA,
                iccerrboundB, iccerrboundC, resulterrbound, ccwerrboundA,
                ccwerrboundB, ccwerrboundC, static_filter_o2d,
                static_filter_i2d, return_arr)
            bad_tri_end = return_arr[0]
            boundary_end = return_arr[1]
            if bad_tri_end >= bad_tri_len or boundary_end >= boundary_len:
                bad_tri_iter = max_(return_arr, 4)

                if bad_tri_end >= bad_tri_len:
                    if bad_tri_end % 16 == 0:
                        new_size = bad_tri_end
                    else:
                        new_size = 16*(bad_tri_end//16 + 1)
                    mem_realloc(bad_tri_p, new_size)
                    bad_tri_len = new_size
                    if return_arr[2] < bad_tri_iter:
                        bad_tri_iter = return_arr[2]

                if boundary_end >= boundary_len:
                    if boundary_end % 16 == 0:
                        new_size = boundary_end
                    else:
                        new_size = 16*(boundary_end//16 + 1)
                    mem_realloc(boundary_tri_p, new_size)
                    mem_realloc(boundary_vtx_p, 2*new_size)
                    boundary_len = new_size
                    if return_arr[3] < bad_tri_iter:
                        bad_tri_iter = return_arr[3]

                return_arr[2] = bad_tri_iter
            else:
                break

        _make_Delaunay_ball(
            point_id, points, neighbour_ID, vertices_ID, num_tri, gv, bad_tri,
            bad_tri_end, boundary_tri, boundary_end, boundary_vtx, return_arr)

        num_tri = return_arr[0]
        old_tri = return_arr[1]

        for i in range(bad_tri_end):
            t = bad_tri[i]
            bad_tri_indicator_arr[t] = False

    free_mem(bad_tri_p)
    free_mem(boundary_tri_p)
    free_mem(boundary_vtx_p)

    return
assembly = Cython(_assembly)


@njit(cache=True)
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
        if vertices_ID[i, 1] == gv:
            is_gt = True
        elif vertices_ID[i, 2] == gv:
            is_gt = True

        if is_gt == True:
            ghost_tri[gt_end] = i
            gt_end += 1
        else:
            for j in range(3):
                rectified_vertices[rt_end, j] = insertion_seq[
                    vertices_ID[i, j]]
                rectified_nbrs[rt_end, j] = neighbour_ID[i, j]//3
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
        self._vertices_ID = np.empty(shape=3*(2*N-2), dtype=np.int64)
        self._neighbour_ID = np.empty(shape=3*(2*N-2), dtype=np.int64)
        self._insertion_seq, self._points = BRIO.make_BRIO(
            np.asarray(points, dtype=np.float64).ravel())
        num_tri = initialize(
            self._points, self._vertices_ID, self._neighbour_ID,
            self._insertion_seq, N)

        ### MAKING THE TRIANGULATION ###
        # Arrays that will be passed into the jit-ed functions so that they
        # don't have to get their hands dirty with object creation.
        bad_tri_indicator_arr = np.zeros(shape=2*N-2, dtype=np.bool_)

        assembly(
            self._points, self._vertices_ID, self._neighbour_ID, num_tri, 
            self._gv, bad_tri_indicator_arr)

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
    simplices, nbrs = tempDT.exportDT()
    print("triangulation exported")
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
        start = time.time()
        simplices, nbrs = DT.exportDT()
        end = time.time()
        print("RUN {} : {} s.".format(i, time_arr[i]))
        print("export time : {} s. \n".format(end - start))
        del DT
        del points

    return np.min(time_arr)

if __name__ == "__main__":
    import sys
    N = int(sys.argv[1])
    time = perf(N)
    print("   Time taken to make the triangulation : {} s".format(time))
