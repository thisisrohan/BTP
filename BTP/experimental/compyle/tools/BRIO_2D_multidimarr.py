import numpy as np
from numba import njit
from compyle.types import annotate, int_, KnownType, declare, _get_type
from compyle.low_level import Cython
from compyle.extern import Extern
from math import *


class _intcast_d(Extern):
    def code(self, backend):
        if backend != 'cython':
            raise NotImplementedError('currently only supported on Cython')
        return 'cdef inline int intcast_d(double x):\n    return <int> x\n'

    def __call__(self, *args):
        return int(*args)
intcast_d = _intcast_d()


class _random_num(Extern):
    def code(self, backend):
        if backend != 'cython':
            raise NotImplementedError('currently only supported on Cython')
        code = 'from libc.stdlib cimport rand, srand\n' + \
               'from libc.time cimport *\n' + \
               'srand(time(NULL))\n' + \
               'cdef inline int random_num(int choices):\n' + \
               '    return rand() % choices\n'
        return code

    def __call__(self, choices):
        return np.random.randint(low=0, high=choices)
random_num = _random_num()


class _qargsort2(Extern):
    def code(self, backend):
        if backend != 'cython':
            raise NotImplementedError('currently only supported on Cython')
        # code = 'cdef extern from "qargsort.c":\n' + \
        #        '    void qargsort (int* points, int* indices, int array_size)'
        
        code = 'cdef inline void qargsort2 (int* points, int* indices, int start, int end):\n' + \
               '    cdef int left, right, pivot_idx, pivot_point, temp, array_size\n' + \
               '    array_size = end - start\n\n' + \
               '    if array_size == 2:\n' + \
               '        # Recursive base case\n' + \
               '        if points[indices[start]] > points[indices[start + 1]]:\n' + \
               '            temp = indices[start]\n' + \
               '            indices[start] = indices[start + 1]\n' + \
               '            indices[start + 1] = temp\n' + \
               '        return\n\n' + \
               '    # Choose a pivot to split the array.\n' + \
               '    pivot_idx = array_size // 2\n' + \
               '    pivot_point = points[indices[start + pivot_idx]]\n' + \
               '    # Split the array.\n' + \
               '    left = -1\n' + \
               '    right = array_size\n' + \
               '    while left < right:\n' + \
               '        # Search for a point which is too large for the left indices.\n' + \
               '        left += 1\n' + \
               '        while left <= right and points[indices[start + left]] < pivot_point:\n' + \
               '            left += 1\n' + \
               '        # Search for a point which is too small for the right indices.\n' + \
               '        right -= 1\n' + \
               '        while left <= right and points[indices[start + right]] > pivot_point:\n' + \
               '            right -= 1\n' + \
               '        if left < right:\n' + \
               '            # Swap the left and right indices.\n' + \
               '            temp = indices[start + left]\n' + \
               '            indices[start + left] = indices[start + right]\n' + \
               '            indices[start + right] = temp\n\n' + \
               '    if left > 1:\n' + \
               '        # Recursively sort the left subset\n' + \
               '        qargsort2(points, indices, start, start + left)\n' + \
               '    if right < array_size - 2:\n' + \
               '        # Recursively sort the right subset.\n' + \
               '        qargsort2(points, indices, start + right + 1, end)\n'

        return code

    def __call__(self, *args, **kw):
        # print('passed')
        pass
qargsort2 = _qargsort2()


@annotate(seed='intp', int='low, high, return_')
def prng(seed, low, high):
    '''
    seed[0] = seed for this prng
    '''
    randomseed = declare('int')
    randomseed = seed[0]
    randomseed = (randomseed*1313 + 1414) % 15151515
    seed[0] = randomseed
    randomseed = ( randomseed // ( (15151515 // (high - low) ) + 1) ) - low
    return randomseed


@annotate(intp='points, indices', int='start, end')
def qargsort(points, indices, start, end):
    left, right, pivot_idx, pivot_point, temp, array_size = declare('int', 6)
    array_size = end - start

    if array_size == 2:
        # Recursive base case
        if points[indices[start]] > points[indices[start + 1]]:
            temp = indices[start]
            indices[start] = indices[start + 1]
            indices[start + 1] = temp
        return

    # Choose a pivot to split the array
    pivot_idx = array_size // 2
    pivot_point = points[indices[start + pivot_idx]]

    # Split the array
    left = -1
    right = array_size
    while left < right:
        # Search for a point that is too large for the left indices
        left += 1
        while left <= right and points[indices[start + left]] < pivot_point:
            left += 1

        # Search for a point that is too small for the right indices
        right -= 1
        while left <= right and points[indices[start + right]] > pivot_point:
            right -= 1

        if left < right:
            # Swap left and right indices
            temp = indices[start + left]
            indices[start + left] = indices[start + right]
            indices[start + right] = temp

    if left > 1:
        # Recursively sort the left subset
        qargsort(points, indices, start, start + left)
    if right < array_size - 2:
        # Recursively sort the right subset
        qargsort(points, indices, start + right + 1, end)
    return
# qargsort = Cython(qargsort)


@annotate(x='double', return_='int')
def _round_(x):
    res = declare('int')
    res = intcast_d(x // 1.0)
    if x % 1.0 >= 0.5:
        res += 1
    return res


@annotate(
    intp='rounds, boundary_indices, points_left_old, points_left_new',
    len_points='int')
def _make_rounds(
        rounds, boundary_indices, points_left_old, points_left_new,
        len_points):
    '''
    Divides the points into different groups, of increasing sizes,
    according to the Biased Randomized Insertion Order.

              rounds : Nx1 array; stores the indices of points corresponding to
                       a round.
    boundary_indices : Stores the indices (corresponding to the rounds array),
                       where the rounds end.
     points_left_old : Nx1 array
     points_left_new : Nx1 array
    '''

    num_rounds = declare('int')
    num_rounds = _round_(log2(len_points))

    points_left_old_end, points_left_new_end = declare('int', 2)
    points_left_old_end = len_points
    points_left_new_end = 0

    boundary_indices_insertion_idx, rounds_insertion_idx = declare('int', 2)
    boundary_indices_insertion_idx = num_rounds
    rounds_insertion_idx = len_points-1

    boundary_indices[boundary_indices_insertion_idx] = rounds_insertion_idx+1
    boundary_indices_insertion_idx -= 1

    round_idx = declare('int')
    i, rnum = declare('int', 2)
    seed = declare('matrix(1, "int")')
    seed[0] = 0
    for round_idx in range(1, num_rounds):
        points_left_new_end = 0

        for i in range(points_left_old_end):
            # rnum = random_num(2)
            rnum = prng(seed, 0, 2)
            if rnum == 1:
                rounds[rounds_insertion_idx] = points_left_old[i]
                rounds_insertion_idx -= 1
            else:
                points_left_new[points_left_new_end] = points_left_old[i]
                points_left_new_end += 1

        for i in range(points_left_new_end):
            points_left_old[i] = points_left_new[i]

        points_left_old_end = points_left_new_end

        boundary_indices[
            boundary_indices_insertion_idx
        ] = rounds_insertion_idx + 1
        boundary_indices_insertion_idx -= 1

    for i in range(points_left_old_end):
        rounds[rounds_insertion_idx] = points_left_old[i]
        rounds_insertion_idx -= 1

    boundary_indices[boundary_indices_insertion_idx] = rounds_insertion_idx + 1
    boundary_indices_insertion_idx -= 1

    return
make_rounds = Cython(_make_rounds)


@annotate(int='hindex, p', return_arr='intp')
def _hindex2xy(hindex, p, return_arr):
    '''
    hindex : Hilbert index of the point
         p : Iteration of the Hilbert curve
    '''

    place, x, y, i, n, tmp = declare('int', 6)
    place = hindex & 3
    if place == 0:
        x = 0
        y = 0
    elif place == 1:
        x = 0
        y = 1
    elif place == 2:
        x = 1
        y = 1
    elif place == 3:
        x = 1
        y = 0

    hindex = hindex >> 2

    for i in range(1, p):
        n = 2**i
        place = hindex & 3

        if place == 0:
            tmp = x
            x = y
            y = tmp
        elif place == 1:
            y = y + n
        elif place == 2:
            x += n
            y += n
        elif place == 3:
            tmp = x
            x = 2*n - 1 - y
            y = n - 1 - tmp

        hindex = hindex >> 2

    return_arr[0] = x
    return_arr[1] = y

    return


@annotate(intp='hilbert_arr, return_arr', p='int')
def _make_hilbert_curve(hilbert_arr, p, return_arr):
    a, i, x, y = declare('int', 4)
    a = 2**p
    for i in range(a*a):
        _hindex2xy(i, p, return_arr)
        x = return_arr[0]
        y = return_arr[1]
        hilbert_arr[a*x + y] = i

    return


@annotate(
    intp='hilbert_arr, h_indices, new_indices', int='p, org_points_end',
    org_points='doublep')
def _sort_along_hilbert_curve(
        org_points, hilbert_arr, p, h_indices, new_indices, org_points_end):
    '''
     org_points : 2N x 1; points to be sorted
         points : 2N x 1; array with dimensions same as that of points
                  used for manipulation
      hilbert_x : 2^8 x 1 array
      hilbert_y : 2^8 x 1 array
    new_indices : N x 1 array
    '''

    min_x, min_y, max_x, max_y, x, y, max_xy = declare('double', 7)
    i, a, x_, y_ = declare('int', 4)

    min_x = org_points[2*0 + 0]
    min_y = org_points[2*0 + 1]
    max_x = org_points[2*0 + 0]
    max_y = org_points[2*0 + 1]

    for i in range(org_points_end):
        x = org_points[2*i + 0]
        y = org_points[2*i + 1]
        if x < min_x:
            min_x = x
        elif max_x < x:
            max_x = x
        if y < min_y:
            min_y = y
        elif max_y < y:
            max_y = y

    max_x -= min_x
    max_y -= min_y

    if max_y > max_x:
        max_xy = max_y
    else:
        max_xy = max_x

    if max_xy == 0.0:
        # occurs when there is only one point in the round
        max_xy = 1.0

    a = 2**p

    for i in range(org_points_end):
        x_ = _round_(((org_points[2*i + 0] - min_x)/max_xy)*(a - 1))
        y_ = _round_(((org_points[2*i + 1] - min_y)/max_xy)*(a - 1))
        h_indices[i] = hilbert_arr[a*x_ + y_]
        new_indices[i] = i

    if org_points_end > 1:
        qargsort(h_indices, new_indices, 0, org_points_end)

    return


@annotate(
    doublep='points, org_points', int='p, num_rounds', intp='rounds,' + \
    ' boundary_indices, points_left_old, points_left_new, hilbert_arr,' + \
    ' new_indices, insertion_seq, return_arr, h_indices')
def _final_assembly(
        points, rounds, boundary_indices, points_left_old, points_left_new,
        hilbert_arr, org_points, h_indices, new_indices, p, insertion_seq,
        return_arr, num_rounds):

    i, j, org_points_end, org_points_idx = declare('int', 4)

    _make_hilbert_curve(hilbert_arr, p, return_arr)

    for i in range(num_rounds):
        org_points_end = 0

        for j in range(boundary_indices[i], boundary_indices[i+1]):
            org_points[2*org_points_end + 0] = points[2*rounds[j] + 0]
            org_points[2*org_points_end + 1] = points[2*rounds[j] + 1]
            org_points_end += 1

        _sort_along_hilbert_curve(
            org_points, hilbert_arr, p, h_indices, new_indices, org_points_end)

        for j in range(boundary_indices[i+1]-1, boundary_indices[i]-1, -1):
            org_points_idx = new_indices[org_points_end-1]
            insertion_seq[j] = rounds[boundary_indices[i]+org_points_idx]
            org_points_end -= 1

    return
final_assembly = Cython(_final_assembly)


def make_BRIO(points):

    len_points = int(len(points)/2)
    num_rounds = int(np.round(np.log2(len_points), 0))

    rounds = np.empty(len_points, dtype=np.int32)
    points_left_old = np.arange(len_points, dtype=np.int32)
    np.random.shuffle(points_left_old)
    points_left_new = np.empty(len_points, dtype=np.int32)
    boundary_indices = np.empty(num_rounds+1, dtype=np.int32)

    make_rounds(
        rounds, boundary_indices, points_left_old, points_left_new, len_points)

    max_number_of_points_in_a_round = np.max(
        boundary_indices[1:] - boundary_indices[0:-1])

    rho = 5  # number of points per cell
    if len_points <= int(rho*(2**(2*4))):
        p = 4
    else:
        p = int(np.ceil(0.5*np.log2(len_points/rho)))

    hilbert_arr = np.empty(shape=2**(2*p), dtype=np.int32)
    org_points = np.empty(
        shape=2*max_number_of_points_in_a_round, dtype=np.float64)
    h_indices = np.empty(max_number_of_points_in_a_round, dtype=np.int32)
    new_indices = np.empty(max_number_of_points_in_a_round, dtype=np.int32)
    insertion_seq = np.empty(len_points, dtype=np.int32)
    return_arr = np.empty(shape=2, dtype=np.int32)

    final_assembly(
        points, rounds, boundary_indices,points_left_old, points_left_new,
        hilbert_arr, org_points, h_indices, new_indices, p, insertion_seq,
        return_arr, num_rounds)
    new_points = points.copy()[insertion_seq]

    return insertion_seq, new_points


def perf(N):

    import time

    points = np.random.rand(2*N)

    temp = np.random.rand(2*10)
    temp_BRIO = make_BRIO(temp)

    for i in range(5):
        start = time.time()
        ins_seq, points_new = make_BRIO(points)
        end = time.time()
        if i == 0:
            running_time = end-start
        else:
            running_time = min(running_time, end-start)
        print("running_time : {} s.\n".format(running_time))

    print("BRIO made for {} points.".format(N))
    print("minimum time taken : {} s".format(running_time))


@annotate(intp='points, indices', num_points='int')
def _check_quicksort(points, indices, num_points):
    start = declare('int')
    start = 0
    qargsort2(points, indices, start, num_points)
    return
check_quicksort = Cython(_check_quicksort)


@annotate(int='num, return_')
def _do_nothing(num):
    if num == 0:
        return 0
    else:
        num = _do_nothing(num - 1)
    return num
do_nothing = Cython(_do_nothing)


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1])
    perf(N)
    # points = np.arange(10, dtype=np.int32)
    # np.random.shuffle(points)
    # indices = np.arange(10, dtype=np.int32)
    # num_points = np.int32(10)
    # print('                 points : {}'.format(points))
    # print('                indices : {}'.format(indices))
    # check_quicksort(points, indices, num_points)
    # print('indices of sorted array : {}'.format(indices))
    # print('          sorted points : {}'.format(points[indices]))
    # a = do_nothing(np.int32(10))
    # print(a)
