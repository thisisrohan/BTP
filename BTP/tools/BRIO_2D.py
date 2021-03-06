import numpy as np
from numba import njit


@njit
def make_rounds(
    rounds,
    boundary_indices,
    points_left_old,
    points_left_new
):
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

    len_points = len(rounds)
    num_rounds = int(np.round(np.log2(len_points), 0))

    for i in range(len_points):
        points_left_old[i] = i
    points_left_old_end = len_points
    points_left_new_end = 0

    boundary_indices_insertion_idx = num_rounds
    rounds_insertion_idx = len_points-1
    round_idx = 1

    boundary_indices[boundary_indices_insertion_idx] = rounds_insertion_idx+1
    boundary_indices_insertion_idx -= 1

    while round_idx < num_rounds:

        points_left_new_end = 0

        for i in range(points_left_old_end):
            if np.random.randint(low=0, high=2) == 1:
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
        ] = rounds_insertion_idx+1
        boundary_indices_insertion_idx -= 1

        round_idx += 1

    for i in range(points_left_old_end):
        rounds[rounds_insertion_idx] = points_left_old[i]
        rounds_insertion_idx -= 1

    boundary_indices[boundary_indices_insertion_idx] = rounds_insertion_idx+1
    boundary_indices_insertion_idx -= 1

    return rounds, boundary_indices


@njit
def hindex2xy(hindex, p):
    '''
    hindex : Hilbert index of the point
         p : Iteration of the Hilbert curve
    '''

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
            x, y = y, x
        elif place == 1:
            y = y + n
        elif place == 2:
            x += n
            y += n
        elif place == 3:
            x, y = (2*n-1) - y, (n-1) - x

        hindex = hindex >> 2

    x = int(x)
    y = int(y)

    return x, y


@njit
def make_hilbert_curve(
    hilbert_arr,
    p
):
    a = int(2**p)
    for i in range(2**(2*p)):
        x, y = hindex2xy(i, p)
        hilbert_arr[x+a*y] = i

    return


@njit
def sort_along_hilbert_curve(
    org_points,
    temp_points,
    hilbert_arr,
    new_indices,
    org_points_end,
    p
):
    '''
     org_points : 2N x 1; points to be sorted
         points : 2N x 1; array with dimensions same as that of points
                  used for manipulation
      hilbert_x : 2^8 x 1 array
      hilbert_y : 2^8 x 1 array
    new_indices : N x 1 array
    '''

    len_points = org_points_end

    min_x = org_points[0]
    min_y = org_points[1]
    max_x = org_points[0]
    max_y = org_points[1]

    for i in range(len_points):
        x = org_points[2*i]
        y = org_points[2*i+1]
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

    if max_xy == 0:
        # occurs when there is only one point in the round
        max_xy = 1

    a = 2**p

    for i in range(len_points):
        x = org_points[2*i]
        y = org_points[2*i+1]

        x = np.round(((x-min_x)/max_xy)*(a-1), 0)
        x = np.round(((y-min_y)/max_xy)*(a-1), 0)

        new_indices[i] = hilbert_arr[int(x+a*y)]

    new_indices[0:len_points] = np.argsort(new_indices[0:len_points])

    for i in range(len_points):
        idx = new_indices[i]
        temp_points[2*i] = org_points[2*idx]
        temp_points[2*i+1] = org_points[2*idx+1]

    return


@njit
def final_assembly(
    points,
    rounds,
    boundary_indices,
    points_left_old,
    points_left_new,
    hilbert_arr,
    org_points,
    temp_points,
    new_indices,
    p
):
    points[0::2] = points[0::2][rounds]
    points[1::2] = points[1::2][rounds]

    make_hilbert_curve(hilbert_arr, p)

    for i in range(len(boundary_indices)-1):
        org_points_end = 0

        for j in range(boundary_indices[i], boundary_indices[i+1]):
            org_points[2*org_points_end] = points[2*j]
            org_points[2*org_points_end+1] = points[2*j+1]
            org_points_end += 1

        sort_along_hilbert_curve(org_points, temp_points, hilbert_arr,
                                 new_indices, org_points_end, p)

        for j in range(boundary_indices[i+1]-1, boundary_indices[i]-1, -1):
            points[2*j] = temp_points[2*(org_points_end-1)]
            points[2*j+1] = temp_points[2*(org_points_end-1)+1]
            org_points_end -= 1

    return


def make_BRIO(points):

    len_points = int(0.5*len(points))
    num_rounds = int(np.round(np.log2(len_points), 0))

    rounds = np.arange(len_points, dtype=np.int64)
    points_left_old = np.empty(len_points, dtype=np.int64)
    points_left_new = np.empty(len_points, dtype=np.int64)
    boundary_indices = np.empty(num_rounds+1, dtype=np.int64)

    rounds, boundary_indices = make_rounds(rounds, boundary_indices,
                                           points_left_old, points_left_new)

    max_number_of_points_in_a_round = np.max(
        boundary_indices[1:] - boundary_indices[0:-1]
    )

    if max_number_of_points_in_a_round <= 2500:
        p = 4
    else:
        p = int(np.round(0.5*np.log2(max_number_of_points_in_a_round*0.1), 0))

    hilbert_arr = np.empty(2**(2*p), dtype=np.int64)
    org_points = np.empty(2*len_points, dtype=np.float64)
    temp_points = np.empty(2*len_points, dtype=np.float64)
    new_indices = np.empty(len_points, dtype=np.int64)
    new_points = points.copy()

    final_assembly(new_points, rounds, boundary_indices,
                   points_left_old, points_left_new, hilbert_arr,
                   org_points, temp_points, new_indices, p)

    return new_points


def perf(N):

    import time

    points = np.random.rand(2*N)

    temp = np.random.rand(20)
    temp_BRIO = make_BRIO(temp)

    for i in range(5):
        start = time.time()
        points_new = make_BRIO(points)
        end = time.time()
        if i == 0:
            running_time = end-start
        else:
            running_time = min(running_time, end-start)

    print("BRIO made for {} points.".format(N))
    print("time taken : {} s".format(running_time))


if __name__ == "__main__":

    import sys

    N = int(sys.argv[1])

    perf(N)
