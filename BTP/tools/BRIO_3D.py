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
    num_rounds = int(
        np.round(
            np.log2(len_points),
            0
        )
    )

    points_left_new_end = 0
    # points_left_old_end = 0
    # for i in range(len_points):
    #     points_left_old[i] = i
    points_left_old_end = len_points

    boundary_indices_insertion_idx = num_rounds
    rounds_insertion_idx = len_points-1
    round_idx = 1

    boundary_indices[boundary_indices_insertion_idx] = len_points
    boundary_indices_insertion_idx -= 1

    while round_idx < num_rounds:

        points_left_new_end = 0

        for i in range(points_left_old_end-1, -1, -1):
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
            boundary_indices_insertion_idx] = rounds_insertion_idx+1
        boundary_indices_insertion_idx -= 1

        round_idx += 1

    for i in range(points_left_old_end):
        rounds[rounds_insertion_idx] = points_left_old[i]
        rounds_insertion_idx -= 1

    boundary_indices[boundary_indices_insertion_idx] = rounds_insertion_idx+1


@njit
def hindex2xyz(hindex, p):
    '''
    hindex : Hilbert index of the point
         p : Iteration of the Hilbert curve
    '''

    place = hindex & 7

    if place == 0:
        x = 0
        y = 0
        z = 0
    elif place == 1:
        x = 0
        y = 1
        z = 0
    elif place == 2:
        x = 1
        y = 1
        z = 0
    elif place == 3:
        x = 1
        y = 0
        z = 0
    elif place == 4:
        x = 1
        y = 0
        z = 1
    elif place == 5:
        x = 1
        y = 1
        z = 1
    elif place == 6:
        x = 0
        y = 1
        z = 1
    elif place == 7:
        x = 0
        y = 0
        z = 1

    for i in range(1, p):

        n = 2**i
        hindex = hindex >> 3
        place = hindex & 7

        if place == 0:
            temp = x
            x = y
            y = z
            z = temp
        elif place == 1:
            temp = x
            x = z
            z = y
            y = temp + n
        elif place == 2:
            temp = x
            x = z + n
            z = y
            y = temp + n
        elif place == 3:
            x = -x + (2*n-1)
            y = -y + (n-1)
        elif place == 4:
            x = -x + (2*n-1)
            y = -y + (n-1)
            z = z + n
        elif place == 5:
            temp = x
            x = -z + (2*n-1)
            z = -y + (2*n-1)
            y = temp + n
        elif place == 6:
            temp = x
            x = -z + (n-1)
            z = -y + (2*n-1)
            y = temp + n
        elif place == 7:
            temp = x
            x = y
            y = -z + (n-1)
            z = -temp + (2*n-1)

    x = int(x)
    y = int(y)
    z = int(z)

    return x, y, z


@njit
def make_hilbert_curve(
    hilbert_arr,
    p
):
    for i in range(2**(3*p)):
        x, y, z = hindex2xyz(i, p)
        hilbert_arr[x, y, z] = i


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

    min_x = org_points[0]
    min_y = org_points[1]
    min_z = org_points[2]

    max_x = org_points[0]
    max_y = org_points[1]
    max_z = org_points[2]

    for i in range(org_points_end):
        x = org_points[3*i+0]
        y = org_points[3*i+1]
        z = org_points[3*i+2]

        if x < min_x:
            min_x = x
        elif max_x < x:
            max_x = x

        if y < min_y:
            min_y = y
        elif max_y < y:
            max_y = y

        if z < min_z:
            min_z = z
        elif max_z < z:
            max_z = z

    max_x -= min_x
    max_y -= min_y
    max_z -= min_z

    max_xyz = max(max_x, max_y, max_z)

    if max_xyz == 0:
        max_xyz = 1

    temp = 2**p

    for i in range(org_points_end):

        x = org_points[3*i+0]
        y = org_points[3*i+1]
        z = org_points[3*i+2]

        x = np.round((temp-1)*((x-min_x)/max_xyz), 0)
        y = np.round((temp-1)*((y-min_y)/max_xyz), 0)
        z = np.round((temp-1)*((z-min_z)/max_xyz), 0)

        new_indices[i] = hilbert_arr[int(x), int(y), int(z)]

    new_indices[0:org_points_end] = np.argsort(new_indices[0:org_points_end])

    for i in range(org_points_end):
        idx = new_indices[i]
        temp_points[3*i+0] = org_points[3*idx+0]
        temp_points[3*i+1] = org_points[3*idx+1]
        temp_points[3*i+2] = org_points[3*idx+2]


@njit
def final_assembly(
    new_points,
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
    new_points[0::3] = new_points[0::3][rounds]
    new_points[1::3] = new_points[1::3][rounds]
    new_points[2::3] = new_points[2::3][rounds]

    make_hilbert_curve(
        hilbert_arr,
        p
    )

    for i in range(len(boundary_indices)-1):
        org_points_end = 0
        for j in range(boundary_indices[i], boundary_indices[i+1]):
            org_points[3*org_points_end+0] = new_points[3*j+0]
            org_points[3*org_points_end+1] = new_points[3*j+1]
            org_points[3*org_points_end+2] = new_points[3*j+2]
            org_points_end += 1

        sort_along_hilbert_curve(org_points, temp_points, hilbert_arr,
                                 new_indices, org_points_end, p)

        for j in range(boundary_indices[i+1]-1, boundary_indices[i]-1, -1):
            new_points[3*j+0] = temp_points[3*(org_points_end-1)+0]
            new_points[3*j+1] = temp_points[3*(org_points_end-1)+1]
            new_points[3*j+2] = temp_points[3*(org_points_end-1)+2]
            org_points_end -= 1


def make_BRIO(points):

    len_points = int(len(points)/3)
    num_rounds = int(
        np.round(
            np.log2(len_points),
            0
        )
    )

    rounds = np.arange(len_points, dtype=np.int64)
    points_left_old = np.arange(len_points, dtype=np.int64)
    points_left_new = np.zeros(len_points, dtype=np.int64)
    boundary_indices = np.zeros(num_rounds+1, dtype=np.int64)

    make_rounds(
        rounds,
        boundary_indices,
        points_left_old,
        points_left_new
    )

    max_number_of_points_in_a_round = np.max(
        boundary_indices[1:]-boundary_indices[0:-1])

    if max_number_of_points_in_a_round <= 2500:
        p = 3
    else:
        p = int(np.ceil((1/3)*np.log2(max_number_of_points_in_a_round*0.2)))

    a = 2**p
    hilbert_arr = np.empty(shape=(a, a, a), dtype=np.int64)
    org_points = np.empty(3*max_number_of_points_in_a_round, dtype=np.float64)
    temp_points = np.empty(3*max_number_of_points_in_a_round, dtype=np.float64)
    new_indices = np.empty(len_points, dtype=np.int64)
    new_points = points.copy()

    final_assembly(new_points, rounds, boundary_indices,
                   points_left_old, points_left_new, hilbert_arr,
                   org_points, temp_points, new_indices, p)

    return new_points


def perf(N):

    import time

    points = np.random.rand(3*N)

    temp = np.random.rand(30)
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
