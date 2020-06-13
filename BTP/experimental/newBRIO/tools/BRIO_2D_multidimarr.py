import numpy as np
from numba import njit


@njit
def make_rounds(
    rounds,
    boundary_indices,
    points_left_old,
    points_left_new,
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

    num_points = len(rounds)
    num_rounds = int(np.round(np.log2(num_points), 0))

    points_left_old_end = num_points
    points_left_new_end = 0

    boundary_indices_insertion_idx = num_rounds
    rounds_insertion_idx = num_points-1
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

    return

@njit
def make_rounds_2(rounds, boundary_indices, num_points, num_rounds, indices):
    bi_insertion_idx = num_rounds
    rounds_insertion_idx = num_points - 1

    boundary_indices[bi_insertion_idx] = rounds_insertion_idx + 1
    bi_insertion_idx -= 1

    points_left = num_points
    last_idx = num_points - 1
    for round_idx in range(1, num_rounds):
        points_added = 0
        for i in range(points_left):
            j = np.random.randint(low=0, high=last_idx + 1)
            flip = np.random.randint(low=0, high=2)
            if flip == 1:
                # update rounds
                rounds[rounds_insertion_idx] = indices[j]
                # swap j and last_idx in indices
                temp = indices[j]
                indices[j] = indices[last_idx]
                indices[last_idx] = temp
                # update points_added, last_idx and rounds_insertion_idx
                points_added += 1
                last_idx -= 1
                rounds_insertion_idx -= 1
        points_left -= points_added
        # update boundary_indices
        boundary_indices[bi_insertion_idx] = rounds_insertion_idx + 1
        bi_insertion_idx -= 1

    for i in range(last_idx + 1):
        rounds[rounds_insertion_idx] = indices[i]
        rounds_insertion_idx -= 1

    boundary_indices[bi_insertion_idx] = rounds_insertion_idx + 1
    bi_insertion_idx -= 1

    return


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
        hilbert_arr[x, y] = i

    return


@njit
def sort_along_hilbert_curve(
    org_points,
    hilbert_arr,
    p,
    new_indices,
    org_points_end,
):
    '''
     org_points : 2N x 1; points to be sorted
         points : 2N x 1; array with dimensions same as that of points
                  used for manipulation
      hilbert_x : 2^8 x 1 array
      hilbert_y : 2^8 x 1 array
    new_indices : N x 1 array
    '''

    org_points_end

    min_x = org_points[0, 0]
    min_y = org_points[0, 1]
    max_x = org_points[0, 0]
    max_y = org_points[0, 1]

    for i in range(org_points_end):
        x = org_points[i, 0]
        y = org_points[i, 1]
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

    for i in range(org_points_end):
        x = org_points[i, 0]
        y = org_points[i, 1]

        x = np.round(((x-min_x)/max_xy)*(a-1), 0)
        y = np.round(((y-min_y)/max_xy)*(a-1), 0)

        new_indices[i] = hilbert_arr[int(x), int(y)]

    new_indices[0:org_points_end] = np.argsort(new_indices[0:org_points_end])

    return


@njit
def final_assembly(
    points,
    rounds,
    boundary_indices,
    hilbert_arr,
    org_points,
    new_indices,
    p,
    insertion_seq
):

    make_hilbert_curve(hilbert_arr, p)

    for i in range(len(boundary_indices)-1):
        org_points_end = 0

        for j in range(boundary_indices[i], boundary_indices[i+1]):
            org_points[org_points_end, 0] = points[rounds[j], 0]
            org_points[org_points_end, 1] = points[rounds[j], 1]
            org_points_end += 1

        sort_along_hilbert_curve(org_points, hilbert_arr, p,
                                 new_indices, org_points_end)

        for j in range(boundary_indices[i+1]-1, boundary_indices[i]-1, -1):
            org_points_idx = new_indices[org_points_end-1]
            insertion_seq[j] = rounds[boundary_indices[i]+org_points_idx]
            org_points_end -= 1

    return


def make_BRIO(points):

    num_points = int(len(points))
    num_rounds = int(np.round(np.log2(num_points), 0))

    rounds = np.empty(num_points, dtype=np.int64)
    points_left_old = np.arange(num_points, dtype=np.int64)
    # np.random.shuffle(points_left_old)
    # points_left_new = np.empty(num_points, dtype=np.int64)
    boundary_indices = np.empty(num_rounds+1, dtype=np.int64)

    # make_rounds(rounds, boundary_indices, points_left_old, points_left_new)
    make_rounds_2(rounds, boundary_indices, num_points, num_rounds, points_left_old)

    max_number_of_points_in_a_round = np.max(
        boundary_indices[1:] - boundary_indices[0:-1]
    )

    rho = 5  # number of points per cell
    if num_points <= int(rho*(2**(2*4))):
        p = 4
    else:
        p = int(np.ceil(0.5*np.log2(num_points/rho)))

    # p = refine_p(p, points, rho)

    hilbert_arr = np.empty(shape=(2**p, 2**p), dtype=np.int64)
    org_points = np.empty(
        shape=(max_number_of_points_in_a_round, 2), dtype=np.float64)
    new_indices = np.empty(max_number_of_points_in_a_round, dtype=np.int64)
    insertion_seq = np.empty(num_points, dtype=np.int64)

    final_assembly(points, rounds, boundary_indices,hilbert_arr,
                   org_points, new_indices, p, insertion_seq)
    new_points = points.copy()[insertion_seq]

    return insertion_seq, new_points


def perf(N):

    import time

    points = np.random.rand(N, 2)

    temp = np.random.rand(10, 2)
    temp_BRIO = make_BRIO(temp)

    for i in range(5):
        start = time.time()
        ins_seq, points_new = make_BRIO(points)
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