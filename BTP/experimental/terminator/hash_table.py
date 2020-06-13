import numpy as np
from numba import njit


@njit
def initialize_seg_ht(seg_ht_cap, seg_ht_arr, segments, num_segs):

    for i in range(num_segs):
        v1 = segments[i, 0]
        v2 = segments[i, 1]
        key = (v1 + 1)*(v2 + 1)
        h_index = key % seg_ht_cap
        while True:
            if seg_ht_arr[h_index, 0] < 0:
                break
            else:
                h_index += 1
                if h_index <= seg_ht_cap:
                    h_index -= seg_ht_cap
        seg_ht_arr[h_index, 0] = key
        seg_ht_arr[h_index, 1] = i
    return


@njit
def get_seg_idx(v1, v2, seg_ht_cap, seg_ht_arr, segments):

    key = (v1 + 1)*(v2 + 1)
    h_index = key % seg_ht_cap
    h_index_org = h_index
    while True:
        if seg_ht_arr[h_index, 0] == -1:
            seg_idx  = -1
            break
        elif seg_ht_arr[h_index, 0] == key:
            seg_idx = seg_ht_arr[h_index, 1]
            v1_s = segments[seg_idx, 0]
            v2_s = segments[seg_idx, 1]
            if (v1_s == v1 and v2_s == v2) or (v2_s == v1 and v1_s == v2):
                break
            else:
                h_index += 1
                if h_index >= seg_ht_cap:
                    h_index -= seg_ht_cap
                if h_index == h_index_org:
                    seg_idx = -1
                    break
        else:
            h_index += 1
            if h_index >= seg_ht_cap:
                h_index -= seg_ht_cap
            if h_index == h_index_org:
                seg_idx = -1
                break

    return seg_idx


@njit
def delete_entry_seg_ht(v1, v2, seg_idx, seg_ht_cap, seg_ht_arr):

    key = (v1 + 1)*(v2 + 1)
    h_index = key % seg_ht_cap
    while True:
        if seg_ht_arr[h_index, 0] == key and seg_ht_arr[h_index, 1] == seg_idx:
            seg_ht_arr[h_index, 0] = -2
            seg_ht_arr[h_index, 1] = -2
            break
        else:
            h_index += 1
            if h_index >= seg_ht_cap:
                h_index -= seg_ht_cap
    return


@njit
def add_entry_seg_ht(v1, v2, seg_idx, seg_ht_cap, seg_ht_arr):

    key = (v1 + 1)*(v2 + 1)
    h_index = key % seg_ht_cap
    while True:
        if seg_ht_arr[h_index, 0] < 0:
            break
        else:
            h_index += 1
            if h_index >= seg_ht_cap:
                h_index -= seg_ht_cap
    seg_ht_arr[h_index, 0] = key
    seg_ht_arr[h_index, 1] = seg_idx
    return


@njit
def initialize_tri_ht(tri_ht_cap, tri_ht_arr, vertices_ID, num_tri):

    for i in range(num_tri):
        v1 = vertices_ID[i, 0]
        v2 = vertices_ID[i, 1]
        v3 = vertices_ID[i, 2]
        key1 = (v1 + 1)*(v1 + 1) + (v2 + 1)
        key2 = (v2 + 1)*(v2 + 1) + (v3 + 1)
        key3 = (v3 + 1)*(v3 + 1) + (v1 + 1)
        h_index1 = key1 % tri_ht_cap
        h_index2 = key2 % tri_ht_cap
        h_index3 = key3 % tri_ht_cap

        while True:
            if tri_ht_arr[h_index1, 0] < 0:
                break
            else:
                h_index1 += 1
                if h_index1 <= tri_ht_cap:
                    h_index1 -= tri_ht_cap
        tri_ht_arr[h_index1, 0] = key1
        tri_ht_arr[h_index1, 1] = i

        while True:
            if tri_ht_arr[h_index2, 2] < 0:
                break
            else:
                h_index2 += 1
                if h_index2 <= tri_ht_cap:
                    h_index2 -= tri_ht_cap
        tri_ht_arr[h_index2, 2] = key1
        tri_ht_arr[h_index2, 3] = i

        while True:
            if tri_ht_arr[h_index3, 4] < 0:
                break
            else:
                h_index3 += 1
                if h_index3 <= tri_ht_cap:
                    h_index3 -= tri_ht_cap
        tri_ht_arr[h_index3, 4] = key1
        tri_ht_arr[h_index3, 5] = i

    return


@njit
def get_tri_idx(v1, v2, tri_ht_cap, tri_ht_arr, vertices_ID):

    key = (v1 + 1)*(v1 + 1) + (v2 + 1)

    # searching in tri_ht_arr1
    h_index = key % tri_ht_cap
    h_index_org = h_index
    while True:
        if tri_ht_arr[h_index, 0] == -1:
            tri_idx  = -1
            j = -1
            break
        elif tri_ht_arr[h_index, 0] == key:
            tri_idx = tri_ht_arr[h_index, 1]
            v1_t = vertices_ID[tri_idx, 0]
            v2_t = vertices_ID[tri_idx, 1]
            if v1_t == v1 and v2_t == v2:
                j = 2
                break
            else:
                h_index += 1
                if h_index >= tri_ht_cap:
                    h_index -= tri_ht_cap
                if h_index == h_index_org:
                    tri_idx = -1
                    j = -1
                    break
        else:
            h_index += 1
            if h_index >= tri_ht_cap:
                h_index -= tri_ht_cap
            if h_index == h_index_org:
                tri_idx = -1
                j = -1
                break

    if tri_idx == -1:
        # searching in tri_ht_arr2
        h_index = key % tri_ht_cap
        h_index_org = h_index
        while True:
            if tri_ht_arr[h_index, 2] == -1:
                tri_idx  = -1
                j = -1
                break
            elif tri_ht_arr[h_index, 2] == key:
                tri_idx = tri_ht_arr[h_index, 3]
                v1_t = vertices_ID[tri_idx, 1]
                v2_t = vertices_ID[tri_idx, 2]
                if v1_t == v1 and v2_t == v2:
                    j = 0
                    break
                else:
                    h_index += 1
                    if h_index >= tri_ht_cap:
                        h_index -= tri_ht_cap
                    if h_index == h_index_org:
                        tri_idx = -1
                        j = -1
                        break
            else:
                h_index += 1
                if h_index >= tri_ht_cap:
                    h_index -= tri_ht_cap
                if h_index == h_index_org:
                    tri_idx = -1
                    j = -1
                    break

    if tri_idx == -1:
        # searching in tri_ht_arr3
        h_index = key % tri_ht_cap
        h_index_org = h_index
        while True:
            if tri_ht_arr[h_index, 4] == -1:
                tri_idx  = -1
                j = -1
                break
            elif tri_ht_arr[h_index, 4] == key:
                tri_idx = tri_ht_arr[h_index, 5s]
                v1_t = vertices_ID[tri_idx, 2]
                v2_t = vertices_ID[tri_idx, 0]
                if v1_t == v1 and v2_t == v2:
                    j = 1
                    break
                else:
                    h_index += 1
                    if h_index >= tri_ht_cap:
                        h_index -= tri_ht_cap
                    if h_index == h_index_org:
                        tri_idx = -1
                        j = -1
                        break
            else:
                h_index += 1
                if h_index >= tri_ht_cap:
                    h_index -= tri_ht_cap
                if h_index == h_index_org:
                    tri_idx = -1
                    j = -1
                    break

    return tri_idx, j


@njit
def delete_entry_tri_ht(v1, v2, v3, t_index, tri_ht_cap, tri_ht_arr):

    key1 = (v1 + 1)*(v1 + 1) + (v2 + 1)
    key2 = (v2 + 1)*(v2 + 1) + (v3 + 1)
    key3 = (v3 + 1)*(v3 + 1) + (v1 + 1)
    h_index1 = key1 % tri_ht_cap
    h_index2 = key2 % tri_ht_cap
    h_index3 = key3 % tri_ht_cap

    # deleting in tri_ht_arr1
    while True:
        if tri_ht_arr[h_index1, 0] == key1 and \
                tri_ht_arr[h_index1, 1] == tri_idx:
            tri_ht_arr[h_index1, 0] = -2
            tri_ht_arr[h_index1, 1] = -2
            break
        else:
            h_index1 += 1
            if h_index1 >= tri_ht_cap:
                h_index1 -= tri_ht_cap

    # deleting in tri_ht_arr2
    while True:
        if tri_ht_arr[h_index2, 2] == key2 and \
                tri_ht_arr[h_index2, 3] == tri_idx:
            tri_ht_arr[h_index2, 2] = -2
            tri_ht_arr[h_index2, 3] = -2
            break
        else:
            h_index2 += 1
            if h_index2 >= tri_ht_cap:
                h_index2 -= tri_ht_cap

    # deleting in tri_ht_arr3
    while True:
        if tri_ht_arr[h_index3, 4] == key3 and \
                tri_ht_arr[h_index3, 5] == tri_idx:
            tri_ht_arr[h_index3, 4] = -2
            tri_ht_arr[h_index3, 5] = -2
            break
        else:
            h_index3 += 1
            if h_index3 >= tri_ht_cap:
                h_index3 -= tri_ht_cap

    return


@njit
def add_entry_tri_ht(v1, v2, v3, tri_idx, tri_ht_cap, tri_ht_arr):

    key1 = (v1 + 1)*(v1 + 1) + (v2 + 1)
    key2 = (v2 + 1)*(v2 + 1) + (v3 + 1)
    key3 = (v3 + 1)*(v3 + 1) + (v1 + 1)
    h_index1 = key1 % tri_ht_cap
    h_index2 = key2 % tri_ht_cap
    h_index3 = key3 % tri_ht_cap

    # adding in tri_ht_arr1
    while True:
        if tri_ht_arr[h_index1, 0] < 0:
            break
        else:
            h_index1 += 1
            if h_index1 >= tri_ht_cap:
                h_index1 -= tri_ht_cap
    tri_ht_arr[h_index1, 0] = key1
    tri_ht_arr[h_index1, 1] = tri_idx

    # adding in tri_ht_arr2
    while True:
        if tri_ht_arr[h_index2, 2] < 0:
            break
        else:
            h_index2 += 1
            if h_index2 >= tri_ht_cap:
                h_index2 -= tri_ht_cap
    tri_ht_arr[h_index2, 2] = key2
    tri_ht_arr[h_index2, 3] = tri_idx

    # adding in tri_ht_arr3
    while True:
        if tri_ht_arr[h_index3, 4] < 0:
            break
        else:
            h_index3 += 1
            if h_index3 >= tri_ht_cap:
                h_index3 -= tri_ht_cap
    tri_ht_arr[h_index3, 4] = key3
    tri_ht_arr[h_index3, 5] = tri_idx

    return


@njit
def delete_bad_tri_from_tri_ht(
        bad_tri, bt_end, vertices_ID, tri_ht_arr, tri_ht_cap):

    for i in range(bt_end):
        t = bad_tri[i]
        v1 = vertices_ID[t, 0]
        v2 = vertices_ID[t, 1]
        v3 = vertices_ID[t, 2]
        delete_entry_tri_ht(v1, v2, v3, t, tri_ht_cap, tri_ht_arr)

    return


@njit
def add_new_tri_to_tri_ht(
        bad_tri, bt_end, boundary_end, num_tri, vertices_ID, tri_ht_arr,
        tri_ht_cap):

    if num_tri >= tri_ht_cap:
        tri_ht_cap *= 2
        tri_ht_arr = np.empty(shape=(tri_ht_cap, 6), dtype=np.int64)
        initialize_tri_ht(
            tri_ht_cap, tri_ht_arr, vertices_ID, num_tri)
    else:
        for i in range(boundary_end):
            if i < bt_end:
                t = bad_tri[i]
            else:
                t = num_tri - (boundary_end - i)
            v1 = vertices_ID[t, 0]
            v2 = vertices_ID[t, 1]
            v3 = vertices_ID[t, 2]
            add_entry_tri_ht(
                v1, v2, v3, t, tri_ht_cap, tri_ht_arr)

    return tri_ht_cap, tri_ht_arr


@njit
def dequeue_st(split_tri, st_params):
    q_head = st_params[0]
    q_tail = st_params[1]
    q_cap = st_params[2]

    val_arr = split_tri[q_head, :]
    q_head += 1
    if q_head >= q_cap:
        q_head -= q_cap

    if q_head == q_tail:
        q_num_items = 0
    elif q_tail > q_head:
        q_num_items = q_tail - q_head
    else:
        q_num_items = (q_cap - q_head) + q_tail

    st_params[0] = q_head
    st_params[3] = q_num_items

    return val_arr


@njit
def dequeue_ss(split_segs, ss_params):
    q_head = ss_params[0]
    q_tail = ss_params[1]
    q_cap = ss_params[2]

    val_arr = split_segs[q_head, :]
    q_head += 1
    if q_head >= q_cap:
        q_head -= q_cap

    if q_head == q_tail:
        q_num_items = 0
    elif q_tail > q_head:
        q_num_items = q_tail - q_head
    else:
        q_num_items = (q_cap - q_head) + q_tail

    ss_params[0] = q_head
    ss_params[3] = q_num_items

    return val_arr


@njit
def enqueue_st(split_tri, st_params, t, a, b, c):
    q_head = st_params[0]
    q_tail = st_params[1]
    q_cap = st_params[2]
    q_num_items = st_params[3]

    if q_num_items == q_cap:
        temp_q = np.empty(shape=(2*q_cap, 4), dtype=np.int64)
        for i in range(q_cap):
            for j in range(4):
                temp_q[i, j] = split_tri[q_head, j]
            q_head += 1
            if q_head >= q_cap:
                q_head -= q_cap
        split_tri = temp_q
        q_head = 0
        q_tail = q_cap
        q_cap *= 2
        st_params[0] = q_head
        st_params[2] = q_cap

    split_tri[q_tail, 0] = t
    split_tri[q_tail, 1] = a
    split_tri[q_tail, 2] = b
    split_tri[q_tail, 3] = c

    q_tail += 1
    if q_tail >= q_cap:
        q_tail -= q_cap

    if q_head == q_tail:
        q_num_items = q_cap
    elif q_tail > q_head:
        q_num_items = q_tail - q_head
    else:
        q_num_items = (q_cap - q_head) + q_tail

    st_params[1] = q_tail
    st_params[3] = q_num_items

    return split_tri


@njit
def enqueue_ss(split_segs, ss_params, seg_idx, v1, v2):
    q_head = ss_params[0]
    q_tail = ss_params[1]
    q_cap = ss_params[2]
    q_num_items = ss_params[3]

    if q_num_items == q_cap:
        temp_q = np.empty(shape=(2*q_cap, 3), dtype=np.int64)
        for i in range(q_cap):
            for j in range(3):
                temp_q[i, j] = split_segs[q_head, j]
            q_head += 1
            if q_head >= q_cap:
                q_head -= q_cap
        split_segs = temp_q
        q_head = 0
        q_tail = q_cap
        q_cap *= 2
        ss_params[0] = q_head
        ss_params[2] = q_cap

    split_segs[q_tail, 0] = seg_idx
    split_segs[q_tail, 1] = v1
    split_segs[q_tail, 2] = v2

    q_tail += 1
    if q_tail >= q_cap:
        q_tail -= q_cap

    if q_head == q_tail:
        q_num_items = q_cap
    elif q_tail > q_head:
        q_num_items = q_tail - q_head
    else:
        q_num_items = (q_cap - q_head) + q_tail

    ss_params[1] = q_tail
    ss_params[3] = q_num_items

    return split_segs