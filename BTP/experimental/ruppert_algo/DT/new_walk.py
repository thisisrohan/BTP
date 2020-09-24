import numpy as np
from tools.adaptive_predicates import Two_Diff, Two_Two_Product, \
                                      fast_expansion_sum_zeroelim, \
                                      exactinit2d, scale_expansion_zeroelim, \
                                      scale_expansion

def njit(f):
    return f
# from numba import njit


@njit
def find_intersection_coeff(
        a_x, a_y, b_x, b_y, p_x, p_y, q_x, q_y, global_arr, splitter):

    pqx, pqx_tail = Two_Diff(p_x, q_x)
    pqy, pqy_tail = Two_Diff(p_y, q_y)
    abx, abx_tail = Two_Diff(a_x, b_x)
    bay, bay_tail = Two_Diff(b_y, a_y)
    bqx, bqx_tail = Two_Diff(b_x, q_x)
    bqy, bqy_tail = Two_Diff(b_y, q_y)

    qbx, qbx_tail = Two_Diff(q_x, b_x)

    det_left = global_arr[0:8]
    det_right = global_arr[8:16]
    det_arr = global_arr[16:32]
    m1_left = det_left
    m1_right = det_right
    m1_arr = global_arr[32:48]
    m2_left = det_left
    m2_right = det_right
    m2_arr = global_arr[48:64]
    temp_arr = global_arr[0:16]
    res_arr = global_arr[64:96]

    det_left[7], det_left[6], det_left[5], det_left[4], det_left[3], \
    det_left[2], det_left[1], det_left[0] = Two_Two_Product(
        pqx, pqx_tail, bay, bay_tail, splitter)
    
    det_right[7], det_right[6], det_right[5], det_right[4], det_right[3], \
    det_right[2], det_right[1], det_right[0] = Two_Two_Product(
        pqy, pqy_tail, abx, abx_tail, splitter)
    det_len = fast_expansion_sum_zeroelim(8, det_left, 8, det_right, det_arr)

    # m1
    m1_left[7], m1_left[6], m1_left[5], m1_left[4], m1_left[3], m1_left[2], \
    m1_left[1], m1_left[0] = Two_Two_Product(
        bay, bay_tail, bqx, bqx_tail, splitter)
    m1_right[7], m1_right[6], m1_right[5], m1_right[4], m1_right[3], \
    m1_right[2], m1_right[1], m1_right[0] = Two_Two_Product(
        abx, abx_tail, bqy, bqy_tail, splitter)
    m1_len = fast_expansion_sum_zeroelim(8, m1_left, 8, m1_right, m1_arr)

    # m2
    m2_left[7], m2_left[6], m2_left[5], m2_left[4], m2_left[3], m2_left[2], \
    m2_left[1], m2_left[0] = Two_Two_Product(
        pqy, pqy_tail, qbx, qbx_tail, splitter)
    m2_right[7], m2_right[6], m2_right[5], m2_right[4], m2_right[3], \
    m2_right[2], m2_right[1], m2_right[0] = Two_Two_Product(
        pqx, pqx_tail, bqy, bqy_tail, splitter)
    m2_len = fast_expansion_sum_zeroelim(8, m1_left, 8, m2_right, m2_arr)

    # '''
    if det_arr[det_len - 1] == 0.0:
        return 0.0, 0.0, 0.0
    elif det_arr[det_len - 1] > 0.0:
        if m1_arr[m1_len - 1] < 0.0:
            if m2_arr[m2_len - 1] > 0.0:
                return 1.0, -0.5, 0.5
            elif m2_arr[m2_len - 1] < 0.0:
                return 1.0, -0.5, -0.5
            else:
                return 1.0, -0.5, 0.0
        else:
            # m1 != 0.0
            temp_len = scale_expansion_zeroelim(
                det_len, det_arr, -1.0, temp_arr, splitter)

            if m1_arr[m1_len - 1] == 0.0:
                m1 = 0.0
            else:
                res_len = fast_expansion_sum_zeroelim(
                    m1_len, m1_arr, temp_len, temp_arr, res_arr)
                if res_arr[res_len - 1] < 0.0:
                    m1 = 0.5
                elif res_arr[res_len - 1] > 0.0:
                    m1 = 1.5
                else:
                    m1 = 1.0

            if m2_arr[m2_len - 1] == 0.0:
                m2 = 0.0
            else:
                res_len = fast_expansion_sum_zeroelim(
                    m2_len, m2_arr, temp_len, temp_arr, res_arr)
                if res_arr[res_len - 1] < 0.0:
                    if m2_arr[m2_len - 1] > 0.0:
                        m2 = 0.5
                    else:
                        m2 = -0.5
                elif res_arr[res_len -1] > 0.0:
                    m2 = 1.5
                else:
                    m2 = 1.0
            return 1.0, m1, m2
    elif det_arr[det_len - 1] < 0.0:
        if m1_arr[m1_len - 1] > 0.0:
            if m2_arr[m2_len - 1] > 0.0:
                return -1.0, -0.5, -0.5
            elif m2_arr[m2_len - 1] < 0.0:
                return -1.0, -0.5, 0.5
            else:
                return -1.0, -0.5, 0.0
        else:
            # m1 != 0.0
            temp_len = scale_expansion_zeroelim(
                det_len, det_arr, -1.0, temp_arr, splitter)

            if m1_arr[m1_len - 1] == 0.0:
                m1 = 0.0
            else:
                res_len = fast_expansion_sum_zeroelim(
                    m1_len, m1_arr, temp_len, temp_arr, res_arr)
                if res_arr[res_len - 1] > 0.0:
                    m1 = 0.5
                elif res_arr[res_len - 1] < 0.0:
                    m1 = 1.5
                else:
                    m1 = 1.0

            if m2_arr[m2_len - 1] == 0.0:
                m2 = 0.0
            else:
                res_len = fast_expansion_sum_zeroelim(
                    m2_len, m2_arr, temp_len, temp_arr, res_arr)
                if res_arr[res_len - 1] > 0.0:
                    if m2_arr[m2_len - 1] < 0.0:
                        m2 = 0.5
                    else:
                        m2 = -0.5
                elif res_arr[res_len -1] < 0.0:
                    m2 = 1.5
                else:
                    m2 = 1.0
            return -1.0, m1, m2
    # '''

    # temp_len = scale_expansion_zeroelim(
    #     det_len, det_arr, -1.0, temp_arr, splitter)
    # print("temp_len : {}, temp_arr : {}".format(temp_len, temp_arr[0:temp_len]))


    # return det_arr, det_len, m1_arr, m1_len, m2_arr, m2_len


@njit
def new_walk(
    p_x, p_y, q_x, q_y, t_index, vertices_ID, neighbour_ID, points_ID,
    exactinit_arr, global_arr):

    while True:
        v1 = vertices_ID[t_index, 0]
        v2 = vertices_ID[t_index, 1]
        v3 = vertices_ID[t_index, 2]

        v1_x = points[v1, 0]
        v1_y = points[v1, 1]
        v2_x = points[v2, 0]
        v2_y = points[v2, 1]
        v3_x = points[v3, 0]
        v3_y = points[v3, 1]

        j = 4
        proceed = int(1)
        det, m1, m2 = find_intersection_coeff(
            v1_x, v1_y, v2_x, v2_y, p_x, p_y, q_x, q_y, global_arr, splitter)
        if det != 0.0:
            if 0.0 < m1 and m1 < 1.0:
                if 0.0 < m2 and m2 < 1.0:
                    j = 2
                    proceed = 0
                elif m2 == 0.0:
                    t_index = walk_assist(p_x, p_y, q_x, q_y, t_index, )
                    proceed = -1



def perf():
    every_other = True
    half = np.float64(0.5)
    epsilon = np.float64(1.0)
    splitter = np.float64(1.0)
    one = np.float64(1.0)
    two = np.float64(2.0)
    while True:
        epsilon *= half
        if every_other:
          splitter *= two
        every_other = not every_other
        if one + epsilon != one:
            pass
        else:
            break
    splitter += one

    a_x = 0.012131321312312313
    a_y = 0.0
    b_x = 1.0
    b_y = 0.5
    p_x = 0.0000000000000012342343242
    p_y = 0.0
    q_x = 1.0
    q_y = 0.499999999999999
    global_arr = np.empty(shape=100, dtype=np.float64)

    # det_arr, det_len, m1_arr, m1_len, m2_arr, m2_len = find_intersection_coeff(
    #     a_x, a_y, b_x, b_y, p_x, p_y, q_x, q_y, global_arr, splitter)

    # print("det_len : {}, det_arr : {}".format(det_len, det_arr[0:det_len]))
    # print("m1_len : {}, m1_arr : {}".format(m1_len, m1_arr[0:m1_len]))
    # print("m2_len : {}, m2_arr : {}".format(m2_len, m2_arr[0:m2_len]))

    det, m1, m2 = find_intersection_coeff(
        a_x, a_y, b_x, b_y, p_x, p_y, q_x, q_y, global_arr, splitter)
    print("det : {}, m1 : {}, m2 : {}".format(det, m1, m2))

if __name__ == '__main__':
    perf()

