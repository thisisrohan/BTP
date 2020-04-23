import numpy as np
import matplotlib.pyplot as plt
import BTP.core.final_2D as t5


def njit(f):
    return f
# from numba import njit


# @njit
def find_and_fix_bad_segments(
    segments,
    num_segments,
    segments_to_delete,
    new_segments,
    ic_bad_tri,
    ic_boundary_tri,
    ic_boundary_vtx,
    points,
    num_points,
    vertices_ID,
    num_tri,
    neighbour_ID,
    plt_iter,
    org_num_points,
    bad_tri_indicator_arr,
):

    seg_iter = 0
    no_more_segs_to_split = True
    num_segments_to_delete = 0
    while seg_iter < num_segments:
        a_idx = segments[2*seg_iter]
        b_idx = segments[2*seg_iter+1]
        a_x = points[2*a_idx]
        a_y = points[2*a_idx+1]
        b_x = points[2*b_idx]
        b_y = points[2*b_idx+1]
        center_x = 0.5*(a_x + b_x)
        center_y = 0.5*(a_y + b_y)
        radius_square = 0.25*((a_x-b_x)*(a_x-b_x)+(a_y-b_y)*(a_y-b_y))
        is_encroached = False

        for i in range(num_points):
            if i == a_idx or i == b_idx:
                continue
            else:
                p_x = points[2*i]
                p_y = points[2*i+1]
                temp = (center_x-p_x)*(center_x-p_x)+(center_y-p_y)*(center_y-p_y)
                if temp < radius_square:
                    is_encroached = True
                    break

        if is_encroached == True:
            no_more_segs_to_split = False

            if 2*num_points >= len(points):
                # checking if the array has space to accommodate another element
                temp_arr_points = np.empty(2*len(points), dtype=np.float64)
                for l in range(2*num_points):
                    temp_arr_points[l] = points[l]
                points = temp_arr_points

            if a_idx < org_num_points and b_idx >= org_num_points:
                # a is an input vertex
                shell_number = int(np.round(np.log2(radius_square**0.5/0.001)))
                radius = radius_square**0.5
                frac = 0.001*(2**(shell_number-1))/radius
                center_x = a_x*(1-frac) + b_x*frac
                center_y = a_y*(1-frac) + b_y*frac
            elif a_idx >= org_num_points and b_idx < org_num_points:
                # b is an input vertex
                shell_number = int(np.round(np.log2(radius_square**0.5/0.001)))
                radius = radius_square**0.5
                frac = 0.001*(2**(shell_number-1))/radius
                center_x = b_x*(1-frac) + a_x*frac
                center_y = b_y*(1-frac) + a_y*frac

            points[2*num_points] = center_x
            points[2*num_points+1] = center_y
            num_points += 1

            # print("segment midpoint inserted")
            # print("x : {}, y : {} \n".format(center_x, center_y))

            old_tri = num_tri-1
            point_id = num_points-1
            neighbour_ID, vertices_ID, num_tri = t5.add_point(
                old_tri,
                ic_bad_tri,
                ic_boundary_tri,
                ic_boundary_vtx,
                points,
                vertices_ID,
                neighbour_ID,
                num_tri,
                bad_tri_indicator_arr,
                point_id,
            )
            if 2*num_tri-2 >= len(bad_tri_indicator_arr):
                bad_tri_indicator_arr = np.zeros(shape=2*(2*num_tri-2), dtype=np.bool_)

            if 2*num_segments+3 >= len(segments):
                # checking if the array has space to accommodate another element
                temp_arr_segs = np.empty(2*len(segments), dtype=np.int64)
                for l in range(len(segments)):
                    temp_arr_segs[l] = segments[l]
                segments = temp_arr_segs

            segments[2*num_segments] = a_idx
            segments[2*num_segments+1] = num_points-1
            num_segments += 1
            segments[2*num_segments] = num_points-1
            segments[2*num_segments+1] = b_idx
            num_segments += 1

            if num_segments_to_delete >= len(segments_to_delete):
                temp_seg_to_delete = np.empty(2*num_segments_to_delete, dtype=np.int64)
                for l in range(num_segments_to_delete):
                    temp_seg_to_delete[l] = segments_to_delete[l]
                segments_to_delete = temp_seg_to_delete

            segments_to_delete[num_segments_to_delete] = seg_iter
            num_segments_to_delete += 1

        seg_iter += 1

    # if no_more_segs_to_split == False:
    #     segments_to_delete_end = 0
    #     new_seg_iter = 0
    #     for i in range(num_segments):
    #         if i != segments_to_delete[segments_to_delete_end]:
    #             if 2*new_seg_iter >= len(new_segments):
    #                 temp_new_seg = np.empty(2*len(new_segments), dtype=np.int64)
    #                 for l in range(len(new_segments)):
    #                     temp_new_seg[l] = new_segments[l]
    #                 new_segments = temp_new_seg
    #             new_segments[2*new_seg_iter] = segments[2*i]
    #             new_segments[2*new_seg_iter+1] = segments[2*i+1]
    #             new_seg_iter += 1
    #         else:
    #             segments_to_delete_end += 1
    #             if segments_to_delete_end == num_segments_to_delete:
    #                 segments_to_delete_end -= 1
    #     # segments = new_segments
    #     # num_segments = new_seg_iter
    #     return new_segments, new_seg_iter, points, num_points, neighbour_ID, num_tri, vertices_ID, no_more_segs_to_split, plt_iter, bad_tri_indicator_arr
    # else:
    #     return segments, num_segments, points, num_points, neighbour_ID, num_tri, vertices_ID, no_more_segs_to_split, plt_iter, bad_tri_indicator_arr
    
    if no_more_segs_to_split == False:
        for i in range(num_segments_to_delete):
            seg = segments_to_delete[num_segments_to_delete-1-i]
            segments[2*seg:2*(num_segments-1)] = segments[2*(seg+1):2*num_segments]
            num_segments -= 1

    return segments, num_segments, points, num_points, neighbour_ID, num_tri, vertices_ID, no_more_segs_to_split, plt_iter, bad_tri_indicator_arr

# @njit
def find_and_fix_bad_triangles(
    segments,
    num_segments,
    segments_to_delete,
    new_segments,
    encroached_segments,
    min_angle,
    ic_bad_tri,
    ic_boundary_tri,
    ic_boundary_vtx,
    points,
    num_points,
    vertices_ID,
    num_tri,
    neighbour_ID,
    plt_iter,
    org_num_points,
    bad_tri_indicator_arr,
    num_tri_to_ignore,
):


    tri_iter = num_tri_to_ignore
    num_points_before_insertions = num_points
    no_more_tri_to_split = True
    encroached_segments_end = 0
    while tri_iter < num_tri:
        is_ghost = False
        if vertices_ID[3*tri_iter] == -1:
            is_ghost = True
        elif vertices_ID[3*tri_iter+1] == -1:
            is_ghost = True
        elif vertices_ID[3*tri_iter+2] == -1:
            is_ghost = True

        if is_ghost == False:
            a_idx = vertices_ID[3*tri_iter]
            b_idx = vertices_ID[3*tri_iter+1]
            c_idx = vertices_ID[3*tri_iter+2]
            a_x = points[2*a_idx]
            a_y = points[2*a_idx+1]
            b_x = points[2*b_idx]
            b_y = points[2*b_idx+1]
            c_x = points[2*c_idx]
            c_y = points[2*c_idx+1]
            a_sq = (b_x-c_x)*(b_x-c_x)+(b_y-c_y)*(b_y-c_y)
            b_sq = (c_x-a_x)*(c_x-a_x)+(c_y-a_y)*(c_y-a_y)
            c_sq = (a_x-b_x)*(a_x-b_x)+(a_y-b_y)*(a_y-b_y)
            a = a_sq**0.5
            b = b_sq**0.5
            c = c_sq**0.5
            # temp_max = max(a, b, c)
            temp_max = a
            if b > temp_max:
                temp_max = b
            if c > temp_max:
                temp_max = c
            if b == temp_max:
                a_idx, b_idx = b_idx, a_idx
                a_x, b_x = b_x, a_x
                a_y, b_y = b_y, a_y
                a_sq, b_sq = b_sq, a_sq
                a, b = b, a
            elif c == temp_max:
                a_idx, c_idx = c_idx, a_idx
                a_x, c_x = c_x, a_x
                a_y, c_y = c_y, a_y
                a_sq, c_sq = c_sq, a_sq
                a, c = c, a
            if c > b:
                c_idx, b_idx = b_idx, c_idx
                c_x, b_x = b_x, c_x
                c_y, b_y = b_y, c_y
                c_sq, b_sq = b_sq, c_sq
                c, b = b, c
            if c <= 0.1*a:
                if ( (a-b) + c )*( c + (b-a) ) < 0:
                    # import sys
                    print("***WTF__WTF__WTF***")
                    # print("a : " + str(a) + ", b : " + str(b) + ", c : " + str(c))
                    # print("tri_iter : {}".format(tri_iter))
                    # print("a_idx : {} , b_idx : {}, c_idx : {}".format(a_idx, b_idx, c_idx))
                    # print("num_points : {}".format(num_points))
                    # sys.exit()
                C = 2*np.arctan(((((a-b)+c)*(c+(b-a)))/((a+(b+c))*((a-c)+b)))**0.5)
                A = np.arccos(((c_sq+(b_sq-a_sq))/(2*c))/a)
                B = (np.pi - A) - C
            else:
                C = np.arccos(((b/a)+(a/b)-(c/a)*(c/b))*0.5)
                A = np.arccos(((c_sq+(b_sq-a_sq))/(2*c))/a)
                B = (np.pi - A) - C
                if np.isnan(A) == True:
                    # import sys
                    print("***WTF__WTF__WTF***")
                    # print("a : " + str(a) + ", b : " + str(b) + ", c : " + str(c))
                    # print("tri_iter : {}".format(tri_iter))
                    # print("a_idx : {} , b_idx : {}, c_idx : {}".format(a_idx, b_idx, c_idx))
                    # print("num_points : {}".format(num_points))
                    # print("arg : {}\n".format(((c_sq+(b_sq-a_sq))/(2*c))/a))
                    # sys.exit()

            if C < min_angle:
                # no_more_tri_to_split = False
                # tri_iter is a skinny triangle

                p1_x = 0.5*(b_x + a_x)
                p1_y = 0.5*(b_y + a_y)

                p2_x = 0.5*(c_x + a_x)
                p2_y = 0.5*(c_y + a_y)

                ex_1 = a_x - b_x
                ey_1 = a_y - b_y
                e0_1 = -(ex_1*p1_x + ey_1*p1_y)
                if np.abs(ex_1) > np.abs(ey_1):
                    ey_1 /= ex_1
                    e0_1 = -(p1_x + ey_1*p1_y)
                    ex_1 = 1.0
                else:
                    ex_1 /= ey_1
                    e0_1 = -(ex_1*p1_x + p1_y)
                    ey_1 = 1.0

                ex_2 = a_x - c_x
                ey_2 = a_y - c_y
                e0_2 = -(ex_2*p2_x + ey_2*p2_y)
                if np.abs(ex_2) > np.abs(ey_2):
                    ey_2 /= ex_2
                    e0_2 = -(p2_x + ey_2*p2_y)
                    ex_2 = 1.0
                else:
                    ex_2 /= ey_2
                    e0_2 = -(ex_2*p2_x + p2_y)
                    ey_2 = 1.0


                temp = ex_1*ey_2 - ey_1*ex_2
                circumcenter_x = (ey_1*e0_2-e0_1*ey_2)/temp
                circumcenter_y = (e0_1*ex_2-ex_1*e0_2)/temp

                if np.isnan(circumcenter_x) or np.isnan(circumcenter_y):
                    print("NaN as circumcenter")

                    p1_x = 0.5*(b_x + c_x)
                    p1_y = 0.5*(b_y + c_y)

                    p2_x = 0.5*(c_x + a_x)
                    p2_y = 0.5*(c_y + a_y)

                    ex_1 = c_x - b_x
                    ey_1 = c_y - b_y
                    e0_1 = -ex_1*p1_x - ey_1*p1_y

                    ex_2 = a_x - c_x
                    ey_2 = a_y - c_y
                    e0_2 = -ex_2*p2_x - ey_2*p2_y

                    temp = ex_1*ey_2 - ey_1*ex_2
                    circumcenter_x = (ey_1*e0_2-e0_1*ey_2)/temp
                    circumcenter_y = (e0_1*ex_2-ex_1*e0_2)/temp

                    if np.isnan(circumcenter_x) or np.isnan(circumcenter_y):
                        print("NaN as circumcenter")

                        p1_x = 0.5*(b_x + c_x)
                        p1_y = 0.5*(b_y + c_y)

                        p2_x = 0.5*(b_x + a_x)
                        p2_y = 0.5*(b_y + a_y)

                        ex_1 = c_x - b_x
                        ey_1 = c_y - b_y
                        e0_1 = -ex_1*p1_x - ey_1*p1_y

                        ex_2 = a_x - b_x
                        ey_2 = a_y - b_y
                        e0_2 = -ex_2*p2_x - ey_2*p2_y

                        temp = ex_1*ey_2 - ey_1*ex_2
                        circumcenter_x = (ey_1*e0_2-e0_1*ey_2)/temp
                        circumcenter_y = (e0_1*ex_2-ex_1*e0_2)/temp

                        if np.isnan(circumcenter_x) or np.isnan(circumcenter_y):
                            import sys
                            print("NaN as circumcenter")
                            print("a : ({}, {})".format(
                                points[2*vertices_ID[3*tri_iter+0]+0],
                                points[2*vertices_ID[3*tri_iter+0]+1]
                            ))
                            print("b : ({}, {})".format(
                                points[2*vertices_ID[3*tri_iter+1]+0],
                                points[2*vertices_ID[3*tri_iter+1]+1]
                            ))
                            print("c : ({}, {})".format(
                                points[2*vertices_ID[3*tri_iter+2]+0],
                                points[2*vertices_ID[3*tri_iter+2]+1]
                            ))
                            sys.exit()

                segs_are_encroached = False
                for i in range(num_segments):
                    # segment between k'th point and h'th point
                    h_idx = segments[2*i]
                    k_idx = segments[2*i+1]
                    h_x = points[2*h_idx]
                    h_y = points[2*h_idx+1]
                    k_x = points[2*k_idx]
                    k_y = points[2*k_idx+1]
                    center_x = 0.5*(h_x + k_x)
                    center_y = 0.5*(h_y + k_y)
                    radius_square = 0.25*((h_x-k_x)*(h_x-k_x)+(h_y-k_y)*(h_y-k_y))
                    temp = (center_x-circumcenter_x)*(center_x-circumcenter_x)+(center_y-circumcenter_y)*(center_y-circumcenter_y)
                    if temp < radius_square:
                        # circumcenter encroaches upon segment i
                        if encroached_segments_end >= len(encroached_segments):
                            # checking if the array has space to accommodate another element
                            temp_arr_encroached_segs = np.empty(2*len(encroached_segments), dtype=np.int64)
                            for l in range(encroached_segments_end):
                                temp_arr_encroached_segs[l] = encroached_segments[l]
                            encroached_segments = temp_arr_encroached_segs

                        encroached_segments[encroached_segments_end] = i
                        encroached_segments_end += 1

                        segs_are_encroached = True

                if segs_are_encroached == False:
                    if 2*num_points >= len(points):
                        # checking if the array has space to accommodate another element
                        temp_arr_points = np.empty(2*len(points), dtype=np.float64)
                        for l in range(2*num_points):
                            temp_arr_points[l] = points[l]
                        points = temp_arr_points

                    points[2*num_points] = circumcenter_x
                    points[2*num_points+1] = circumcenter_y
                    num_points += 1

        tri_iter += 1

    if num_points > num_points_before_insertions:
        no_more_tri_to_split = False
        for i in range(num_points_before_insertions, num_points):
            # print("circumcenter inserted")
            # print("x : {} , y : {} \n".format(points[2*i+0], points[2*i+1]))
            old_tri = num_tri-1
            point_id = i
            neighbour_ID, vertices_ID, num_tri = t5.add_point(
                old_tri,
                ic_bad_tri,
                ic_boundary_tri,
                ic_boundary_vtx,
                points,
                vertices_ID,
                neighbour_ID,
                num_tri,
                bad_tri_indicator_arr,
                point_id,
            )
            if 2*num_tri-2 >= len(bad_tri_indicator_arr):
                bad_tri_indicator_arr = np.zeros(shape=2*(2*num_tri-2), dtype=np.bool_)

            '''
            ########## PLOTTING ##########
            final_vertices = np.empty(shape=(num_tri, 3), dtype=np.int64)
            end = 0
            for i in range(num_tri):
                idx = np.where(vertices_ID[3*i:3*i+3] == -1)[0]
                if len(idx) == 0:
                    final_vertices[end] = vertices_ID[3*i:3*i+3]
                    end += 1
            final_vertices = final_vertices[0:end]
            plt.triplot(
                points[0:2*num_points:2],
                points[1:2*num_points:2],
                final_vertices
            )
            plt.plot(points[2*point_id], points[2*point_id+1], '*', color='k')
            plt.show()
            ##############################
            '''

    if encroached_segments_end > 0:
        no_more_tri_to_split = False
        for i in range(encroached_segments_end):
            seg_idx = encroached_segments[i]
            h_idx = segments[2*seg_idx]
            k_idx = segments[2*seg_idx+1]
            h_x = points[2*h_idx]
            h_y = points[2*h_idx+1]
            k_x = points[2*k_idx]
            k_y = points[2*k_idx+1]
            center_x = 0.5*(h_x + k_x)
            center_y = 0.5*(h_y + k_y)

            if 2*num_points >= len(points):
                # checking if the array has space to accommodate another element
                temp_arr_points = np.empty(2*len(points), dtype=np.float64)
                for l in range(2*num_points):
                    temp_arr_points[l] = points[l]
                points = temp_arr_points

            # '''
            if h_idx < org_num_points and k_idx >= org_num_points:
                # h is an input vertex
                radius = 0.5*((((h_x-k_x)*(h_x-k_x)+(h_y-k_y)*(h_y-k_y)))**0.5)
                shell_number = int(np.round(np.log2(radius/0.001)))
                frac = 0.001*(2**(shell_number-1))/radius
                center_x = h_x*(1-frac) + k_x*frac
                center_y = h_y*(1-frac) + k_y*frac
            elif h_idx >= org_num_points and k_idx < org_num_points:
                # k is an input vertex
                radius = 0.5*((((h_x-k_x)*(h_x-k_x)+(h_y-k_y)*(h_y-k_y)))**0.5)
                shell_number = int(np.round(np.log2(radius/0.001)))
                frac = 0.001*(2**(shell_number-1))/radius
                center_x = k_x*(1-frac) + h_x*frac
                center_y = k_y*(1-frac) + h_y*frac
            # '''

            # print("segment midpoint inserted")
            # print("x : {}, y : {} \n".format(center_x, center_y))

            points[2*num_points] = center_x
            points[2*num_points+1] = center_y
            num_points += 1

            old_tri = num_tri-1
            point_id = num_points-1

            neighbour_ID, vertices_ID, num_tri = t5.add_point(
                old_tri,
                ic_bad_tri,
                ic_boundary_tri,
                ic_boundary_vtx,
                points,
                vertices_ID,
                neighbour_ID,
                num_tri,
                bad_tri_indicator_arr,
                point_id,
            )
            if 2*num_tri-2 >= len(bad_tri_indicator_arr):
                bad_tri_indicator_arr = np.zeros(shape=2*(2*num_tri-2), dtype=np.bool_)

            '''
            ########## PLOTTING ##########
            final_vertices = np.empty(shape=(num_tri, 3), dtype=np.int64)
            end = 0
            for i in range(num_tri):
                idx = np.where(vertices_ID[3*i:3*i+3] == -1)[0]
                if len(idx) == 0:
                    final_vertices[end] = vertices_ID[3*i:3*i+3]
                    end += 1
            final_vertices = final_vertices[0:end]
            plt.triplot(
                points[0:2*num_points:2],
                points[1:2*num_points:2],
                final_vertices
            )
            plt.plot(points[2*point_id], points[2*point_id+1], '*', color='k')
            plt.show()
            ##############################
            '''

            if 2*num_segments >= len(segments):
                # checking if the array has space to accommodate another element
                temp_arr_segs = np.empty(2*len(segments), dtype=np.int64)
                for l in range(len(segments)):
                    temp_arr_segs[l] = segments[l]
                segments = temp_arr_segs

            segments[2*seg_idx+1] = point_id
            segments[2*num_segments] = point_id
            segments[2*num_segments+1] = k_idx
            num_segments += 1

    return segments, num_segments, points, num_points, neighbour_ID, num_tri, vertices_ID, no_more_tri_to_split, plt_iter, bad_tri_indicator_arr

@njit
def reset_tri(
    insertion_points,
    vertices_ID,
    num_tri,
    neighbour_ID,
    points,
    num_points,
    segments,
    num_segments,
    tri_to_be_deleted,
):
    '''
    insertion_points : 2k x 1 array (virus introduced at these k points)
    '''
    # print(insertion_points)
    num_viral_points = int(0.5*len(insertion_points))
    num_tri_to_be_deleted = 0
    old_tri = 0

    for k in range(num_viral_points):
        tri_iter = num_tri_to_be_deleted

        insertion_point_x = insertion_points[2*k]
        insertion_point_y = insertion_points[2*k+1]

        enclosing_tri = t5._walk(
            insertion_point_x,
            insertion_point_y,
            old_tri,
            vertices_ID,
            neighbour_ID,
            points,
        )

        # old_tri = enclosing_tri

        if num_tri_to_be_deleted >= len(tri_to_be_deleted):
            # checking if the array has space for another element
            temp_arr_del_tri = np.empty(2*num_tri_to_be_deleted, dtype=np.int64)
            for l in range(num_tri_to_be_deleted):
                temp_arr_del_tri[l] = tri_to_be_deleted[l]
            tri_to_be_deleted = temp_arr_del_tri

        tri_to_be_deleted[num_tri_to_be_deleted] = enclosing_tri
        num_tri_to_be_deleted += 1

        # last_tri = enclosing_tri

        while tri_iter < num_tri_to_be_deleted:

            tri_idx = tri_to_be_deleted[tri_iter]

            a_idx = vertices_ID[3*tri_idx]
            b_idx = vertices_ID[3*tri_idx+1]
            c_idx = vertices_ID[3*tri_idx+2]

            nbr_a = neighbour_ID[3*tri_idx]//3
            del_nbr_a = True
            nbr_b = neighbour_ID[3*tri_idx+1]//3
            del_nbr_b = True
            nbr_c = neighbour_ID[3*tri_idx+2]//3
            del_nbr_c = True

            for i in range(num_segments):
                h_idx = segments[2*i]
                k_idx = segments[2*i+1]

                if (h_idx == a_idx and k_idx == b_idx) or (h_idx == b_idx and k_idx == a_idx):
                    del_nbr_c = False
                if (h_idx == b_idx and k_idx == c_idx) or (h_idx == c_idx and k_idx == b_idx):
                    del_nbr_a = False
                if (h_idx == c_idx and k_idx == a_idx) or (h_idx == a_idx and k_idx == c_idx):
                    del_nbr_b = False

            for i in range(num_tri_to_be_deleted):
                temp_tri = tri_to_be_deleted[i]
                if nbr_a == temp_tri:
                    del_nbr_a = False
                if nbr_b == temp_tri:
                    del_nbr_b = False
                if nbr_c == temp_tri:
                    del_nbr_c = False

            if del_nbr_a == True:
                if num_tri_to_be_deleted >= len(tri_to_be_deleted):
                    # checking if the array has space for another element
                    temp_arr_del_tri = np.empty(2*num_tri_to_be_deleted, dtype=np.int64)
                    for l in range(num_tri_to_be_deleted):
                        temp_arr_del_tri[l] = tri_to_be_deleted[l]
                    tri_to_be_deleted = temp_arr_del_tri

                tri_to_be_deleted[num_tri_to_be_deleted] = nbr_a
                num_tri_to_be_deleted += 1

            if del_nbr_b == True:
                if num_tri_to_be_deleted >= len(tri_to_be_deleted):
                    # checking if the array has space for another element
                    temp_arr_del_tri = np.empty(2*num_tri_to_be_deleted, dtype=np.int64)
                    for l in range(num_tri_to_be_deleted):
                        temp_arr_del_tri[l] = tri_to_be_deleted[l]
                    tri_to_be_deleted = temp_arr_del_tri

                tri_to_be_deleted[num_tri_to_be_deleted] = nbr_b
                num_tri_to_be_deleted += 1

            if del_nbr_c == True:
                if num_tri_to_be_deleted >= len(tri_to_be_deleted):
                    # checking if the array has space for another element
                    temp_arr_del_tri = np.empty(2*num_tri_to_be_deleted, dtype=np.int64)
                    for l in range(num_tri_to_be_deleted):
                        temp_arr_del_tri[l] = tri_to_be_deleted[l]
                    tri_to_be_deleted = temp_arr_del_tri

                tri_to_be_deleted[num_tri_to_be_deleted] = nbr_c
                num_tri_to_be_deleted += 1

            # print("num_tri_to_be_deleted : {}".format(num_tri_to_be_deleted))
            tri_iter += 1

    tri_to_be_deleted[0:num_tri_to_be_deleted] = np.sort(
        tri_to_be_deleted[0:num_tri_to_be_deleted]
    )
    for i in range(num_tri_to_be_deleted):
        tri = tri_to_be_deleted[i]
        idx = i
        if tri != i:
            tri_j = 4
            idx_j = 4
            for j in range(3):
                if neighbour_ID[3*tri+j]//3 == idx:
                    tri_j = j
                if neighbour_ID[3*idx+j]//3 == tri:
                    idx_j = j

            if tri_j == 4 and idx_j == 4:
                for j in range(3):
                    neighbour_ID[neighbour_ID[3*tri+j]] = 3*idx + j
                for j in range(3):
                    neighbour_ID[neighbour_ID[3*idx+j]] = 3*tri + j
                for j in range(3):
                    temp = vertices_ID[3*tri+j]
                    vertices_ID[3*tri+j] = vertices_ID[3*idx+j]
                    vertices_ID[3*idx+j] = temp

                    temp = neighbour_ID[3*tri+j]
                    neighbour_ID[3*tri+j] = neighbour_ID[3*idx+j]
                    neighbour_ID[3*idx+j] = temp
            else:
                for j in range(3):
                    if j != tri_j:
                        neighbour_ID[neighbour_ID[3*tri+j]] = 3*idx + j
                for j in range(3):
                    if j != idx_j:
                        neighbour_ID[neighbour_ID[3*idx+j]] = 3*tri + j
                
                for j in range(3):
                    temp = vertices_ID[3*tri+j]
                    vertices_ID[3*tri+j] = vertices_ID[3*idx+j]
                    vertices_ID[3*idx+j] = temp

                for j in range(3):
                    temp = neighbour_ID[3*tri+j]
                    neighbour_ID[3*tri+j] = neighbour_ID[3*idx+j]
                    neighbour_ID[3*idx+j] = temp

                neighbour_ID[3*tri+idx_j] = 3*idx+tri_j
                neighbour_ID[3*idx+tri_j] = 3*tri+idx_j

            tri_to_be_deleted[i] = idx
            for j in range(i+1, num_tri_to_be_deleted):
                if tri_to_be_deleted[j] == idx:
                    tri_to_be_deleted[j] = tri

    return vertices_ID, neighbour_ID, num_tri_to_be_deleted


@njit
def insert_virus(
    insertion_points,
    vertices_ID,
    num_tri,
    neighbour_ID,
    points,
    num_points,
    segments,
    num_segments,
    tri_to_be_deleted,
):
    '''
    insertion_points : 2k x 1 array (virus introduced at these k points)
    '''

    num_viral_points = int(0.5*len(insertion_points))
    num_tri_to_be_deleted = 0
    old_tri = 0

    for k in range(num_viral_points):
        tri_iter = num_tri_to_be_deleted

        insertion_point_x = insertion_points[2*k]
        insertion_point_y = insertion_points[2*k+1]

        enclosing_tri = t5._walk(
            insertion_point_x,
            insertion_point_y,
            old_tri,
            vertices_ID,
            neighbour_ID,
            points,
        )

        old_tri = enclosing_tri

        if num_tri_to_be_deleted >= len(tri_to_be_deleted):
            # checking if the array has space for another element
            temp_arr_del_tri = np.empty(2*num_tri_to_be_deleted, dtype=np.int64)
            for l in range(num_tri_to_be_deleted):
                temp_arr_del_tri[l] = tri_to_be_deleted[l]
            tri_to_be_deleted = temp_arr_del_tri

        tri_to_be_deleted[num_tri_to_be_deleted] = enclosing_tri
        num_tri_to_be_deleted += 1

        last_tri = enclosing_tri

        while tri_iter < num_tri_to_be_deleted:

            tri_idx = tri_to_be_deleted[tri_iter]

            a_idx = vertices_ID[3*tri_idx]
            b_idx = vertices_ID[3*tri_idx+1]
            c_idx = vertices_ID[3*tri_idx+2]

            nbr_a = neighbour_ID[3*tri_idx]//3
            del_nbr_a = True
            nbr_b = neighbour_ID[3*tri_idx+1]//3
            del_nbr_b = True
            nbr_c = neighbour_ID[3*tri_idx+2]//3
            del_nbr_c = True

            for i in range(num_segments):
                h_idx = segments[2*i]
                k_idx = segments[2*i+1]

                if (h_idx == a_idx and k_idx == b_idx) or (h_idx == b_idx and k_idx == a_idx):
                    del_nbr_c = False
                if (h_idx == b_idx and k_idx == c_idx) or (h_idx == c_idx and k_idx == b_idx):
                    del_nbr_a = False
                if (h_idx == c_idx and k_idx == a_idx) or (h_idx == a_idx and k_idx == c_idx):
                    del_nbr_b = False

            for i in range(num_tri_to_be_deleted):
                temp_tri = tri_to_be_deleted[i]
                if nbr_a == temp_tri:
                    del_nbr_a = False
                if nbr_b == temp_tri:
                    del_nbr_b = False
                if nbr_c == temp_tri:
                    del_nbr_c = False

            if del_nbr_a == True:
                if num_tri_to_be_deleted >= len(tri_to_be_deleted):
                    # checking if the array has space for another element
                    temp_arr_del_tri = np.empty(2*num_tri_to_be_deleted, dtype=np.int64)
                    for l in range(num_tri_to_be_deleted):
                        temp_arr_del_tri[l] = tri_to_be_deleted[l]
                    tri_to_be_deleted = temp_arr_del_tri

                tri_to_be_deleted[num_tri_to_be_deleted] = nbr_a
                num_tri_to_be_deleted += 1

            if del_nbr_b == True:
                if num_tri_to_be_deleted >= len(tri_to_be_deleted):
                    # checking if the array has space for another element
                    temp_arr_del_tri = np.empty(2*num_tri_to_be_deleted, dtype=np.int64)
                    for l in range(num_tri_to_be_deleted):
                        temp_arr_del_tri[l] = tri_to_be_deleted[l]
                    tri_to_be_deleted = temp_arr_del_tri

                tri_to_be_deleted[num_tri_to_be_deleted] = nbr_b
                num_tri_to_be_deleted += 1

            if del_nbr_c == True:
                if num_tri_to_be_deleted >= len(tri_to_be_deleted):
                    # checking if the array has space for another element
                    temp_arr_del_tri = np.empty(2*num_tri_to_be_deleted, dtype=np.int64)
                    for l in range(num_tri_to_be_deleted):
                        temp_arr_del_tri[l] = tri_to_be_deleted[l]
                    tri_to_be_deleted = temp_arr_del_tri

                tri_to_be_deleted[num_tri_to_be_deleted] = nbr_c
                num_tri_to_be_deleted += 1

            tri_iter += 1

    tri_to_be_deleted[0:num_tri_to_be_deleted] = np.sort(tri_to_be_deleted[0:num_tri_to_be_deleted])
    for tri in tri_to_be_deleted[0:num_tri_to_be_deleted][::-1]:
        for j in range(3):
            if neighbour_ID[3*tri+j] != -1:
                neighbour_ID[neighbour_ID[3*tri+j]] = -1

        vertices_ID[3*tri:3*(num_tri-1)] = vertices_ID[3*(tri+1):3*num_tri]
        neighbour_ID[3*tri:3*(num_tri-1)] = neighbour_ID[3*(tri+1):3*num_tri]
        num_tri -= 1

        for i in range(num_tri):
            for j in range(3):
                if neighbour_ID[3*i+j]//3 > tri:
                    neighbour_ID[3*i+j] = 3*(neighbour_ID[3*i+j]//3-1) + neighbour_ID[3*i+j]%3

    return vertices_ID[0:3*num_tri], neighbour_ID[0:3*num_tri], num_tri


# @njit
def assembly(
    segments,
    num_segments,
    segments_to_delete,
    new_segments,
    encroached_segments,
    min_angle,
    ic_bad_tri,
    ic_boundary_tri,
    ic_boundary_vtx,
    points,
    num_points,
    vertices_ID,
    num_tri,
    neighbour_ID,
    bad_tri_indicator_arr,
    insertion_points,
    tri_to_be_deleted,
):

    print("\n   ASSEMBLY STARTED   \n")

    org_num_points = num_points
    plt_iter = 1
    num_tri_to_ignore = 0

    while True:
        print("--------- find_and_fix_bad_segments entered ---------")
        segments, num_segments, points, num_points, neighbour_ID, num_tri, vertices_ID, no_more_segs_to_split, plt_iter, bad_tri_indicator_arr = find_and_fix_bad_segments(
            segments,
            num_segments,
            segments_to_delete,
            new_segments,
            ic_bad_tri,
            ic_boundary_tri,
            ic_boundary_vtx,
            points,
            num_points,
            vertices_ID,
            num_tri,
            neighbour_ID,
            plt_iter,
            org_num_points,
            bad_tri_indicator_arr,
        )

        ########## PLOTTING ##########
        final_vertices = np.empty(shape=(num_tri, 3), dtype=np.int64)
        end = 0
        for i in range(num_tri):
            idx = np.where(vertices_ID[3*i:3*i+3] == -1)[0]
            if len(idx) == 0:
                final_vertices[end] = vertices_ID[3*i:3*i+3]
                end += 1
        final_vertices = final_vertices[0:end]
        plt.triplot(
            points[0:2*num_points:2],
            points[1:2*num_points:2],
            final_vertices
        )
        for i in range(num_segments):
            a_idx = segments[2*i+0]
            b_idx = segments[2*i+1]
            a_x = points[2*a_idx+0]
            a_y = points[2*a_idx+1]
            b_x = points[2*b_idx+0]
            b_y = points[2*b_idx+1]
            plt.plot([a_x, b_x], [a_y, b_y], linestyle='--',
                     marker='.', color='k', linewidth=0.7, markersize=5)
            if a_idx >= org_num_points:
                plt.plot(a_x, a_y, marker='.', color='g', markersize=5)
            if b_idx >= org_num_points:
                plt.plot(b_x, b_y, marker='.', color='g', markersize=5)
        plt.axis("equal")
        plt.show()
        ##############################

        if no_more_segs_to_split == False:
            print("no_more_segs_to_split is False")
        print("--------- find_and_fix_bad_segments exited --------- \n")

        if no_more_segs_to_split == True:
            break

    while True:
        print("--------- find_and_fix_bad_segments entered ---------")
        segments, num_segments, points, num_points, neighbour_ID, num_tri, vertices_ID, no_more_segs_to_split, plt_iter, bad_tri_indicator_arr = find_and_fix_bad_segments(
            segments,
            num_segments,
            segments_to_delete,
            new_segments,
            ic_bad_tri,
            ic_boundary_tri,
            ic_boundary_vtx,
            points,
            num_points,
            vertices_ID,
            num_tri,
            neighbour_ID,
            plt_iter,
            org_num_points,
            bad_tri_indicator_arr,
        )
        if no_more_segs_to_split == False:
            print("no_more_segs_to_split is False")
        print("--------- find_and_fix_bad_segments exited --------- \n")

        # vertices_ID, neighbour_ID, num_tri_to_ignore = reset_tri(
        #     insertion_points,
        #     vertices_ID,
        #     num_tri,
        #     neighbour_ID,
        #     points,
        #     num_points,
        #     segments,
        #     num_segments,
        #     tri_to_be_deleted,
        # )
        # print("num_tri : {}".format(num_tri))
        # print("num_tri_to_ignore : {}".format(num_tri_to_ignore))

        '''
        ########## PLOTTING ##########
        final_vertices = np.empty(shape=(num_tri, 3), dtype=np.int64)
        end = 0
        for i in range(num_tri):
            idx = np.where(vertices_ID[3*i:3*i+3] == -1)[0]
            if len(idx) == 0:
                final_vertices[end] = vertices_ID[3*i:3*i+3]
                end += 1
        final_vertices = final_vertices[0:end]
        plt.triplot(
            points[0:2*num_points:2],
            points[1:2*num_points:2],
            final_vertices
        )
        plt.axis("equal")
        plt.show()
        ##############################
        '''

        ########## PLOTTING ##########
        final_vertices = np.empty(shape=(num_tri-num_tri_to_ignore, 3), dtype=np.int64)
        end = 0
        for i in range(num_tri_to_ignore, num_tri):
            idx = np.where(vertices_ID[3*i:3*i+3] == -1)[0]
            if len(idx) == 0:
                final_vertices[end] = vertices_ID[3*i:3*i+3]
                end += 1
        final_vertices = final_vertices[0:end]
        plt.triplot(
            points[0:2*num_points:2],
            points[1:2*num_points:2],
            final_vertices
        )
        for i in range(num_segments):
            a_idx = segments[2*i+0]
            b_idx = segments[2*i+1]
            a_x = points[2*a_idx+0]
            a_y = points[2*a_idx+1]
            b_x = points[2*b_idx+0]
            b_y = points[2*b_idx+1]
            plt.plot([a_x, b_x], [a_y, b_y], linestyle='--',
                     marker='.', color='k', linewidth=0.7, markersize=5)
            if a_idx >= org_num_points:
                plt.plot(a_x, a_y, marker='.', color='g', markersize=5)
            if b_idx >= org_num_points:
                plt.plot(b_x, b_y, marker='.', color='g', markersize=5)
        plt.axis("equal")
        plt.show()
        ##############################

        print("--------- find_and_fix_bad_triangles entered ---------")
        segments, num_segments, points, num_points, neighbour_ID, num_tri, vertices_ID, no_more_tri_to_split, plt_iter, bad_tri_indicator_arr = find_and_fix_bad_triangles(
            segments,
            num_segments,
            segments_to_delete,
            new_segments,
            encroached_segments,
            min_angle,
            ic_bad_tri,
            ic_boundary_tri,
            ic_boundary_vtx,
            points,
            num_points,
            vertices_ID,
            num_tri,
            neighbour_ID,
            plt_iter,
            org_num_points,
            bad_tri_indicator_arr,
            num_tri_to_ignore,
        )

        print("--------- find_and_fix_bad_triangles exited ---------\n")


        if no_more_tri_to_split == True:
            print("EXITING ASSEMBLY")
            break

    return points[0:2*num_points], neighbour_ID[0:3*num_tri], vertices_ID[0:3*num_tri], segments[0:2*num_segments]

@njit
def remove_ghost_tri(
    vertices_ID,
    num_tri,
    final_vertices,
):

    num_tri_final = 0
    for i in np.arange(0, num_tri):
        is_ghost = False
        if vertices_ID[3*i] == -1:
            is_ghost = True
        elif vertices_ID[3*i+1] == -1:
            is_ghost = True
        elif vertices_ID[3*i+2] == -1:
            is_ghost = True
        if is_ghost == False:
            # i.e. i is not a ghost triangle
            final_vertices[3*num_tri_final] = vertices_ID[3*i]
            final_vertices[3*num_tri_final+1] = vertices_ID[3*i+1]
            final_vertices[3*num_tri_final+2] = vertices_ID[3*i+2]
            num_tri_final += 1

    return final_vertices[0:3*num_tri_final]


class RefinedDelaunay:

    def __init__(
        self,
        points,
        segments,
        insertion_points,
        min_angle=10
    ):
        '''
          points : 2N x 1
        segments : M x 2
        '''

        temp = np.random.rand(20)
        tempDT = t5.Delaunay2D(temp)
        tempDT.makeDT()
        del tempDT
        del temp

        # if len(insertion_points.shape) > 1:
        #     insertion_points = insertion_points.reshape(2*len(insertion_points))
        insertion_points = insertion_points.ravel()

        segments = segments.ravel()
        DT = t5.Delaunay2D(points, segments)
        segments = DT.makeDT()
        self.points = DT.points
        self.neighbour_ID = DT.neighbour_ID
        self.vertices_ID = DT.vertices_ID

        self.num_points = int(0.5*len(self.points))
        self.num_tri = DT.num_tri
        self.num_segments = int(0.5*len(segments))

        self.segments = np.empty(4*self.num_segments, dtype=np.int64)
        self.segments[0:2*self.num_segments] = segments

        for i in range(self.num_segments):
            a_idx = self.segments[2*i]
            b_idx = self.segments[2*i+1]
            a_x = self.points[2*a_idx]
            a_y = self.points[2*a_idx+1]
            b_x = self.points[2*b_idx]
            b_y = self.points[2*b_idx+1]
            # print("\n segment : {}".format(i))
            # print("a_idx : {} , a_x : {} , a_y : {}".format(a_idx, a_x, a_y))
            # print("b_idx : {} , b_x : {} , b_y : {}".format(b_idx, b_x, b_y))
            plt.plot([a_x, b_x], [a_y, b_y], linewidth=2, color='k')
            plt.text(0.5*(a_x+b_x), 0.5*(a_y+b_y), str(i))

        for i in range(self.num_points):
            plt.plot(self.points[2*i], self.points[2*i+1], '.', color='brown')


        plt.axis('equal')
        plt.title("Initial Point Set and the Constraints")
        plt.show()
        # plt.savefig("initial.png", dpi=300, bbox_inches='tight')

        # final_vertices = np.empty_like(self.vertices_ID)
        # final_vertices = remove_ghost_tri(
        #     self.vertices_ID,
        #     self.num_tri,
        #     final_vertices,
        # )
        # num_tri = int(len(final_vertices)/3)
        # plt.clf()
        # plt.triplot(
        #     self.points[0::2],
        #     self.points[1::2],
        #     final_vertices.reshape(num_tri, 3),
        #     linewidth=0.75
        # )
        # plt.axis('equal')
        # plt.title("Delaunay triangulation of initial point set")
        # plt.savefig("00.png", dpi=300, bbox_inches='tight')
        # plt.show()

        self.min_angle = np.pi*min_angle/180

        # Arrays that will be passed into the jit-ed functions
        # so that they don't have to get their hands dirty with
        # object creation.
        ### _identify_cavity
        ic_bad_tri = np.empty(50, dtype=np.int64)
        ic_boundary_tri = np.empty(50, dtype=np.int64)
        ic_boundary_vtx = np.empty(2*50, dtype=np.int64)
        bad_tri_indicator_arr = np.zeros(shape=2*self.num_points-2, dtype=np.bool_)
        ### find_and_fix_bad_segments
        segments_to_delete = np.empty(4*self.num_segments, dtype=np.int64)
        new_segments = np.empty(4*self.num_segments, dtype=np.int64)
        ### find_and_fix_bad_triangles
        encroached_segments = np.empty(4*self.num_segments, dtype=np.int64)
        tri_to_be_deleted = np.empty(self.num_tri, dtype=np.int64)

        self.points, self.neighbour_ID, self.vertices_ID, self.segments = assembly(
            self.segments,
            self.num_segments,
            segments_to_delete,
            new_segments,
            encroached_segments,
            self.min_angle,
            ic_bad_tri,
            ic_boundary_tri,
            ic_boundary_vtx,
            self.points,
            self.num_points,
            self.vertices_ID,
            self.num_tri,
            self.neighbour_ID,
            bad_tri_indicator_arr,
            insertion_points,
            tri_to_be_deleted,
        )

        self.num_points = int(0.5*len(self.points))
        self.num_tri = int(len(self.vertices_ID)/3)
        self.num_segments = int(0.5*len(self.segments))

        ### insert_virus

        self.vertices_ID, self.neighbour_ID, self.num_tri = insert_virus(
            insertion_points,
            self.vertices_ID,
            self.num_tri,
            self.neighbour_ID,
            self.points,
            self.num_points,
            self.segments,
            self.num_segments,
            tri_to_be_deleted,
        )

        # self.num_tri = int(len(self.vertices_ID)/3)


    def plotDT(self):
        final_vertices = np.empty_like(self.vertices_ID)
        final_vertices = remove_ghost_tri(
            self.vertices_ID,
            self.num_tri,
            final_vertices,
        )
        print(final_vertices)
        num_tri = int(len(final_vertices)/3)
        plt.clf()
        plt.triplot(
            self.points[0::2],
            self.points[1::2],
            final_vertices.reshape(num_tri, 3),
            linewidth=0.75
        )
        for i in range(self.num_segments):
            a_idx = self.segments[2*i]
            b_idx = self.segments[2*i+1]
            a_x = self.points[2*a_idx]
            a_y = self.points[2*a_idx+1]
            b_x = self.points[2*b_idx]
            b_y = self.points[2*b_idx+1]
            plt.plot([a_x, b_x], [a_y, b_y], linewidth=2, color='k')

        plt.axis('equal')
        plt.title("Final Triangulation")
        # plt.savefig("final.png", dpi=300, bbox_inches='tight')
        plt.show()



def make_data():
    # points = np.array([
    #     0.5, 0,
    #     4, 0,
    #     6, 2,
    #     0, 4,
    #     1.5, 0.5,
    #     3, 0.5,
    #     3, 1.5,
    #     1, 1.5,
    #     1, 2,
    #     1, 3, 
    #     2, 2
    # ])
    # segments = np.array([
    #     [4, 5],
    #     [5, 6],
    #     [6, 7],
    #     [7, 4],
    #     [0, 1],
    #     [1, 2],
    #     [2, 3],
    #     [3, 0],
    #     [8, 9],
    #     [9, 10],
    #     [10, 8]
    # ])
    # insertion_points = np.array([[2, 1], [1.1, 2.1]])

    # points = np.array([
    #     0, 0,
    #     1, 0,
    #     2, 0.5,
    #     3, 4,
    #     3, 5,
    #     2, 3,
    #     1, 4,
    #     2, 5,
    #     -1, 5,
    #     -1, 3, 
    #     1, 2,
    #     1, 1,
    #     # -5, -5,
    #     # 20, -5,
    #     # 20, 10,
    #     # -5, 10
    # ])
    # segments = np.array([
    #     [0, 1],
    #     [1, 2],
    #     [2, 3],
    #     [3, 4],
    #     [4, 5],
    #     [5, 6],
    #     [6, 7],
    #     [7, 8],
    #     [8, 9],
    #     [9, 10],
    #     [10, 11],
    #     [11, 0],
    #     # [12, 13],
    #     # [13, 14],
    #     # [14, 15],
    #     # [15, 12]
    # ])
    # insertion_points = np.array([0, 1])

    # points = np.array([
    #     0, 0,
    #     4, 0,
    #     4, 0.25,
    #     4, 0.5,
    #     4, 0.75,
    #     4, 1,
    #     4, 1.25,
    #     4, 1.5,
    #     4, 1.75,
    #     4, 2,
    #     4, 2.25,
    #     4, 2.5,
    #     4, 2.75,
    #     4, 3,
    #     0, 3
    # ])
    # segments = np.array([
    #     [0, 1],
    #     [1, 2],
    #     [2, 3],
    #     [3, 4],
    #     [4, 5],
    #     [5, 6],
    #     [6, 7],
    #     [7, 8],
    #     [8, 9],
    #     [9, 10],
    #     [10, 11],
    #     [11, 12],
    #     [12, 13],
    #     [13, 14],
    #     [14, 0]
    # ])
    # insertion_points = np.array([-1, 1])

    # points = 2*np.array([
    #     [0, 0],
    #     [0.051673, -0.014696552140281],
    #     [0.091043, -0.010076888237317],
    #     [0.120571, -0.004775805350042],
    #     [0.15502, 0.002348107051437],
    #     [0.199311, 0.011722536638535],
    #     [0.238681, 0.019467808525461],
    #     [0.280512, 0.026813946153663],
    #     [0.327264, 0.033801721445369],
    #     [0.356791, 0.037465229088082],
    #     [0.396161, 0.041377339370863],
    #     [0.43061, 0.043921863378774],
    #     [0.46998, 0.045929673463347],
    #     [0.504429, 0.047025465882686],
    #     [0.54872, 0.04763133198028],
    #     [0.583169, 0.047510713731955],
    #     [0.617618, 0.046871583225751],
    #     [0.656988, 0.045478849114091],
    #     [0.691437, 0.043636195070778],
    #     [0.725886, 0.04119194505892],
    #     [0.765256, 0.037668154787437],
    #     [0.799705, 0.033980982002866],
    #     [1, 0],
    #     [0.765256, 0.089657930233398],
    #     [0.725886, 0.100782773538959],
    #     [0.691437, 0.110032733067277],
    #     [0.656988, 0.118794121075188],
    #     [0.617618, 0.128123842345224],
    #     [0.583169, 0.135601270805556],
    #     [0.54872, 0.142340760456029],
    #     [0.504429, 0.149644630770221],
    #     [0.46998, 0.153953703014286],
    #     [0.43061, 0.157182393169817],
    #     [0.396161, 0.158457311722946],
    #     [0.356791, 0.158092729977226],
    #     [0.327264, 0.156509949910914],
    #     [0.280512, 0.151610048134506],
    #     [0.238681, 0.144616286473781],
    #     [0.199311, 0.135662517154915],
    #     [0.15502, 0.122458931825376],
    #     [0.120571, 0.109361591025835],
    #     [0.091043, 0.095598458819934],
    #     [0.051673, 0.071816746190796],
    #     [0.031988, 0.055318035275167],
    #     [-1, -1],
    #     [2, -1],
    #     [2, 1],
    #     [-1, 1]
    # ])-[1, 0]
    
    # segments = [[i, i+1] for i in range(len(points)-1-4)]
    # segments.append([len(points)-1-4, 0])
    # segments.append([len(points)-1-3, len(points)-1-2])
    # segments.append([len(points)-1-2, len(points)-1-1])
    # segments.append([len(points)-1-1, len(points)-1-0])
    # segments.append([len(points)-1-0, len(points)-1-3])
    # points = points.reshape(2*len(points))
    # points = np.round(points, 4)
    # segments = np.array(segments)
    # insertion_points = 2*np.array([0.051673, 0])-[1, 0]

    # points_1 = np.array([
    #     0.0, 0.0,
    #     .0075, .0176,
    #     .0125, .0215,
    #     .0250, .0276,
    #     .0375, .0316,
    #     .0500, .0347,
    #     .0750, .0394,
    #     .1000, .0428,
    #     .1250, .0455,
    #     .1500, .0476,
    #     .1750, .0493,
    #     .2000, .0507,
    #     .2500, .0528,
    #     .3000, .0540,
    #     .3500, .0547,
    #     .4000, .0550,
    #     .4500, .0548,
    #     .5000, .0543,
    #     .5500, .0533,
    #     .5750, .0527,
    #     .6000, .0519,
    #     .6250, .0511,
    #     .6500, .0501,
    #     .6750, .0489,
    #     .7000, .0476,
    #     .7250, .0460,
    #     .7500, .0442,
    #     .7750, .0422,
    #     .8000, .0398,
    #     .8250, .0370,
    #     .8500, .0337,
    #     .8750, .0300,
    #     .9000, .0255,
    #     .9250, .0204,
    #     .9500, .0144,
    #     .9750, .0074,
    #     1.0000, -.0008,
    # ])
    # points_2 = np.array([
    #     .0075, -.0176,
    #     .0125, -.0216,
    #     .0250, -.0281,
    #     .0375, -.0324,
    #     .0500, -.0358,
    #     .0750, -.0408,
    #     .1000, -.0444,
    #     .1250, -.0472,
    #     .1500, -.0493,
    #     .1750, -.0510,
    #     .2000, -.0522,
    #     .2500, -.0540,
    #     .3000, -.0548,
    #     .3500, -.0549,
    #     .4000, -.0541,
    #     .4500, -.0524,
    #     .5000, -.0497,
    #     .5500, -.0455,
    #     .5750, -.0426,
    #     .6000, -.0389,
    #     .6250, -.0342,
    #     .6500, -.0282,
    #     .6750, -.0215,
    #     .7000, -.0149,
    #     .7250, -.0090,
    #     .7500, -.0036,
    #     .7750, .0012,
    #     .8000, .0053,
    #     .8250, .0088,
    #     .8500, .0114,
    #     .8750, .0132,
    #     .9000, .0138,
    #     .9250, .0131,
    #     .9500, .0106,
    #     .9750, .0060,
    #     1.000, -.0013,
    # ])
    # points_2[0::2] = points_2[0::2][::-1]
    # points_2[1::2] = points_2[1::2][::-1]

    # points = np.append(points_1, points_2)
    # num_points = int(0.5*len(points))

    # segments = np.array([[i, (i+1)%num_points] for i in range(num_points)])

    # points = np.append(
    #     3*points,
    #     3*np.array([
    #         -1, -0.5,
    #         2, -0.5,
    #         2, 0.5,
    #         -1, 0.5,
    #     ])
    # )
    # num_points = int(0.5*len(points))
    # segments = np.append(
    #     segments,
    #     [
    #         [num_points-1-3, num_points-1-2],
    #         [num_points-1-2, num_points-1-1],
    #         [num_points-1-1, num_points-1-0],
    #         [num_points-1-0, num_points-1-3]
    #     ],
    #     axis=0
    # )
    # insertion_points = np.array([
    #     [3*0.5, 0],
    #     [100, 100]
    # ])

    points = np.array([
        1.000000, 0.001260,
        0.998459, 0.001476,
        0.993844, 0.002120,
        0.986185, 0.003182,
        0.975528, 0.004642,
        0.961940, 0.006478,
        0.945503, 0.008658,
        0.926320, 0.011149,
        0.904508, 0.013914,
        0.880203, 0.016914,
        0.853553, 0.020107,
        0.824724, 0.023452,
        0.793893, 0.026905,
        0.761249, 0.030423,
        0.726995, 0.033962,
        0.691342, 0.037476,
        0.654508, 0.040917,
        0.616723, 0.044237,
        0.578217, 0.047383,
        0.539230, 0.050302,
        0.500000, 0.052940,
        0.460770, 0.055241,
        0.421783, 0.057148,
        0.383277, 0.058609,
        0.345492, 0.059575,
        0.308658, 0.060000,
        0.273005, 0.059848,
        0.238751, 0.059092,
        0.206107, 0.057714,
        0.175276, 0.055709,
        0.146447, 0.053083,
        0.119797, 0.049854,
        0.095492, 0.046049,
        0.073680, 0.041705,
        0.054497, 0.036867,
        0.038060, 0.031580,
        0.024472, 0.025893,
        0.013815, 0.019854,
        0.006156, 0.013503,
        0.001541, 0.006877,
        0.000000, 0.000000,
        0.001541, -0.006877,
        0.006156, -0.013503,
        0.013815, -0.019854,
        0.024472, -0.025893,
        0.038060, -0.031580,
        0.054497, -0.036867,
        0.073680, -0.041705,
        0.095492, -0.046049,
        0.119797, -0.049854,
        0.146447, -0.053083,
        0.175276, -0.055709,
        0.206107, -0.057714,
        0.238751, -0.059092,
        0.273005, -0.059848,
        0.308658, -0.060000,
        0.345492, -0.059575,
        0.383277, -0.058609,
        0.421783, -0.057148,
        0.460770, -0.055241,
        0.500000, -0.052940,
        0.539230, -0.050302,
        0.578217, -0.047383,
        0.616723, -0.044237,
        0.654508, -0.040917,
        0.691342, -0.037476,
        0.726995, -0.033962,
        0.761249, -0.030423,
        0.793893, -0.026905,
        0.824724, -0.023452,
        0.853553, -0.020107,
        0.880203, -0.016914,
        0.904508, -0.013914,
        0.926320, -0.011149,
        0.945503, -0.008658,
        0.961940, -0.006478,
        0.975528, -0.004642,
        0.986185, -0.003182,
        0.993844, -0.002120,
        0.998459, -0.001476,
        1.000000, -0.001260,
    ])

    num_points = int(0.5*len(points))
    segments = np.array([[i, (i+1)%num_points] for i in range(num_points)])

    points = np.append(
        3*points,
        3*np.array([
            -1, -0.5,
            2, -0.5,
            2, 0.5,
            -1, 0.5,
        ])
    )
    num_points = int(0.5*len(points))
    segments = np.append(
        segments,
        [
            [num_points-1-3, num_points-1-2],
            [num_points-1-2, num_points-1-1],
            [num_points-1-1, num_points-1-0],
            [num_points-1-0, num_points-1-3]
        ],
        axis=0
    )
    insertion_points = np.array([
        [3*0.5, 0],
        [100, 100]
    ])

    # N = 30
    # points = 2*np.array([[np.cos(2*np.pi*i/N), np.sin(2*np.pi*i/N)] for i in range(N)])
    # points = np.append(
    #     points,
    #     3*np.array([
    #         [-2, -2],
    #         [4, -2],
    #         [4, 2],
    #         [-2, 2]
    #     ]),
    #     axis=0
    # )
    # segments = np.empty((len(points), 2), dtype=np.int64)
    # segments[0:-4-1] = np.array([[i, i+1] for i in range(len(points)-4-1)])
    # segments[-4-1] = [len(points)-5, 0]
    # segments[-4:] = np.array([
    #     [len(points)-4, len(points)-4+1],
    #     [len(points)-4+1, len(points)-4+2],
    #     [len(points)-4+2, len(points)-4+3],
    #     [len(points)-4+3, len(points)-4],
    # ])

    # num_points_old = len(points)
    # points_2 = 2.1*np.array([[np.cos(2*np.pi*i/N), np.sin(2*np.pi*i/N)] for i in range(N)])
    # points = np.append(points, points_2, axis=0)

    # segments_2 = np.array([[num_points_old + i, num_points_old + (i+1)%N] for i in range(N)])
    # segments = np.append(
    #     segments,
    #     segments_2,
    #     axis=0
    # )

    # points = points.reshape(2*len(points))
    # insertion_points = np.array([[0, 0], [100, 100]])

    return points, segments, insertion_points

if __name__ == "__main__":
    #import sys
    #perf(int(sys.argv[1]))

    points, segments, insertion_points = make_data()

    RDT = RefinedDelaunay(
        points,
        segments,
        insertion_points,
        min_angle=10,
    )

    RDT.plotDT()