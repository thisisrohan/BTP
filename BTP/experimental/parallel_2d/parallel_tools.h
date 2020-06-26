#ifndef PARALLEL_TOOLS_H
#define PARALLEL_TOOLS_H

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "predicates.h"
#include "basic_tools.h"
#include "queue.h"

#ifndef INT64
#define INT64 int64_t
#endif
#ifndef UINT64
#define UINT64 uint64_t
#endif


INT64 _walk_parallel(
        INT64 t_index, INT64 p, INT64 gv, INT64* vertices_ID,
        INT64* neighbour_ID, double* points, double* global_arr,
        char* tri_color, INT64 tID) {

    // printf("tID : %ld, point : %ld, walk entered\n", tID, p);
    INT64 a, b, c, internal_idx;
    double *a_pts, *b_pts, *c_pts, *p_pts;
    double det;

    internal_idx = 4;
    if ( vertices_ID[3*t_index + 1] == gv ) {
        internal_idx = 1;
    } else if ( vertices_ID[3*t_index + 2] == gv ) {
        internal_idx = 2;
    }

    if ( internal_idx != 4 ) {
        t_index = neighbour_ID[3*t_index + internal_idx] / 3;
    }

    p_pts = &points[2*p];

    while ( true ) {
        if ( tri_color[t_index] != (char) tID ) return -1;

        a = vertices_ID[3*t_index + 0];
        b = vertices_ID[3*t_index + 1];
        c = vertices_ID[3*t_index + 2];

        // printf("t_index : %ld , [%ld, %ld, %ld], p : %ld \n", t_index, a, b, c, p);

        a_pts = &points[2*a];
        b_pts = &points[2*b];
        c_pts = &points[2*c];

        if ( (p_pts[0] == a_pts[0]) && (p_pts[1] == a_pts[1]) ) return -2;
        if ( (p_pts[0] == b_pts[0]) && (p_pts[1] == b_pts[1]) ) return -2;
        if ( (p_pts[0] == c_pts[0]) && (p_pts[1] == c_pts[1]) ) return -2;

        // printf("t : %ld, a : (%f, %f), b : (%f, %f), c : (%f, %f), p : (%f, %f)\n", t_index, a_pts[0], a_pts[1], b_pts[0], b_pts[1], c_pts[0], c_pts[1], p_pts[0], p_pts[1]);

        internal_idx = 4;

        det = orient2d(a_pts, p_pts, b_pts, global_arr);
        if ( det > 0.0 ) {
            internal_idx = 2;
        } else {
            det = orient2d(b_pts, p_pts, c_pts, global_arr);
            if ( det > 0.0 ) {
                internal_idx = 0;
            } else {
                det = orient2d(c_pts, p_pts, a_pts, global_arr);
                if ( det > 0.0 ) {
                    internal_idx = 1;
                }
            }
        }

        if ( internal_idx == 4 ) {
            break;
        } else {
            t_index = neighbour_ID[3*t_index + internal_idx] / 3;
            if (vertices_ID[3*t_index + 1] == gv) {
                break;
            } else if (vertices_ID[3*t_index + 2] == gv) {
                break;
            }
        }
    }
    // printf("tID : %ld, point : %ld, walk exiting\n", tID, p);

    return t_index;
}


INT64 _identify_cavity_parallel(
        INT64 t_index, INT64 p, INT64 gv, INT64* vertices_ID,
        INT64* neighbour_ID, double* points, bool* bad_tri_indicator_arr,
        INT64* arr_sizes, INT64** arr_pointers, double* global_arr,
        char* tri_color, INT64 tID) {

    INT64 bt_len, boundary_len, bt_end, boundary_end, bt_iter, j, jth_nbr;
    bt_len = arr_sizes[0];
    boundary_len = arr_sizes[1];
    bt_end = 0;
    boundary_end = 0;
    bt_iter = 0;

    INT64 *bad_tri, *boundary_tri, *boundary_vtx;
    bad_tri = arr_pointers[0];
    boundary_tri = arr_pointers[1];
    boundary_vtx = arr_pointers[2];

    bad_tri[bt_end] = t_index;
    bad_tri_indicator_arr[t_index] = true;
    bt_end += 1;

    bool inside_tri;
    while ( true ) {
        t_index = bad_tri[ bt_iter ];
        if ( tri_color[t_index] != (char) tID ) {
            arr_sizes[2] = bt_end;
            return -1;
        }

        for ( j = 0; j < 3; ++j ) {
            jth_nbr = neighbour_ID[3*t_index + j] / 3;
            if ( tri_color[jth_nbr] != (char) tID ) {
                arr_sizes[2] = bt_end;
                return -1;
            }

            if ( bad_tri_indicator_arr[jth_nbr] == false ) {
                inside_tri = _cavity_helper(
                    jth_nbr, p, gv, vertices_ID, points, global_arr);

                if ( inside_tri == true ) {
                    // add jth_nbr to the bad_tri array
                    if ( bt_end >= bt_len ) {
                        bt_len += 32;
                        #pragma omp critical (MEM_ALLOC)
                        {
                            bad_tri = (INT64 *) aligned_realloc(
                                bad_tri, bt_len * sizeof(INT64));
                        }
                        arr_sizes[0] = bt_len;
                        arr_pointers[0] = bad_tri;
                    }
                    bad_tri[bt_end] = jth_nbr;
                    bad_tri_indicator_arr[jth_nbr] = true;
                    bt_end += 1;
                } else {
                    if ( boundary_end >= boundary_len ) {
                        boundary_len += 32;
                        #pragma omp critical (MEM_ALLOC)
                        {
                            boundary_tri = (INT64 *) aligned_realloc(
                                boundary_tri, boundary_len * sizeof(INT64));
                            boundary_vtx = (INT64 *) aligned_realloc(
                                boundary_vtx, 2 * boundary_len * sizeof(INT64));
                        }
                        arr_sizes[1] = boundary_len;
                        arr_pointers[1] = boundary_tri;
                        arr_pointers[2] = boundary_vtx;
                    }
                    boundary_tri[boundary_end] = neighbour_ID[3*t_index + j];
                    boundary_vtx[2*boundary_end + 0] = vertices_ID[3*t_index + (j + 1) % 3];
                    boundary_vtx[2*boundary_end + 1] = vertices_ID[3*t_index + (j + 2) % 3];
                    boundary_end += 1;
                }
            }
        }

        bt_iter += 1;
        if ( bt_iter == bt_end ) {
            break;
        }
    }

    arr_sizes[2] = bt_end;
    arr_sizes[3] = boundary_end;

    return 0;
}


INT64 _make_Delaunay_ball_parallel(
        INT64 p, INT64* vertices_ID, INT64* neighbour_ID, INT64* arr_sizes,
        INT64** arr_pointers, INT64* num_entities, bool* existing_tri,
        INT64* q_params, INT64 tID, char* tri_color) {

    INT64 bt_end, boundary_end, i, j;
    bt_end = arr_sizes[2];
    boundary_end = arr_sizes[3];

    INT64 nbr_info, t1, t2, num_tri;
    // num_tri = num_entities[1];

    INT64 *bad_tri, *boundary_tri, *boundary_vtx, *tri_queue;
    bad_tri = arr_pointers[0];
    boundary_tri = arr_pointers[1];
    boundary_vtx = arr_pointers[2];
    tri_queue = arr_pointers[3];

    INT64 q_head1, q_head2, q_cap, q_tail, q_num_items;
    q_head1 = q_params[0];
    q_tail = q_params[1];
    q_cap = q_params[2];

    for ( i = 0; i < boundary_end; i++ ) {
        if ( i < bt_end ) {
            t1 = bad_tri[i];
        } else {
            // t1 = dequeue(arr_pointers, q_params)
            // num_tri += 1;
            t1 = tri_queue[q_head1];
            q_head1 += 1;
            if ( q_head1 >= q_cap ) q_head1 -= q_cap;
            existing_tri[t1] = true;
            tri_color[t1] = (char) tID;
        }

        nbr_info = boundary_tri[i];
        neighbour_ID[3*t1 + 0] = nbr_info;
        vertices_ID[3*t1 + 0] = p;
        vertices_ID[3*t1 + 1] = boundary_vtx[2*i + 0];
        vertices_ID[3*t1 + 2] = boundary_vtx[2*i + 1];
        neighbour_ID[nbr_info] = 3*t1;
    }

    q_head1 = q_params[0];
    for ( i = 0; i < boundary_end; ++i ) {
        if ( i < bt_end ) {
            t1 = bad_tri[i];
        } else {
            t1 = tri_queue[q_head1];
            q_head1 += 1;
            if ( q_head1 >= q_cap ) q_head1 -= q_cap;
        }
        q_head2 = q_params[0];
        for ( j = 0; j < boundary_end; ++j ) {
            if ( j < bt_end ) {
                t2 = bad_tri[j];
            } else {
                t2 = tri_queue[q_head2];
                q_head2 += 1;
                if ( q_head2 >= q_cap ) q_head2 -= q_cap;
            }
            if ( vertices_ID[3*t1 + 1] == vertices_ID[3*t2 + 2] ) {
                neighbour_ID[3*t1 + 2] = 3*t2 + 1;
                neighbour_ID[3*t2 + 1] = 3*t1 + 2;
                break;
            }
        }
    }

    if ( q_head1 == q_tail ) q_num_items = 0;
    else if ( q_tail > q_head1 ) q_num_items = q_tail - q_head1;
    else q_num_items = (q_cap - q_head1) + q_tail;

    q_params[0] = q_head1;
    q_params[3] = q_num_items;

    return bad_tri[bt_end - 1];
}

#endif /* PARALLEL_TOOLS_H */