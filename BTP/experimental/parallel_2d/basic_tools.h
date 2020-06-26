#ifndef BASIC_TOOLS_H
#define BASIC_TOOLS_H

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "predicates.h"
#include "BRIO_2D.h"
#include "helpful_tools.h"

#ifndef INT64
#define INT64 int64_t
#endif
#ifndef UINT64
#define UINT64 uint64_t
#endif


INT64 _walk(
        INT64 t_index, INT64 p, INT64 gv, INT64* vertices_ID,
        INT64* neighbour_ID, double* points, double* global_arr) {

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

        a = vertices_ID[3*t_index + 0];
        b = vertices_ID[3*t_index + 1];
        c = vertices_ID[3*t_index + 2];

        a_pts = &points[2*a];
        b_pts = &points[2*b];
        c_pts = &points[2*c];

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
            if ( (p_pts[0] == a_pts[0]) && (p_pts[1] == a_pts[1]) ) return -2;
            if ( (p_pts[0] == b_pts[0]) && (p_pts[1] == b_pts[1]) ) return -2;
            if ( (p_pts[0] == c_pts[0]) && (p_pts[1] == c_pts[1]) ) return -2;
            break;
        } else {
            t_index = neighbour_ID[3*t_index + internal_idx] / 3;
            if (vertices_ID[3*t_index + 1] == gv) {
                a = vertices_ID[3*t_index + 0];
                c = vertices_ID[3*t_index + 2];
                a_pts = &points[2*a];
                c_pts = &points[2*c];
                if ( (p_pts[0] == a_pts[0]) && (p_pts[1] == a_pts[1]) ) return -2;
                if ( (p_pts[0] == c_pts[0]) && (p_pts[1] == c_pts[1]) ) return -2;
                break;
            } else if (vertices_ID[3*t_index + 2] == gv) {
                a = vertices_ID[3*t_index + 0];
                b = vertices_ID[3*t_index + 1];
                a_pts = &points[2*a];
                b_pts = &points[2*b];
                if ( (p_pts[0] == a_pts[0]) && (p_pts[1] == a_pts[1]) ) return -2;
                if ( (p_pts[0] == b_pts[0]) && (p_pts[1] == b_pts[1]) ) return -2;
                break;
            }
        }
    }

    return t_index;
}


bool _cavity_helper(
        INT64 t_index, INT64 p, INT64 gv, INT64* vertices_ID, double* points,
        double* global_arr) {

    INT64 a, b, c, internal_idx;
    double *a_pts, *b_pts, *c_pts, *p_pts;
    double det, alpha;
    bool res;

    internal_idx = 4;
    if ( vertices_ID[3*t_index + 1] == gv ) {
        internal_idx = 1;
    } else if ( vertices_ID[3*t_index + 2] == gv ) {
        internal_idx = 2;
    }

    p_pts = &points[2*p];

    if ( internal_idx != 4 ) {
        // t_index is a ghost triangle
        a = vertices_ID[3*t_index + (internal_idx + 1) % 3];
        b = vertices_ID[3*t_index + (internal_idx + 2) % 3];

        a_pts = &points[2*a];
        b_pts = &points[2*b];

        det = orient2d(p_pts, a_pts, b_pts, global_arr);

        if ( det > 0 ) {
            res = true;
        } else if ( det < 0 ) {
            res = false;
        } else {
            alpha = find_intersection(a_pts, b_pts, p_pts);
            if ( ( 0.0 < alpha ) && ( alpha < 1.0 ) ) {
                res = true;
            } else {
                res = false;
            }
        }
    } else {
        // t_index is a real triangle
        a = vertices_ID[3*t_index + 0];
        b = vertices_ID[3*t_index + 1];
        c = vertices_ID[3*t_index + 2];

        a_pts = &points[2*a];
        b_pts = &points[2*b];
        c_pts = &points[2*c];

        det = incircle(a_pts, b_pts, c_pts, p_pts, global_arr);

        if ( det >= 0 ) {
            res = true;
        } else {
            res = false;
        }
    }

    // if ( res == true ) printf("true\n");
    // else printf("false\n");
    return res;
}


void _identify_cavity(
        INT64 t_index, INT64 p, INT64 gv, INT64* vertices_ID,
        INT64* neighbour_ID, double* points, bool* bad_tri_indicator_arr,
        INT64* arr_sizes, INT64** arr_pointers, double* global_arr) {

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

    // long jth_nbr;
    bool inside_tri;
    while ( true ) {
        t_index = bad_tri[ bt_iter ];

        for ( j = 0; j < 3; j++ ) {
            jth_nbr = neighbour_ID[3*t_index + j] / 3;

            if ( bad_tri_indicator_arr[jth_nbr] == false ) {
                inside_tri = _cavity_helper(
                    jth_nbr, p, gv, vertices_ID, points, global_arr);

                if ( inside_tri == true ) {
                    // add jth_nbr to the bad_tri array
                    if ( bt_end >= bt_len ) {
                        bt_len += 32;
                        bad_tri = (INT64 *) aligned_realloc(
                            bad_tri, bt_len * sizeof(INT64));
                        arr_sizes[0] = bt_len;
                        arr_pointers[0] = bad_tri;
                    }
                    bad_tri[bt_end] = jth_nbr;
                    bad_tri_indicator_arr[jth_nbr] = true;
                    bt_end += 1;
                } else {
                    if ( boundary_end >= boundary_len ) {
                        boundary_len += 32;
                        boundary_tri = (INT64 *) aligned_realloc(
                            boundary_tri, boundary_len * sizeof(INT64));
                        boundary_vtx = (INT64 *) aligned_realloc(
                            boundary_vtx, 2 * boundary_len * sizeof(INT64));
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

    return;
}


INT64 _make_Delaunay_ball(
        INT64 p, INT64* vertices_ID, INT64* neighbour_ID, INT64* arr_sizes,
        INT64** arr_pointers, INT64* num_entities, bool* existing_tri) {

    INT64 bt_end, boundary_end, i, j;
    bt_end = arr_sizes[2];
    boundary_end = arr_sizes[3];

    INT64 nbr_info, t1, t2, num_tri;
    num_tri = num_entities[1];

    INT64 *bad_tri, *boundary_tri, *boundary_vtx;
    bad_tri = arr_pointers[0];
    boundary_tri = arr_pointers[1];
    boundary_vtx = arr_pointers[2];

    for ( i = 0; i < boundary_end; i++ ) {
        if ( i < bt_end ) {
            t1 = bad_tri[i];
        } else {
            t1 = num_tri;
            num_tri += 1;
            existing_tri[t1] = true;
        }

        nbr_info = boundary_tri[i];
        neighbour_ID[3*t1 + 0] = nbr_info;
        vertices_ID[3*t1 + 0] = p;
        vertices_ID[3*t1 + 1] = boundary_vtx[2*i + 0];
        vertices_ID[3*t1 + 2] = boundary_vtx[2*i + 1];
        neighbour_ID[nbr_info] = 3*t1;
    }

    for ( i = 0; i < boundary_end; i++ ) {
        if ( i < bt_end ) {
            t1 = bad_tri[i];
        } else {
            t1 = num_tri - boundary_end + i;
        }
        for ( j = 0; j < boundary_end; j++ ) {
            if ( j < bt_end ) {
                t2 = bad_tri[j];
            } else {
                t2 = num_tri - boundary_end + j;
            }
            if ( vertices_ID[3*t1 + 1] == vertices_ID[3*t2 + 2] ) {
                neighbour_ID[3*t1 + 2] = 3*t2 + 1;
                neighbour_ID[3*t2 + 1] = 3*t1 + 2;
                break;
            }
        }
    }

    num_entities[1] = num_tri;
    return bad_tri[bt_end - 1];
}


void _initialize(
        INT64* vertices_ID, INT64* neighbour_ID, double* points,
        INT64* num_entities, INT64* insertion_seq, INT64 gv,
        double* global_arr) {

    INT64 idx, i, num_points, templ;
    num_points = num_entities[0];

    double det, tempd;

    double *a_pts, *b_pts, *p_pts;

    a_pts = &points[0];
    b_pts = &points[2];

    for ( i = 2; i < num_points; i++ ) {
        p_pts = &points[2*i];
        det = orient2d(a_pts, b_pts, p_pts, global_arr);

        if ( det != 0.0 ) {
            tempd = points[4];
            points[4] = points[2*i];
            points[2*i] = tempd;

            tempd = points[5];
            points[5] = points[2*i + 1];
            points[2*i + 1] = tempd;

            templ = insertion_seq[2];
            insertion_seq[2] = insertion_seq[i];
            insertion_seq[i] = templ;

            if ( det < 0.0 ) {
                tempd = points[0];
                points[0] = points[2];
                points[2] = tempd;

                tempd = points[1];
                points[1] = points[3];
                points[3] = tempd;

                templ = insertion_seq[0];
                insertion_seq[0] = insertion_seq[1];
                insertion_seq[1] = templ;
            }

            break;
        }
    }

    vertices_ID[3*0 + 0] = 0;            //
    vertices_ID[3*0 + 1] = 1;            // ---> 0th triangle [real]
    vertices_ID[3*0 + 2] = 2;            //

    vertices_ID[3*1 + 0] = 0;            //
    vertices_ID[3*1 + 1] = gv;           // ---> 1st triangle [ghost]
    vertices_ID[3*1 + 2] = 1;            //

    vertices_ID[3*2 + 0] = 1;            //
    vertices_ID[3*2 + 1] = gv;           // ---> 2nd triangle [ghost]
    vertices_ID[3*2 + 2] = 2;            //

    vertices_ID[3*3 + 0] = 2;            //
    vertices_ID[3*3 + 1] = gv;           // ---> 3rd triangle [ghost]
    vertices_ID[3*3 + 2] = 0;            //

    neighbour_ID[3*0 + 0] = 3*2 + 1;     //
    neighbour_ID[3*0 + 1] = 3*3 + 1;     // ---> 0th triangle [real]
    neighbour_ID[3*0 + 2] = 3*1 + 1;     //

    neighbour_ID[3*1 + 0] = 3*2 + 2;     //
    neighbour_ID[3*1 + 1] = 3*0 + 2;     // ---> 1st triangle [ghost]
    neighbour_ID[3*1 + 2] = 3*3 + 0;     //

    neighbour_ID[3*2 + 0] = 3*3 + 2;     //
    neighbour_ID[3*2 + 1] = 3*0 + 0;     // ---> 2nd triangle [ghost]
    neighbour_ID[3*2 + 2] = 3*1 + 0;     //

    neighbour_ID[3*3 + 0] = 3*1 + 2;     //
    neighbour_ID[3*3 + 1] = 3*0 + 1;     // ---> 3rd triangle [ghost]
    neighbour_ID[3*3 + 2] = 3*2 + 0;     //

    num_entities[1] = 4;

    return;
}

/*
void _assembly(
        INT64* vertices_ID, INT64* neighbour_ID, double* original_points,
        INT64* num_entities, INT64* insertion_seq, INT64 gv) {

    INT64 i, p, num_points, bt_end, enclosing_tri, old_tri, t;
    INT64 *bad_tri, *boundary_indices;
    double *points, *global_arr;

    num_points = num_entities[0];

    points = (double *) aligned_malloc(2 * num_points * sizeof(double));
    global_arr = (double *) aligned_malloc(3236 * sizeof(double));
    // printf("yo\n");

    INT64* sfc_related_arr[2];
    INT64 h_deg = make_BRIO(
        original_points, points, insertion_seq, num_points, sfc_related_arr);
    // printf("BRIO made\n");

    exactinit2d(points, num_points);
    // printf("exactinit2d done\n");

    _initialize(
        vertices_ID, neighbour_ID, points, num_entities, insertion_seq, gv,
        global_arr);
    // printf("initialized vertices_ID and neighbour_ID\n");

    INT64 arr_sizes[4];
    arr_sizes[0] = 64;
    arr_sizes[1] = 64;

    INT64* arr_pointers[3];
    arr_pointers[0] = (INT64 *) aligned_malloc(64 * sizeof(INT64));
    arr_pointers[1] = (INT64 *) aligned_malloc(64 * sizeof(INT64));
    arr_pointers[2] = (INT64 *) aligned_malloc(2 * 64 * sizeof(INT64));

    bool* bad_tri_indicator_arr;
    bad_tri_indicator_arr = (bool *) malloc((2*num_points - 2) * sizeof(bool));
    for ( i = 0; i < 2*num_points - 2; i++ ) {
        bad_tri_indicator_arr[i] = false;
    }

    old_tri = 0;
    for ( p = 3; p < num_points; p++ ) {
        // printf("point : %ld \n", p);

        enclosing_tri = _walk(
            old_tri, p, gv, vertices_ID, neighbour_ID, points, global_arr);
        // printf("_walk done\n");

        _identify_cavity(
            enclosing_tri, p, gv, vertices_ID, neighbour_ID, points,
            bad_tri_indicator_arr, arr_sizes, arr_pointers, global_arr);
        // printf("_identify_cavity done\n");

        old_tri = _make_Delaunay_ball(
            p, vertices_ID, neighbour_ID, arr_sizes, arr_pointers,
            num_entities);
        // printf("_make_Delaunay_ball done\n");

        bad_tri = arr_pointers[0];
        bt_end = arr_sizes[2];
        for ( i = 0; i < bt_end; i++ ){
            t = bad_tri[i];
            bad_tri_indicator_arr[t] = false;
        }
    }

    // free(arr_pointers[0]);
    // free(arr_pointers[1]);
    // free(arr_pointers[2]);
    // free(points);
    // free(global_arr);
    aligned_free(arr_pointers[0]);
    aligned_free(arr_pointers[1]);
    aligned_free(arr_pointers[2]);
    aligned_free(points);
    aligned_free(global_arr);

    return;
}
*/

#endif /* BASIC_TOOLS_H */