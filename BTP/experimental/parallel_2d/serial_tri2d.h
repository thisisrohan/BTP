#ifndef SERIAL_TRI_H
#define SERIAL_TRI_H

#include <omp.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "predicates.h"
#include "BRIO_2D_serial.h"
#include "helpful_tools.h"
#include "basic_tools.h"
#include "parallel_tools.h"
#include "queue.h"

#ifndef INT64
#define INT64 int64_t
#endif
#ifndef UINT64
#define UINT64 uint64_t
#endif


void _assembly_serial(
        INT64* vertices_ID, INT64* neighbour_ID, double* original_points,
        INT64* num_entities, INT64* insertion_seq, INT64 gv) {

    INT64 i, p, num_points, bt_end, enclosing_tri, old_tri, t;
    INT64* bad_tri;
    num_points = num_entities[0];

    // double points[2*num_points];
    double* points;
    // points = (double *) malloc(2 * num_points * sizeof(double));
    points = (double *) aligned_malloc(2 * num_points * sizeof(double));
    // printf("yo\n");

    make_BRIO_serial(original_points, points, insertion_seq, num_points);
    // printf("BRIO made\n");

    exactinit2d(points, num_points);
    // printf("exactinit2d done\n");

    double* global_arr;
    // global_arr = (double *) malloc(3236 * sizeof(double));
    global_arr = (double *) aligned_malloc(3236 * sizeof(double));

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

    // bool bad_tri_indicator_arr[2*num_points - 2];
    bool* bad_tri_indicator_arr;
    bool *existing_tri = (bool *) aligned_malloc((2*num_points - 2) * sizeof(bool));
    bad_tri_indicator_arr = (bool *) malloc((2*num_points - 2) * sizeof(bool));
    for ( i = 0; i < 2*num_points - 2; i++ ) {
        existing_tri[i] = false;
        bad_tri_indicator_arr[i] = false;
    }

    old_tri = 0;
    for ( p = 3; p < num_points; p++ ) {
        // printf("point : %ld \n", p);

        enclosing_tri = _walk(
            old_tri, p, gv, vertices_ID, neighbour_ID, points, global_arr);
        if ( enclosing_tri == -2 ) {
            // aborting because `p` is a duplicate point
            // if ( existing_vtx[pg] == false ) { existing_vtx[pg] = true; }
            continue;
        }
        // printf("_walk done\n");

        _identify_cavity(
            enclosing_tri, p, gv, vertices_ID, neighbour_ID, points,
            bad_tri_indicator_arr, arr_sizes, arr_pointers, global_arr);
        // printf("_identify_cavity done\n");

        old_tri = _make_Delaunay_ball(
            p, vertices_ID, neighbour_ID, arr_sizes, arr_pointers,
            num_entities, existing_tri);
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

#endif /* SERIAL_TRI_H */