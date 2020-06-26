#include <omp.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "predicates.h"
#include "BRIO_2D.h"
#include "helpful_tools.h"
#include "basic_tools.h"
#include "parallel_tools.h"
#include "queue.h"
#include "serial_tri2d.h"

#ifndef INT64
#define INT64 int64_t
#endif
#ifndef UINT64
#define UINT64 uint64_t
#endif

INT64 MIN_NUM_PTS = 5000;


#define get_nth_digit(num, N, digit) \
    digit = ( num / ipow( 10, (uint8_t) N) ) % 10


void _assembly_parallel(
        INT64** mesh_arr, double* original_points, INT64* num_entities,
        INT64* insertion_seq, INT64 gv) {

    INT64 *vertices_ID, *neighbour_ID;
    vertices_ID = mesh_arr[0];
    neighbour_ID = mesh_arr[1];

    double start_g, end_g, cpu_time_g;

    INT64 num_points = num_entities[0];
    INT64 mesh_cap = num_entities[2];
    double *points = (double *) aligned_malloc(2 * num_points * sizeof(double));

    /* making BRIO */
    start_g = omp_get_wtime();
    INT64 *sfc_related_arr[2];
    INT64 h_deg = make_BRIO(
        original_points, points, insertion_seq, num_entities, sfc_related_arr);
    INT64 *boundary_indices = sfc_related_arr[0];
    INT64 *h_arr = sfc_related_arr[1];
    INT64 num_rounds = num_entities[3];
    end_g = omp_get_wtime();
    cpu_time_g = end_g - start_g;
    printf("BRIO time : %f s. \n", cpu_time_g);
    printf("[");
    for (INT64 jj = 0; jj < num_rounds - 1; ++jj ) {
        printf("%ld, ", boundary_indices[jj]);
    }
    printf("%ld]\n", boundary_indices[num_rounds - 1]);
    /* BRIO made */

    /* initializing the triangulation */
    double *global_arr_g = (double *) aligned_malloc(3236 * sizeof(double));
    _initialize(
        vertices_ID, neighbour_ID, points, num_entities, insertion_seq, gv,
        global_arr_g);
    /* triangulation initialized */

    /* building helper arrays */
    bool *existing_tri = (bool *) aligned_malloc(mesh_cap * sizeof(bool));
    bool *existing_vtx = (bool *) aligned_malloc(num_points * sizeof(bool));
    bool *bad_tri_indicator_arr = (bool *) aligned_malloc(
        mesh_cap * sizeof(bool));
    INT64 ig;
    for ( ig = 0; ig < mesh_cap; ++ig ) {
        existing_tri[ig] = false;
        bad_tri_indicator_arr[ig] = false;
        if ( ig < num_points ) { existing_vtx[ig] = false; }
    }
    existing_tri[0] = true;
    existing_tri[1] = true;
    existing_tri[2] = true;
    existing_tri[3] = true;

    existing_vtx[0] = true;
    existing_vtx[1] = true;
    existing_vtx[2] = true;
    /* helper arrays built */

    /* finding base_mesh_idx */
    INT64 base_mesh_idx = 1;
    // INT64 min_points = num_points / 20;
    INT64 min_points = 2000;
    if ( num_points < min_points ) {
        min_points = num_points / 20;
    }
    // INT64 min_points = MIN_NUM_PTS;
    // if (min_points < 3) { min_points = 3; }
    // INT64 min_points = 3;
    while ( boundary_indices[base_mesh_idx] < min_points ) {
        base_mesh_idx++;
    }
    printf("%ld \n", boundary_indices[base_mesh_idx]);
    /* base_mesh_idx found */

    /* building the base mesh */
    INT64 arr_sizes_g[4];
    INT64 *arr_pointers_g[3];
    arr_pointers_g[0] = (INT64 *) aligned_malloc(64 * sizeof(INT64));
    arr_pointers_g[1] = (INT64 *) aligned_malloc(64 * sizeof(INT64));
    arr_pointers_g[2] = (INT64 *) aligned_malloc(2 * 64 * sizeof(INT64));
    arr_sizes_g[0] = 64;
    arr_sizes_g[1] = 64;
    INT64 pg, enclosing_tri, old_tri, tg, bt_end_g;
    INT64 *bad_tri_g;
    old_tri = 0;

    start_g = omp_get_wtime();
    for ( pg = 0; pg < boundary_indices[base_mesh_idx]; ++pg ) {
        enclosing_tri = _walk(
            old_tri, pg, gv, vertices_ID, neighbour_ID, points, global_arr_g);
        if ( enclosing_tri == -2 ) {
            // aborting because `pg` is a duplicate point
            if ( existing_vtx[pg] == false ) { existing_vtx[pg] = true; }
            continue;
        }

        _identify_cavity(
            enclosing_tri, pg, gv, vertices_ID, neighbour_ID, points,
            bad_tri_indicator_arr, arr_sizes_g, arr_pointers_g, global_arr_g);

        _make_Delaunay_ball(
            pg, vertices_ID, neighbour_ID, arr_sizes_g, arr_pointers_g,
            num_entities, existing_tri);

        bad_tri_g = arr_pointers_g[0];
        bt_end_g = arr_sizes_g[2];
        for ( ig = 0; ig < bt_end_g; ++ig ) {
            tg = bad_tri_g[ig];
            bad_tri_indicator_arr[tg] = false;
        }

        existing_vtx[pg] = true;
    }
    end_g = omp_get_wtime();
    cpu_time_g = end_g - start_g;
    printf("base mesh time : %f s.\n", cpu_time_g);

    aligned_free(arr_pointers_g[0]);
    aligned_free(arr_pointers_g[1]);
    aligned_free(arr_pointers_g[2]);
    aligned_free(global_arr_g);
    /* base mesh built */
    // printf("base mesh built \n");

    /* defining some shared variables for the parallel region */
    INT64 NUM_THREADS, num_threads, b1, b2, NUM_POINTS_ri, NUM_POINTS_INSERTED;
    INT64 a, num_new_threads = 0, num_tri, random_num, SUCCESSFUL_INSERTIONS;
    INT64 *tri_indices, *INT64_arr, *h_index, *indices, *g_hist, *g_head;
    INT64 *local_hist, *aux, *temp_arr;
    double *max_x, *max_y, *min_x, *min_y;
    double gxmax, gxmin, gymax, gymin, gxymax, rho = 1.0;
    char *vtx_color, *tri_color;

    h_index = (INT64 *) aligned_malloc(num_points * sizeof(INT64));
    indices = (INT64 *) aligned_malloc(num_points * sizeof(INT64));
    vtx_color = (char *) aligned_malloc(num_points * sizeof(char));
    tri_color = (char *) aligned_malloc(mesh_cap * sizeof(char));
    a = 1 << h_deg;
    g_hist = (INT64 *) aligned_malloc(10 * sizeof(INT64));
    for ( ig = 0; ig < 10; ++ig ) {
        g_hist[ig] = 0;
    }
    g_head = (INT64 *) aligned_malloc(10 * sizeof(INT64));
    aux = (INT64 *) aligned_malloc(num_points * sizeof(INT64));

    double start, end, cpu_time;

    INT64 highest_pos = 0, num;
    num = 2 * a * a;
    while ( num / 10 != 0 ) {
        highest_pos += 1;
        num = num / 10;
    }
    // printf("a_sq : %ld, highest_pos : %ld \n", a*a, highest_pos);
    /* shared variables definition complete */

    /* starting meshing in parallel */
    start_g = omp_get_wtime();
    #pragma omp parallel
    {
        /* initializing a bunch of shared stuff */
        #pragma omp single
        {
            NUM_THREADS = omp_get_num_threads();
            max_x = (double *) aligned_malloc(NUM_THREADS * sizeof(double));
            max_y = (double *) aligned_malloc(NUM_THREADS * sizeof(double));
            min_x = (double *) aligned_malloc(NUM_THREADS * sizeof(double));
            min_y = (double *) aligned_malloc(NUM_THREADS * sizeof(double));
            tri_indices = (INT64 *) aligned_malloc(
                (NUM_THREADS + 1) * sizeof(INT64));
            INT64_arr = (INT64 *) aligned_malloc(NUM_THREADS * sizeof(INT64));
            local_hist = (INT64 *) aligned_malloc(10 * NUM_THREADS * sizeof(INT64));
        }
        /* some shared stuff initialized */

        /* initializing some thread private stuff */
        INT64 tID, xi, yi, a_sq, low, high, t_iter, enclosing_tri = 0, old_tri = 0, p = 0;
        INT64 ri, nt, step, hi, last_idx, num_points_left = 0, temp = 0, ip = 0, bt_end = 0;
        INT64 boundary_end = 0, num_tri_to_create = 0, num_new_tri_needed = 0, t_low = 0, t_high = 0;
        INT64 r_low, r_high, elem, digit, level, pos;
        double pxmin, pxmax, pymin, pymax, x, y;

        tID = omp_get_thread_num();
        a_sq = a * a;

        INT64 **arr_pointers;
        INT64 *arr_sizes, *q_params, *tri_queue, *bad_tri, *running_sum;
        double *global_arr;
        #pragma omp critical (MEM_ALLOC)
        {
            arr_pointers = (INT64 **) aligned_malloc(4 * sizeof(INT64 *));
            arr_sizes = (INT64 *) aligned_malloc(4 * sizeof(INT64));
            q_params = (INT64 *) aligned_malloc(4 * sizeof(INT64));
            global_arr = (double *) aligned_malloc(3236 * sizeof(double));

            arr_pointers[0] = (INT64 *) aligned_malloc(64 * sizeof(INT64));
            arr_pointers[1] = (INT64 *) aligned_malloc(64 * sizeof(INT64));
            arr_pointers[2] = (INT64 *) aligned_malloc(2 * 64 * sizeof(INT64));
            arr_pointers[3] = (INT64 *) aligned_malloc(8192 * sizeof(INT64));

            running_sum = (INT64 *) aligned_malloc(10 * sizeof(INT64));
        }

        arr_sizes[0] = 64;
        arr_sizes[1] = 64;

        tri_queue = arr_pointers[3];
        tri_queue[0] = -1;
        q_params[0] = 0;
        q_params[1] = 0;
        q_params[2] = 8192;
        q_params[3] = 0;
        /* some thread private stuff initialized */

        // #pragma omp barrier
        // low = (tID * num_points) / NUM_THREADS;
        // high = ( (tID + 1) * num_points ) / NUM_THREADS;
        // pxmin = points[2*low];
        // pxmax = points[2*low];
        // pymin = points[2*low + 1];
        // pymax = points[2*low + 1];

        // for ( t_iter = low; t_iter < high; ++t_iter ) {
        //     x = points[2*t_iter];
        //     y = points[2*t_iter + 1];
            
        //     if ( x > pxmax ) { pxmax = x; }
        //     else if ( x < pxmin ) { pxmin = x; }

        //     if ( y > pymax ) { pymax = y; }
        //     else if ( y < pymin ) { pymin = y; }
        // }
        // max_x[tID] = pxmax;
        // min_x[tID] = pxmin;
        // max_y[tID] = pymax;
        // min_y[tID] = pymin;
        // #pragma omp barrier

        // #pragma omp single
        // {
        //     gxmax = max_x[0];
        //     gxmin = min_x[0];
        //     gymax = max_y[0];
        //     gymin = min_y[0];
        //     for ( t_iter = 0; t_iter < NUM_THREADS; ++t_iter) {
        //         if ( max_x[t_iter] > gxmax ) {gxmax = max_x[t_iter];}
        //         if ( max_y[t_iter] > gymax ) {gymax = max_y[t_iter];}
        //         if ( min_x[t_iter] < gxmin ) {gxmin = min_x[t_iter];}
        //         if ( min_y[t_iter] < gymin ) {gymin = min_y[t_iter];}
        //     }

        //     gxmax -= gxmin;
        //     gymax -= gymin;

        //     if ( gxmax > gymax ) {gxymax = gxmax;}
        //     else {gxymax = gymax;}
        // }


        for ( ri = base_mesh_idx; ri < num_rounds; ri++ ) {
            #pragma omp barrier
            /* initializing some shared stuff for this round */
            #pragma omp single
            {
                b1 = boundary_indices[ri];
                b2 = boundary_indices[ri + 1];
                NUM_POINTS_ri = b2 - b1;

                num_threads = NUM_THREADS;
                if ( NUM_POINTS_ri > MIN_NUM_PTS ) {
                    while ( NUM_POINTS_ri / num_threads < MIN_NUM_PTS ) {
                        num_threads /= 2;
                    }
                } else {
                    num_threads = 1;
                }
                // if ( (NUM_POINTS_ri / num_threads < MIN_NUM_PTS) || (rho < 0.2) || (rho < 1.0/((double) num_threads)) ) {
                //     num_threads /= 2;
                // }
                if ( (num_threads > 1) && ((rho < 0.2) || (rho < 1.0/((double) num_threads))) ) {
                    num_threads /= 2;
                }

                NUM_POINTS_INSERTED = 0;
                num_new_threads = 0;
                num_tri = num_entities[1];
                mesh_cap = num_entities[2];
            }
            /* round specific shared stuff initialized */

            /* initializing `tri_queue` if it hasn't already been done so */
            INT64 tID_l;
            if ( (tID < num_threads) && (tri_queue[0] == -1) ) {
                #pragma omp atomic capture
                {
                    tID_l = num_new_threads;
                    num_new_threads += 1;
                }
            }
            #pragma omp barrier

            INT64 q_head, q_tail, q_cap, q_num_items;
            if ( (tID < num_threads) && (tri_queue[0] == -1) ) {
                if ( (mesh_cap - num_tri) / num_new_threads < 8192 ) {
                    low = num_tri + (tID_l * (mesh_cap - num_tri)) / num_new_threads;
                    high = num_tri + ((tID_l + 1) * (mesh_cap - num_tri)) / num_new_threads;
                } else {
                    low = num_tri + tID_l * 8192;
                    high = num_tri + (tID_l + 1) * 8192;
                }

                // printf("\n tID : %ld, low : %ld, high : %ld \n \n", tID, low, high);
                q_tail = q_params[1];
                q_cap = q_params[2];
                for ( t_iter = low; t_iter < high; ++t_iter ) {
                    tri_queue[q_tail++] = t_iter;
                }
                q_params[1] = q_tail % q_cap;
                q_params[3] = high - low;

                #pragma omp atomic update
                num_entities[1] += q_params[3];
            }
            #pragma omp barrier
            /* `tri_queue` initialization complete */

            /* inserting into the mesh the points in this round */
            while ( NUM_POINTS_INSERTED < NUM_POINTS_ri ) {

                /* assigning color to vertices present in the current mesh */
                low = ( tID * b2 ) / NUM_THREADS;
                high = ( (tID + 1) * b2 ) / NUM_THREADS;

                if ( NUM_POINTS_INSERTED == 0 ) {
                    // this means this is the first attempt
                    /* setting the values for gxmin, gymin, gxymax */
                    pxmin = points[2*low];
                    pxmax = points[2*low];
                    pymin = points[2*low + 1];
                    pymax = points[2*low + 1];

                    for ( t_iter = low; t_iter < high; ++t_iter ) {
                        x = points[2*t_iter];
                        y = points[2*t_iter + 1];
                        
                        if ( x > pxmax ) { pxmax = x; }
                        else if ( x < pxmin ) { pxmin = x; }

                        if ( y > pymax ) { pymax = y; }
                        else if ( y < pymin ) { pymin = y; }
                    }
                    max_x[tID] = pxmax;
                    min_x[tID] = pxmin;
                    max_y[tID] = pymax;
                    min_y[tID] = pymin;
                    #pragma omp barrier

                    nt = (NUM_THREADS + 1) / 2;
                    step = 1;
                    while ( nt >= 1 ) {
                        if ( (tID < nt) && ((2*tID + 1)*step < NUM_THREADS) ) {
                            // reduction on max_x
                            if (max_x[2*tID*step] < max_x[(2*tID + 1)*step]) {
                                max_x[2*tID*step] = max_x[(2*tID + 1)*step];
                            }
                            // reduction on min_x
                            if (min_x[2*tID*step] > min_x[(2*tID + 1)*step]) {
                                min_x[2*tID*step] = min_x[(2*tID + 1)*step];
                            }
                            // reduction on max_y
                            if (max_y[2*tID*step] < max_y[(2*tID + 1)*step]) {
                                max_y[2*tID*step] = max_y[(2*tID + 1)*step];
                            }
                            // reduction on min_y
                            if (min_y[2*tID*step] > min_y[(2*tID + 1)*step]) {
                                min_y[2*tID*step] = min_y[(2*tID + 1)*step];
                            }
                        }
                        #pragma omp barrier
                        if ( nt == 1 ) break;
                        nt = (nt + 1) / 2;
                        step *= 2;
                    }

                    #pragma omp single
                    {
                        gxmax = max_x[0];
                        gxmin = min_x[0];
                        gymax = max_y[0];
                        gymin = min_y[0];
                        for ( t_iter = 0; t_iter < NUM_THREADS; ++t_iter) {
                            if ( max_x[t_iter] > gxmax ) {gxmax = max_x[t_iter];}
                            if ( max_y[t_iter] > gymax ) {gymax = max_y[t_iter];}
                            if ( min_x[t_iter] < gxmin ) {gxmin = min_x[t_iter];}
                            if ( min_y[t_iter] < gymin ) {gymin = min_y[t_iter];}
                        }

                        gxmax -= gxmin;
                        gymax -= gymin;

                        if ( gxmax > gymax ) {gxymax = gxmax;}
                        else {gxymax = gymax;}

                        // random_num = randomnation(0, a_sq);
                        random_num = 0;
                        num_tri = num_entities[1];
                    }
                    /* gxmin, gymin, gxymax found */
                } else {
                    #pragma omp single
                    {
                        // num_threads = NUM_THREADS;
                        num_points_left = NUM_POINTS_ri - NUM_POINTS_INSERTED;

                        if ( num_points_left > MIN_NUM_PTS ) {
                            while ( num_points_left / num_threads < MIN_NUM_PTS ) {
                                num_threads /= 2;
                            }
                        } else {
                            num_threads = 1;
                        }
                        if ( (num_threads > 1) && ((rho < 0.25) || (rho < 1.0/((double) num_threads))) ) {
                            num_threads /= 2;
                        }
                        // if ( (num_points_left / num_threads < MIN_NUM_PTS) || (rho < 0.2) || (rho < 1.0/((double) num_threads)) ) {
                        //     num_threads /= 2;
                        // }

                        random_num = randomnation(0, a_sq);
                        // random_num = random() % a_sq;
                        num_tri = num_entities[1];
                    }
                }

                /* ordering the points along a Moore SFC */
                // these may have been changed down the road, need to redefine
                low = ( tID * b2 ) / NUM_THREADS;
                high = ( (tID + 1) * b2 ) / NUM_THREADS;
                for ( t_iter = low; t_iter < high; ++t_iter ) {
                    x = points[2*t_iter];
                    y = points[2*t_iter + 1];
                    x = ((x - gxmin) / gxymax)*(a - 1) + 0.5;
                    y = ((y - gymin) / gxymax)*(a - 1) + 0.5;
                    xi = (INT64) x;
                    yi = (INT64) y;

                    // /* testing purposes */
                    // if ( xi >= a ) printf("xi : %ld, x : %f, points[2*t_iter] : %f, tID : %ld\n", xi, x, points[2*t_iter], tID);
                    // if ( yi >= a ) printf("yi : %ld, y : %f, points[2*t_iter+1] : %f, tID : %ld\n", yi, y, points[2*t_iter+1], tID);
                    // // /* ---------------- */

                    hi = h_arr[xi + a*yi] - random_num;
                    if ( hi < 0 ) {hi += a_sq;}

                    if ( existing_vtx[t_iter] == true ) { h_index[t_iter] = hi; }
                    else { h_index[t_iter] = hi + a_sq; }

                    indices[t_iter] = t_iter;
                }
                #pragma omp barrier

                /* radix sort */
                // #pragma omp single
                // {
                //     start = omp_get_wtime();
                // }
                for ( level = 0; level <= highest_pos; ++level) {
                    // #pragma omp single
                    // {
                    //     printf("level : %ld \n", level);
                    // }

                    /* building the local histogram */
                    r_low = ( tID * b2 ) / NUM_THREADS;
                    r_high = ( (tID + 1) * b2 ) / NUM_THREADS;
                    for ( t_iter = 0; t_iter < 10; ++t_iter ) {
                        local_hist[10*tID + t_iter] = 0;
                    }
                    for ( t_iter = r_low; t_iter < r_high; ++t_iter ) {
                        elem = h_index[indices[t_iter]];
                        get_nth_digit(elem, level, digit);
                        local_hist[10*tID + digit] += 1;
                    }
                    // printf("local_hist built for tID : %ld \n", tID);
                    #pragma omp barrier
                    /* local histogram built */


                    /* building global histogram */
                    r_low = (tID * 10) / NUM_THREADS;
                    r_high = ( (tID + 1) * 10 ) / NUM_THREADS;
                    if ( r_low != r_high ) {
                        for ( t_iter = r_low; t_iter < r_high; ++t_iter ) {
                            g_hist[t_iter] = 0;
                            for ( ip = 0; ip < NUM_THREADS; ++ip ) {
                                g_hist[t_iter] += local_hist[10*ip + t_iter];
                            }
                        }
                    }
                    // printf("g_hist built for tID : %ld \n", tID);
                    #pragma omp barrier

                    if ( r_low != r_high ) {
                        for ( t_iter = r_low; t_iter < r_high; ++t_iter ) {
                            g_head[t_iter] = 0;
                            for ( ip = 0; ip < t_iter; ++ip ) {
                                g_head[t_iter] += g_hist[ip];
                            }
                        }
                    }
                    // printf("g_head built for tID : %ld \n", tID);
                    #pragma omp barrier
                    /* global histogram built */

                    for ( t_iter = 0; t_iter < 10; ++t_iter ) {
                        running_sum[t_iter] = 0;
                    }
                    // printf("running_sum set for tID : %ld \n", tID);

                    r_low = (tID * b2) / NUM_THREADS;
                    r_high = ((tID + 1) * b2) / NUM_THREADS;
                    if ( r_low != r_high ) {
                        for ( t_iter = r_low; t_iter < r_high; ++t_iter ) {
                            elem = h_index[indices[t_iter]];
                            get_nth_digit(elem, level, digit);
                            pos = g_head[digit];
                            for ( ip = 0; ip < tID; ++ip ) {
                                pos += local_hist[ip*10 + digit];
                            }
                            pos += running_sum[digit];
                            running_sum[digit] += 1;

                            aux[pos] = indices[t_iter];
                        }
                    }
                    #pragma omp barrier

                    #pragma omp single
                    {
                        temp_arr = indices;
                        indices = aux;
                        aux = temp_arr;
                        // printf("sorting done for level : %ld \n", level);
                    }
                }
                // if ( highest_pos % 2 == 1 ) {
                //     #pragma omp single
                //     {
                //         temp_arr = indices;
                //         indices = aux;
                //         aux = temp_arr;
                //     }
                // }
                // #pragma omp single
                // {
                //     end = omp_get_wtime();
                //     cpu_time = end - start;
                //     printf("\n radix_sort time : %f s. \n \n", cpu_time);
                // }
                // #pragma omp single
                // {
                //     printf("[");
                //     for ( t_iter = 0; t_iter < b2 - 1; ++t_iter ) {
                //         printf("%ld, ", h_index[indices[t_iter]]);
                //     }
                //     printf("%ld]\n", h_index[indices[b2 - 1]]);
                // }
                /* radix sort complete */
                /* sorted along SFC */

                last_idx = b1 + NUM_POINTS_INSERTED;
                if ( tID < num_threads ) {
                    low = (tID * last_idx) / num_threads;
                    high = ((tID + 1) * last_idx) / num_threads;
                    for ( t_iter = low; t_iter < high; ++t_iter ) {
                        vtx_color[indices[t_iter]] = (char) tID;
                    }
                }
                /* colors assigned to vertices present in the current mesh */

                /* assigning colors to the vertices that are to be inserted */
                if ( tID < num_threads ) {
                    num_points_left = NUM_POINTS_ri - NUM_POINTS_INSERTED;
                    low = last_idx + (tID * num_points_left) / num_threads;
                    high = last_idx + ((tID + 1) * num_points_left) / num_threads;
                    for ( t_iter = low; t_iter < high; ++t_iter) {
                        vtx_color[indices[t_iter]] = (char) tID;
                    }
                }
                // #pragma omp barrier
                /* colors assigned to vertices that are to be inserted */

                /* assigning colors to triangles present in the mesh */
                low = (tID * num_tri) / NUM_THREADS;
                high = ((tID + 1) * num_tri) / NUM_THREADS;
                INT64_arr[tID] = 0;
                for ( t_iter = low; t_iter < high; ++t_iter ) {
                    if ( existing_tri[t_iter] == true ) {INT64_arr[tID] += 1;}
                }
                #pragma omp barrier

                // nt = (NUM_THREADS + 1) / 2;
                // step = 1;
                // while (nt >= 1) {
                //     if ( (tID < nt) && ((2*tID + 1)*step < NUM_THREADS) ) {
                //         INT64_arr[2*tID*step] += INT64_arr[(2*tID + 1)*step];
                //     }
                //     #pragma omp barrier
                //     if ( nt == 1 ) {break;}
                //     nt = (nt + 1) / 2;
                //     step *= 2;
                // }

                #pragma omp single
                {
                    // num_tri = INT64_arr[0];
                    num_tri = 0;
                    for ( t_iter = 0; t_iter < NUM_THREADS; ++t_iter ) {
                        num_tri += INT64_arr[t_iter];
                    }
                    tri_indices[0] = 0;
                    for ( t_iter = 0; t_iter < NUM_THREADS; ++t_iter ) {
                        low = (t_iter * num_tri) / NUM_THREADS;
                        high = ((t_iter + 1) * num_tri) / NUM_THREADS;
                        for ( temp = tri_indices[t_iter]; temp < mesh_cap; ++temp ) {
                            if ( existing_tri[temp] == true ) {++low;}
                            if  ( low == high ) {
                                tri_indices[t_iter + 1] = temp + 1;
                                break;
                            }
                        }
                    }
                }

                low = tri_indices[tID];
                high = tri_indices[tID + 1];
                INT64 v1, v2, v3;
                char c1, c2, c3;
                for ( t_iter = low; t_iter < high; ++t_iter ) {
                    if ( existing_tri[t_iter] == true ) {
                        v1 = vertices_ID[3*t_iter];
                        v2 = vertices_ID[3*t_iter + 1];
                        v3 = vertices_ID[3*t_iter + 2];
                        c1 = vtx_color[v1];
                        if ( v2 == gv ) {
                            c3 = vtx_color[v3];
                            if ( c1 == c3 ) {tri_color[t_iter] = c1;}
                            else {tri_color[t_iter] = (char) NUM_THREADS;}
                        } else if ( v3 == gv ) {
                            c2 = vtx_color[v2];
                            if ( c1 == c2 ) {tri_color[t_iter] = c1;}
                            else {tri_color[t_iter] = (char) NUM_THREADS;}
                        } else {
                            c2 = vtx_color[v2];
                            c3 = vtx_color[v3];
                            if ( (c1 == c2) && (c2 == c3) ) {
                                tri_color[t_iter] = c1;
                            } else {
                                tri_color[t_iter] = (char) NUM_THREADS;
                            }
                        }
                    }
                }
                #pragma omp barrier
                /* colors assigned to triangles present in the current mesh */

                /* attempting to insert the vertices */
                INT64_arr[tID] = 0;
                if ( tID < num_threads ) {
                    // finding the triangle to start the walk from
                    for ( ip = 0; ip < mesh_cap; ++ip ) {
                        if ( existing_tri[ip] == true ) {
                            if ( tri_color[ip] == (char) tID ) {
                                old_tri = ip;
                                break;
                            }
                        }
                    }

                    num_points_left = NUM_POINTS_ri - NUM_POINTS_INSERTED;
                    low = last_idx + (tID * num_points_left) / num_threads;
                    high = last_idx + ((tID + 1) * num_points_left) / num_threads;

                    for ( ip = low; ip < high; ++ip ) {
                        p = indices[ip];

                        // printf("tID : %ld, ip : %ld, p : %ld \n", tID, ip, p);
                        enclosing_tri = _walk_parallel(
                            old_tri, p, gv, vertices_ID, neighbour_ID, points,
                            global_arr, tri_color, tID);
                        if ( enclosing_tri == -1 ) {
                            // walked into a foreign triangle
                            // printf("tID : %ld, ip : %ld, p : %ld aborted cuz walk \n", tID, ip, p);
                            continue;
                        }
                        if ( enclosing_tri == -2 ) {
                            // point already exists in the triangulation
                            // printf("tID : %ld, ip : %ld, p : %ld aborted cuz already exists \n", tID, ip, p);
                            existing_vtx[p] = true;
                            INT64_arr[tID] += 1;
                            continue;
                        }

                        enclosing_tri = _identify_cavity_parallel(
                            enclosing_tri, p, gv, vertices_ID, neighbour_ID,
                            points, bad_tri_indicator_arr, arr_sizes,
                            arr_pointers, global_arr, tri_color, tID);
                        if ( enclosing_tri == -1 ) {
                            // foreign triangle in cavity or boundary
                            // printf("tID : %ld, ip : %ld, p : %ld aborted cuz cavity \n", tID, ip, p);
                            bad_tri = arr_pointers[0];
                            bt_end = arr_sizes[2];
                            for ( t_iter = 0; t_iter < bt_end; ++t_iter ) {
                                temp = bad_tri[t_iter];
                                bad_tri_indicator_arr[temp] = false;
                            }
                            continue;
                        }

                        bt_end = arr_sizes[2];
                        boundary_end = arr_sizes[3];
                        num_tri_to_create = boundary_end - bt_end;
                        q_cap = q_params[2];
                        q_num_items = q_params[3];
                        if ( num_tri_to_create > q_num_items ) {
                            // request more triangles from the global arrays
                            // printf("\n ASKING FOR MORE TRI \n \n");
                            if ( 8192 > num_tri_to_create ) {
                                num_new_tri_needed = 8192 - q_num_items;
                            } else {
                                num_new_tri_needed = num_tri_to_create - q_num_items;
                            }

                            #pragma omp critical (MEM_ALLOC)
                            {
                                t_low = num_entities[1];
                                num_entities[1] += num_new_tri_needed;
                                t_high = num_entities[1];
                                mesh_cap = num_entities[2];

                                // printf("tID : %ld, t_low : %ld, t_high : %ld, mesh_cap : %ld \n", tID, t_low, t_high, mesh_cap);

                                if ( t_high > mesh_cap ) {
                                    // printf("\n RE-ALLOCATING \n \n");
                                    mesh_cap = 2*t_high;

                                    vertices_ID = (INT64 *) aligned_realloc(
                                        vertices_ID, 3 * mesh_cap * sizeof(INT64));
                                    // printf("vertices_ID done\n");

                                    neighbour_ID = (INT64 *) aligned_realloc(
                                        neighbour_ID, 3 * mesh_cap * sizeof(INT64));
                                    // printf("neighbour_ID done\n");

                                    tri_color = (char *) aligned_realloc(
                                        tri_color, mesh_cap * sizeof(char));
                                    // printf("tri_color done\n");

                                    existing_tri = (bool *) aligned_realloc(
                                        existing_tri, mesh_cap * sizeof(bool));
                                    // printf("existing_tri done\n");

                                    bad_tri_indicator_arr = (bool *) aligned_realloc(
                                        bad_tri_indicator_arr, mesh_cap * sizeof(bool));
                                    // printf("bad_tri_indicator_arr done\n");

                                    for ( t_iter = t_low; t_iter < mesh_cap; ++t_iter ) {
                                        existing_tri[t_iter] = false;
                                        bad_tri_indicator_arr[t_iter] = false;
                                    }

                                    num_entities[2] = mesh_cap;
                                }

                                if ( num_new_tri_needed + q_num_items > q_cap ) {
                                    // printf("\n EXPANDING tri_queue \n \n");
                                    q_cap = num_new_tri_needed + q_num_items;
                                    tri_queue = (INT64 *) aligned_realloc(
                                        tri_queue, q_cap * sizeof(INT64));
                                    arr_pointers[3] = tri_queue;
                                    q_params[2] = q_cap;
                                }
                            }

                            q_head = q_params[0];
                            q_tail = q_params[1];
                            for ( t_iter = t_low; t_iter < t_high; ++t_iter ) {
                                tri_queue[q_tail++] = t_iter;
                                if ( q_tail >= q_cap ) {q_tail -= q_cap;}
                            }
                            if ( q_tail == q_head ) {q_num_items = q_cap;}
                            else if ( q_tail > q_head ) {q_num_items = q_tail - q_head;}
                            else {q_num_items = (q_cap - q_head) + q_tail;}
                            q_params[1] = q_tail;
                            q_params[3] = q_num_items;
                        }

                        old_tri = _make_Delaunay_ball_parallel(
                            p, vertices_ID, neighbour_ID, arr_sizes,
                            arr_pointers, num_entities, existing_tri, q_params,
                            tID, tri_color);

                        bad_tri = arr_pointers[0];
                        bt_end = arr_sizes[2];
                        for ( t_iter = 0; t_iter < bt_end; ++t_iter ) {
                            temp = bad_tri[t_iter];
                            bad_tri_indicator_arr[temp] = false;
                        }
                        existing_vtx[p] = true;
                        INT64_arr[tID] += 1;
                        // printf("tID : %ld, ip : %ld, p : %ld inserted \n", tID, ip, p);
                    }
                }
                #pragma omp barrier

                // nt = (num_threads + 1) / 2;
                // step = 1;
                // while ( nt >= 1 ) {
                //     if ( (tID < nt) && ((2*tID + 1)*step < num_threads) ) {
                //         INT64_arr[2*tID*step] += INT64_arr[(2*tID + 1)*step];
                //     }
                //     #pragma omp barrier
                //     if ( nt == 1 ) {break;}
                //     nt = (nt + 1) / 2;
                //     step *= 2;
                // }

                #pragma omp single
                {
                    // printf("\n tID : %ld, ri : %ld \n \n", tID, ri);
                    SUCCESSFUL_INSERTIONS = 0;
                    for ( t_iter = 0; t_iter < num_threads; ++t_iter) {
                        SUCCESSFUL_INSERTIONS += INT64_arr[t_iter];
                    }
                    rho = ((double) SUCCESSFUL_INSERTIONS) / ((double) (NUM_POINTS_ri - NUM_POINTS_INSERTED));
                    // NUM_POINTS_INSERTED += SUCCESSFUL_INSERTIONS;
                    // NUM_POINTS_INSERTED += INT64_arr[0];
                    printf("\n round : %ld | num_active_threads : %ld | %ld / %ld | %ld / %ld | rho : %f\n", ri + 1, num_threads, SUCCESSFUL_INSERTIONS, NUM_POINTS_ri - NUM_POINTS_INSERTED, NUM_POINTS_INSERTED + SUCCESSFUL_INSERTIONS, NUM_POINTS_ri, rho);
                    NUM_POINTS_INSERTED += SUCCESSFUL_INSERTIONS;
                }
            }
            // #pragma omp barier
        }
    }
    end_g = omp_get_wtime();
    cpu_time_g = end_g - start_g;
    printf("parallel meshing time : %f s. \n", cpu_time_g);
    /* parallel meshing complete */
}






int main(int argc, char const *argv[])
{
    INT64 num_points, num_tri, gv, i;
    num_points = 10000000;
    gv = num_points;
    num_tri = 3*num_points;

    srandom(time(NULL));

    // double original_points[2*num_points];
    double* original_points;
    // original_points = (double *) malloc(2 * num_points * sizeof(double));
    original_points = (double *) aligned_malloc(2 * num_points * sizeof(double));
    printf("yo yo 1 of 5\n");

    // long vertices_ID[3*num_tri];
    INT64* vertices_ID;
    // vertices_ID = (INT64 *) malloc(3 * num_tri * sizeof(INT64));
    vertices_ID = (INT64 *) aligned_malloc(3 * num_tri * sizeof(INT64));
    printf("yo yo 2 of 5\n");

    // long neighbour_ID[3*num_tri];
    INT64* neighbour_ID;
    // neighbour_ID = (INT64 *) malloc(3 * num_tri * sizeof(INT64));
    neighbour_ID = (INT64 *) aligned_malloc(3 * num_tri * sizeof(INT64));
    printf("yo yo 3 of 5\n");

    // long insertion_seq[num_points];
    INT64* insertion_seq;
    // insertion_seq = (INT64 *) malloc(num_points * sizeof(INT64));
    insertion_seq = (INT64 *) aligned_malloc(num_points * sizeof(INT64));
    printf("yo yo 4 of 5\n");

    INT64 num_entities[4];
    printf("yo yo 5 of 5\n");

    num_entities[0] = num_points;
    num_entities[1] = 0;
    num_entities[2] = num_tri;

    INT64* mesh_arr[2];
    mesh_arr[0] = vertices_ID;
    mesh_arr[1] = neighbour_ID;

    for ( i = 0; i < num_points; ++i ) {
        original_points[2*i] = uniformdoublerand();
        original_points[2*i + 1] = uniformdoublerand();
    }

    // clock_t start, end;
    // size_t start, end;
    double start, end;
    double cpu_time;

    printf("all arrays initialized\n");
    // start = clock();
    // start = time(NULL);
    start = omp_get_wtime();
    _assembly_parallel(
        mesh_arr, original_points, num_entities, insertion_seq, gv);
    // end = clock();
    // end = time(NULL);
    end = omp_get_wtime();
    printf("triangulation made\n");
    cpu_time = (double) (end - start);
    printf("time taken : %f\n", cpu_time);

    start = omp_get_wtime();
    _assembly_serial(
        vertices_ID, neighbour_ID, original_points, num_entities, insertion_seq, gv);
    // end = clock();
    // end = time(NULL);
    end = omp_get_wtime();
    printf("triangulation made\n");

    // cpu_time = ((double) (end - start))/CLOCKS_PER_SEC;
    cpu_time = (double) (end - start);
    printf("time taken : %f\n", cpu_time);

    return 0;
}