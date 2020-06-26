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

#ifndef INT64
#define INT64 int64_t
#endif
#ifndef UINT64
#define UINT64 uint64_t
#endif

INT64 TEN = 10;

#define get_nth_digit(num, N, digit) \
    digit = ( num / ipow( TEN, (uint8_t) N) ) % 10


void print_arr(INT64 *arr, INT64 array_size) {
    printf("\n[");
    for ( INT64 i = 0; i < array_size - 1; ++i ) {
        printf("%ld, ", arr[i]);
    }
    printf("%ld]\n", arr[array_size - 1]);
}


void _radix_argsort(
        INT64 *points, INT64 *indices, INT64 array_size, INT64 highest_pos) {

    INT64 gi, NUM_THREADS, g_count;
    INT64 *g_hist, *g_head, *local_hist, *aux, *temp;

    g_hist = (INT64 *) aligned_malloc(10 * sizeof(INT64));
    for ( gi = 0; gi < 10; ++gi ) {
        g_hist[gi] = 0;
    }
    g_head = (INT64 *) aligned_malloc(10 * sizeof(INT64));
    aux = (INT64 *) aligned_malloc(array_size * sizeof(INT64));

    #pragma omp parallel
    {
        #pragma omp single
        {
            NUM_THREADS = omp_get_num_threads();
            local_hist = (INT64 *) aligned_malloc(10 * NUM_THREADS * sizeof(INT64));
        }

        INT64 tID, low, high, t_iter, digit, elem, i, level, pos;
        INT64 *running_sum = (INT64 *) aligned_malloc(10 * sizeof(INT64));

        tID = omp_get_thread_num();

        for ( level = 0; level <= highest_pos; ++level) {
            #pragma omp barrier

            /* building the local histogram */
            low = ( tID * array_size ) / NUM_THREADS;
            high = ( (tID + 1) * array_size ) / NUM_THREADS;
            for ( t_iter = 0; t_iter < 10; ++t_iter ) {
                local_hist[10*tID + t_iter] = 0;
            }
            for ( t_iter = low; t_iter < high; ++t_iter ) {
                elem = points[indices[t_iter]];
                get_nth_digit(elem, level, digit);
                local_hist[10*tID + digit] += 1;
            }
            #pragma omp barrier
            /* local histogram built */


            /* building global histogram */
            low = (tID * 10) / NUM_THREADS;
            high = ( (tID + 1) * 10 ) / NUM_THREADS;
            if ( low != high ) {
                for ( t_iter = low; t_iter < high; ++t_iter ) {
                    g_hist[t_iter] = 0;
                    for ( i = 0; i < NUM_THREADS; ++i ) {
                        g_hist[t_iter] += local_hist[10*i + t_iter];
                    }
                }
            }
            #pragma omp barrier

            if ( low != high ) {
                for ( t_iter = low; t_iter < high; ++t_iter ) {
                    g_head[t_iter] = 0;
                    for ( i = 0; i < t_iter; ++i ) {
                        g_head[t_iter] += g_hist[i];
                    }
                }
            }
            #pragma omp barrier
            /* global histogram built */

            for ( i = 0; i < 10; ++i ) {
                running_sum[i] = 0;
            }

            low = (tID * array_size) / NUM_THREADS;
            high = ((tID + 1) * array_size) / NUM_THREADS;
            if ( low != high ) {
                for ( t_iter = low; t_iter < high; ++t_iter ) {
                    elem = points[indices[t_iter]];
                    get_nth_digit(elem, level, digit);
                    pos = g_head[digit];
                    for ( i = 0; i < tID; ++i ) {
                        pos += local_hist[i*10 + digit];
                    }
                    pos += running_sum[digit];
                    running_sum[digit] += 1;

                    aux[pos] = indices[t_iter];
                }
            }
            #pragma omp barrier

            #pragma omp single
            {
                temp = indices;
                indices = aux;
                aux = temp;
            }
        }
    }

    if ( highest_pos % 2 == 0 ) {
        temp = indices;
        indices = aux;
        aux = temp;
    }
}


int main(int argc, char const *argv[])
{

    INT64 i, num_points = 64;
    INT64 *points, *indices;

    points = (INT64 *) aligned_malloc(num_points * sizeof(num_points));
    indices = (INT64 *) aligned_malloc(num_points * sizeof(num_points));

    srandom(time(NULL));

    for ( i = 0; i < num_points; ++i ) {
        points[i] = random() % num_points;
        indices[i] = i;
    }

    print_arr(indices, num_points);
    print_arr(points, num_points);

    INT64 highest_pos = 0, num;
    num = num_points;
    while (num / 10 != 0 ) {
        highest_pos += 1;
        num = num / 10;
    }
    printf("highest_pos : %ld \n", highest_pos);

    PARADIS(points, indices, num_points, highest_pos);

    print_arr(indices, num_points);
    printf("\n[");
    for ( INT64 i = 0; i < num_points - 1; ++i ) {
        printf("%ld, ", points[indices[i]]);
    }
    printf("%ld]\n", points[indices[num_points - 1]]);



    return 0;
}