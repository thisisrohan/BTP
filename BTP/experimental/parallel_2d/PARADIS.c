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
    digit = ( num / ipow( TEN, (uint8_t) N) ) % TEN


void print_arr(INT64 *arr, INT64 array_size) {
    printf("\n[");
    for ( INT64 i = 0; i < array_size - 1; ++i ) {
        printf("%ld, ", arr[i]);
    }
    printf("%ld]\n", arr[array_size - 1]);
}


void PARADIS_permute(
        INT64 *points, INT64 *indices, INT64 *pp_head, INT64 *pp_tail,
        INT64 tID, INT64 level, INT64 NUM_THREADS) {

    INT64 i, head, v, k, vi_idx, temp, offset;
    offset = tID*10;

    for ( i = 0; i < 10; ++i ) {
        head = pp_head[offset + i];
        // printf("i : %ld, tID : %ld, head : %ld \n", i, tID, head);

        while ( head < pp_tail[offset + i] ) {
            vi_idx = head;
            v = points[indices[vi_idx]];
            get_nth_digit(v, level, k);
            // printf("v : %ld, level : %ld, k : %ld ", v, level, k);

            while ( (k != i) && (pp_head[offset + k] < pp_tail[offset + k]) ) {
                temp = indices[vi_idx];
                indices[vi_idx] = indices[pp_head[offset + k]];
                indices[pp_head[offset + k]] = temp;
                // vi_idx = pp_head[offset + k];
                pp_head[offset + k] += 1;

                v = points[indices[vi_idx]];
                get_nth_digit(v, level, k);
            }

            if ( k == i ) {
                temp = indices[vi_idx]; // = indices[head]
                indices[head] = indices[pp_head[offset + i]];
                head += 1;
                indices[pp_head[offset + i]] = temp;
                pp_head[offset + i] += 1;
            } else {
                indices[head] = indices[vi_idx];
                head += 1;
            }
        }
    }
}


void PARADIS_repair(
        INT64 *points, INT64 *indices, INT64 *pp_head, INT64 *pp_tail,
        INT64 *g_head, INT64 *g_tail, INT64 bucket_i, INT64 level,
        INT64 NUM_THREADS) {

    INT64 tail, tID, offset, head, v, vi_idx, k1, k2, w, wi_idx, temp1, temp2;

    tail = g_tail[bucket_i];

    for ( tID = 0; tID < NUM_THREADS; ++tID) {
        offset = tID * 10;
        head = pp_head[offset + bucket_i];

        while ( (head < pp_tail[offset + bucket_i]) && (head < tail) ) {
            vi_idx = head;
            v = points[indices[vi_idx]];
            get_nth_digit(v, level, k1);
            head += 1;

            if ( k1 != bucket_i ) {
                while ( head < tail ) {
                    tail -= 1;
                    wi_idx = tail;
                    w = points[indices[wi_idx]];
                    get_nth_digit(w, level, k2);
                    if ( k2 == bucket_i ) {
                        temp1 = indices[wi_idx];
                        temp2 = indices[vi_idx];
                        // indices[head - 1] = indices[wi_idx];
                        // indices[tail] = indices[vi_idx];
                        indices[head - 1] = temp1;
                        indices[tail] = temp2;
                    }
                }
            }
        }
    }

    g_head[bucket_i] = tail;

}




void PARADIS(
        INT64 *points, INT64 *indices, INT64 array_size, INT64 highest_pos) {

    INT64 gi, NUM_THREADS, g_count;
    INT64 *g_hist, *g_head, *g_tail, *pp_head, *pp_tail, *local_hist;

    g_hist = (INT64 *) aligned_malloc(10 * sizeof(INT64));
    for ( gi = 0; gi < 10; ++gi ) {
        g_hist[gi] = 0;
    }
    g_head = (INT64 *) aligned_malloc(10 * sizeof(INT64));
    g_tail = (INT64 *) aligned_malloc(10 * sizeof(INT64));


    #pragma omp parallel
    {
        #pragma omp single
        {
            NUM_THREADS = omp_get_num_threads();
            local_hist = (INT64 *) aligned_malloc(10 * NUM_THREADS * sizeof(INT64));
            pp_head = (INT64 *) aligned_malloc(10 * NUM_THREADS * sizeof(INT64));
            pp_tail = (INT64 *) aligned_malloc(10 * NUM_THREADS * sizeof(INT64));
        }

        INT64 tID, low, high, t_iter, digit, elem, i, num_elem, level;

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
                elem = points[t_iter];
                get_nth_digit(elem, level, digit);
                // printf("num : %ld, level : %ld, digit : %ld \n", elem, level, digit);
                local_hist[10*tID + digit] += 1;
            }
            // print_arr(&local_hist[tID*10], 10);
            // printf("local_hist built for tID : %ld \n", tID);
            #pragma omp barrier
            /* local histogram built */


            /* building global histogram */
            low = (tID * 10) / NUM_THREADS;
            high = ( (tID + 1) * 10 ) / NUM_THREADS;
            if ( low != high ) {
                for ( t_iter = low; t_iter < high; ++t_iter ) {
                    // printf("tID : %ld -- 1\n", tID);
                    g_hist[t_iter] = 0;
                    // printf("tID : %ld -- 2\n", tID);
                    for ( i = 0; i < NUM_THREADS; ++i ) {
                        g_hist[t_iter] += local_hist[10*i + t_iter];
                    }
                }
            }
            // printf("g_hist built for tID : %ld \n", tID);
            #pragma omp barrier
            #pragma omp single
            {
                print_arr(g_hist, 10);
            }

            if ( low != high ) {
                for ( t_iter = low; t_iter < high; ++t_iter ) {
                    g_head[t_iter] = 0;
                    for ( i = 0; i < t_iter; ++i ) {
                        g_head[t_iter] += g_hist[i];
                    }
                    g_tail[t_iter] = g_head[t_iter] + g_hist[t_iter];
                }
            }
            // printf("g_head and g_tail built for tID : %ld \n", tID);
            #pragma omp barrier
            #pragma omp single
            {
                print_arr(g_head, 10);
                print_arr(g_tail, 10);
            }
            /* global histogram built */


            /* computing PartitionsForPermutation and g_count */
            if ( tID == 0 ) {
                g_count = 0;
            }
            for ( t_iter = 0; t_iter < 10; ++t_iter ) {
                num_elem = g_tail[t_iter] - g_head[t_iter];
                if ( tID == 0 ) {
                    g_count += num_elem;
                }
                pp_head[10*tID + t_iter] = g_head[t_iter] + ( tID * num_elem) / NUM_THREADS;
                pp_tail[10*tID + t_iter] = g_head[t_iter] + ( (tID + 1) * num_elem) / NUM_THREADS;
            }
            #pragma omp critical
            {
                printf("PartitionsForPermutation done for tID : %ld \n", tID);
                printf("tID : %ld\n", tID);
                print_arr(&pp_head[10*tID], 10);
                print_arr(&pp_tail[10*tID], 10);
                printf("\n");
            }
            
            #pragma omp barrier
            /* PartitionsForPermutation and g_count computed */


            while ( g_count > 0 ) {
                // printf("g_count : %ld : ", g_count);

                PARADIS_permute(
                    points, indices, pp_head, pp_tail, tID, level, NUM_THREADS);
                #pragma omp barrier

                if ( low != high ) {
                    for ( t_iter = low; t_iter < high; ++t_iter) {
                        PARADIS_repair(
                            points, indices, pp_head, pp_tail, g_head, g_tail,
                            t_iter, level, NUM_THREADS);
                    }
                }
                printf("PARADIS_permute and PARADIS_repair done for tID : %ld \n", tID);
                #pragma omp barrier

                #pragma omp single
                {
                    print_arr(indices, array_size);
                    printf("\n[");
                    for ( INT64 i = 0; i < array_size - 1; ++i ) {
                        printf("%ld, ", points[indices[i]]);
                    }
                    printf("%ld]\n \n", points[indices[array_size - 1]]);
                }

                /* computing PartitionsForPermutation and g_count */
                if ( tID == 0 ) {
                    g_count = 0;
                }
                for ( t_iter = 0; t_iter < 10; ++t_iter ) {
                    num_elem = g_tail[t_iter] - g_head[t_iter];
                    if ( tID == 0 ) {
                        g_count += num_elem;
                    }
                    pp_head[10*tID + t_iter] = g_head[t_iter] + ( tID * num_elem ) / NUM_THREADS;
                    pp_tail[10*tID + t_iter] = g_head[t_iter] + ( (tID + 1) * num_elem ) / NUM_THREADS;
                }
                #pragma omp barrier
                /* PartitionsForPermutation and g_count computed */
            }

            #pragma omp single
            {
                print_arr(indices, array_size);
                printf("\n[");
                for ( INT64 i = 0; i < array_size - 1; ++i ) {
                    printf("%ld, ", points[indices[i]]);
                }
                printf("%ld]\n", points[indices[array_size - 1]]);
            }
            #pragma omp barrier
        }
    }
}


int main(int argc, char const *argv[])
{

    INT64 i, num_points = 24;
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