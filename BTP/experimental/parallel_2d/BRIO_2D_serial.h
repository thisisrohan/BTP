#ifndef BRIO_2D_SERIAL_H
#define BRIO_2D_SERIAL_H

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include "helpful_tools.h"

#ifndef INT64
#define INT64 int64_t
#endif
#ifndef UINT64
#define UINT64 uint64_t
#endif

#define _round_serial(x) \
    temp = (INT64) (x + 0.5); \
    x = (double) temp

INT64 randomseed_serial = 1;
INT64 prng_a_serial = 13663;
INT64 prng_c_serial = 1508891;
INT64 prng_m_serial = 7140257;

INT64 randomnation_serial(INT64 low, INT64 high) {
    // the linear congruential prng from Shewchuk's triangle, modified slightly
    randomseed_serial = (randomseed_serial * prng_a_serial + prng_c_serial) % prng_m_serial;
    return randomseed_serial / (prng_m_serial / (high - low) + 1) - low;
}


void _qargsort_serial(INT64* points, INT64* indices, INT64 array_size) {

    INT64 left, right, pivot_idx, pivot_point, temp;

    if ( array_size == 2 ) {
        /* Recursive base case */
        if ( points[indices[0]] > points[indices[1]] ) {
            temp = indices[0];
            indices[0] = indices[1];
            indices[1] = temp;
        }
        return;
    }

    /* Choose a random pivot to split the array. */
    pivot_idx = array_size / 2;
    pivot_point = points[indices[pivot_idx]];
    
    /* Split the array. */
    left = -1;
    right = array_size;
    while ( left < right ) {
        /* Search for a point which is too large for the left indices. */
        do {
            left++;
        } while ( (left <= right) && (points[indices[left]] < pivot_point) );
        /* Search for a point which is too small for the right indices. */
        do {
            right--;
        } while ( (left <= right) && (points[indices[right]] > pivot_point) );
        if ( left < right ) {
            /* Swap the left and right indices. */
            temp = indices[left];
            indices[left] = indices[right];
            indices[right] = temp;
        }
    }

    if ( left > 1 ) {
        /* Recursively sort the left subset. */
        _qargsort_serial(points, indices, left);
    }
    if ( right < array_size - 2 ) {
        /* Recursively sort the right subset. */
        _qargsort_serial(points, &indices[right + 1], array_size - right - 1);
    }
}


void _make_rounds_serial(
        INT64* rounds, INT64* boundary_indices, INT64 num_points, INT64 num_rounds,
        INT64* indices) {

    INT64 bi_insertion_idx, rounds_insertion_idx, points_left, last_idx;
    INT64 round_idx, points_added, i, j, temp, flip;

    bi_insertion_idx = num_rounds;
    rounds_insertion_idx = num_points - 1;

    boundary_indices[bi_insertion_idx] = rounds_insertion_idx + 1;
    bi_insertion_idx -= 1;

    points_left = num_points;
    last_idx = num_points - 1;
    for ( round_idx = 1; round_idx < num_rounds; round_idx++ ) {
        points_added = 0;
        for ( i = 0; i < points_left; i++ ) {
            j = randomnation_serial(0l, last_idx + 1);
            flip = randomnation_serial(0l, 2l);
            if ( flip == 1 ) {
                // update rounds
                rounds[rounds_insertion_idx] = indices[j];

                // swap j and last_idx in indices
                temp = indices[j];
                indices[j] = indices[last_idx];
                indices[last_idx] = temp;

                // update points_added, last_idx and rounds_insertion_idx
                points_added += 1;
                last_idx -= 1;
                rounds_insertion_idx -= 1;
            }
        }

        points_left -= points_added;

        // update boundary_indices
        boundary_indices[bi_insertion_idx] = rounds_insertion_idx + 1;
        bi_insertion_idx -= 1;
    }

    for ( i = 0; i < last_idx + 1; i++ ) {
        rounds[rounds_insertion_idx] = indices[i];
        rounds_insertion_idx -= 1;
    }

    boundary_indices[bi_insertion_idx] = rounds_insertion_idx + 1;
    bi_insertion_idx -= 1;

    return;
}


#define hindex2xy_serial(hindex, p) \
    place = hindex & 3; \
    if ( place == 0 ) { \
        x = 0; \
        y = 0; \
    } else if ( place == 1 ) { \
        x = 0; \
        y = 1; \
    } else if ( place == 2 ) { \
        x = 1; \
        y = 1; \
    } else { \
        x = 1; \
        y = 0; \
    } \
    hindex = hindex >> 2; \
    for ( i = 1; i < p; i++ ) { \
        n = 1l << i; \
        place = hindex & 3; \
        if ( place == 0 ) { \
            temp = x; \
            x = y; \
            y = temp; \
        } else if ( place == 1 ) { \
            y += n; \
        } else if ( place == 2 ) { \
            x += n; \
            y += n; \
        } else { \
            temp = x; \
            x = 2*n - 1 - y; \
            y = n - 1 - temp; \
        } \
        hindex = hindex >> 2; \
    }


void _make_hilbert_curve_serial(INT64* h_arr, uint8_t p) {

    INT64 a, x, y, hindex, place, i, j, n, temp;
    a = 1l << p; // faster way to do a = pow(2, p)
    for ( j = 0; j < a*a; j++ ) {
        hindex = j;
        hindex2xy_serial(hindex, p);
        h_arr[x + a*y] = j;
        // printf("hindex : ");
        // printf("%ld\n", j);
    }
    return;
}


void _sort_along_hilbert_curve_serial(
    double* org_points, INT64* h_arr, INT64* h_indices, INT64* new_indices,
    INT64 op_end, uint8_t p) {

    double min_x, max_x, min_y, max_y, max_xy, x, y;
    INT64 i, a, xi, yi, temp;

    min_x = org_points[0];
    max_x = org_points[0];
    min_y = org_points[1];
    max_y = org_points[1];

    for ( i = 0; i < op_end; i++ ) {
        x = org_points[2*i];
        y = org_points[2*i + 1];
        if ( x < min_x ) min_x = x;
        else if ( x > max_x ) max_x = x;
        if ( y < min_y ) min_y = y;
        else if ( y > max_y ) max_y = y;
    }

    max_x -= min_x;
    max_y -= min_y;
    if ( max_y > max_x ) max_xy = max_y;
    else max_xy = max_x;

    if ( max_xy == 0 ) max_xy = 1;

    a = 1l << p;

    for ( i = 0; i < op_end; i++ ) {
        x = org_points[2*i];
        y = org_points[2*i + 1];
        x = ( (x - min_x)/max_xy )*(a - 1);
        y = ( (y - min_y)/max_xy )*(a - 1);
        _round_serial(x);
        _round_serial(y);
        xi = (INT64) x;
        yi = (INT64) y;
        h_indices[i] = h_arr[xi + a*yi];
        new_indices[i] = i;
    }

    _qargsort_serial(h_indices, new_indices, op_end);

    return;
}


void make_BRIO_serial(
        double* points, double* new_points, INT64* insertion_seq,
        INT64 num_points) {

    if ( num_points > prng_m_serial ) prng_m_serial = num_points + 13l;

    INT64 num_rounds, i, max_pts_in_a_round, j, rho, op_end;
    uint8_t p;
    num_rounds = log2_64(num_points);

    INT64* rounds;
    rounds = (INT64 *) aligned_malloc(num_points * sizeof(INT64));
    // printf("yoy 1 of 3\n");

    INT64* points_left_old;
    points_left_old = (INT64 *) aligned_malloc(num_points * sizeof(INT64));
    // printf("yoy 2 of 3\n");

    INT64* boundary_indices;
    boundary_indices = (INT64 *) aligned_malloc((num_rounds + 1) * sizeof(INT64));
    // printf("yoy 3 of 3\n");

    for ( i = 0; i < num_points; i++ ) {
        points_left_old[i] = i;
    }

    _make_rounds_serial(
        rounds, boundary_indices, num_points, num_rounds, points_left_old);
    // printf("rounds made\n");

    max_pts_in_a_round = boundary_indices[1] - boundary_indices[0];
    for ( i = 1; i < num_rounds; i++ ) {
        j = boundary_indices[i + 1] - boundary_indices[i];
        if ( j > max_pts_in_a_round ) max_pts_in_a_round = j;
    }
    // printf("max_pts_in_a_round : %ld \n", max_pts_in_a_round);

    rho = 5; // This is the number of points per grid cell that we are comfortable with
    if ( num_points <= rho*(1l << (2*4)) ) p = 4;
    else {
        j = log2_64(num_points / rho);
        i = j / 2;
        if ( 2*i > j ) i += 1;
        p = (uint8_t) i;
    }
    // printf("p : "); printf("%d\n", p);

    i = 1l << p;
    // printf("i : %ld\n", i);

    INT64* h_arr;
    h_arr = (INT64 *) aligned_malloc(i * i * sizeof(INT64));
    // printf("yo 1 of 4\n");

    double* org_points;
    org_points = (double *) aligned_malloc(2 * max_pts_in_a_round * sizeof(double));
    // printf("yo 2 of 4\n");

    INT64* h_indices;
    h_indices = (INT64 *) aligned_malloc(max_pts_in_a_round * sizeof(INT64));
    // printf("yo 3 of 4\n");

    INT64* new_indices;
    new_indices = (INT64 *) aligned_malloc(max_pts_in_a_round * sizeof(INT64));
    // printf("yo 4 of 4\n");

    _make_hilbert_curve_serial(h_arr, p);
    // printf("hilbert curve made\n");

    INT64 b1, b2;
    for ( i = 0; i < num_rounds; i++ ) {
        op_end = 0;
        b1 = boundary_indices[i];
        b2 = boundary_indices[i + 1];
        for ( j = b1; j < b2; j++ ) {
            org_points[2*op_end] = points[2*rounds[j]];
            org_points[2*op_end + 1] = points[2*rounds[j] + 1];
            op_end += 1;
        }

        _sort_along_hilbert_curve_serial(
            org_points, h_arr, h_indices, new_indices, op_end, p);

        for ( j = b2 - 1; j >= b1; j--) {
            insertion_seq[j] = rounds[b1 + new_indices[op_end - 1]];
            op_end -= 1;
        }
    }

    for ( i = 0; i < num_points; i++ ) {
        new_points[2*i] = points[2*insertion_seq[i]];
        new_points[2*i + 1] = points[2*insertion_seq[i] + 1];
    }

    aligned_free(rounds);
    aligned_free(points_left_old);
    aligned_free(boundary_indices);
    aligned_free(h_arr);
    aligned_free(org_points);
    aligned_free(h_indices);
    aligned_free(new_indices);

    return;
}


// int main(int argc, char const *argv[]) {
//     /* code */
//     return 0;
// }

#endif /* BRIO_2D_SERIAL_H */