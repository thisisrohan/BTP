#include <stdlib.h>
#include <time.h>


// void srand((unsigned) time(NULL));
void qargsort (int* points, int* indices, int array_size) {

    int left, right, pivot_idx, pivot_point, temp;

    if (array_size == 2) {
        /* Recursive base case */
        if (points[indices[0]] > points[indices[1]]) {
            temp = indices[0];
            indices[0] = indices[1];
            indices[1] = temp;
        }
        return;
    }

    /* Choose a random pivot to split the array. */
    pivot_idx = rand() % array_size;
    pivot_point = points[indices[pivot_idx]];
    /* Split the array. */
    left = -1;
    right = array_size;
    while (left < right) {
        /* Search for a point which is too large for the left indices. */
        do {
            left++;
        } while ((left <= right) && (points[indices[left]] < pivot_point));
        /* Search for a point which is too small for the right indices. */
        do {
            right--;
        } while ((left <= right) && (points[indices[right]] > pivot_point));
        if (left < right) {
            /* Swap the left and right indices. */
            temp = indices[left];
            indices[left] = indices[right];
            indices[right] = temp;
        }
    }
    if (left > 1) {
        /* Recursively sort the left subset. */
        qargsort(points, indices, left);
    }
    if (right < array_size - 2) {
        /* Recursively sort the right subset. */
        qargsort(points, &indices[right + 1], array_size - right - 1);
    }
}