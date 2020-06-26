#ifndef QUEUE_H
#define QUEUE_H

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "helpful_tools.h"

#ifndef INT64
#define INT64 int64_t
#endif
#ifndef UINT64
#define UINT64 uint64_t
#endif


void enqueue(INT64** arr_pointers, INT64* q_params, INT64 elem) {

    INT64* queue = arr_pointers[3];
    INT64 q_head, q_tail, q_cap, q_num_items;
    q_head = q_params[0];
    q_tail = q_params[1];
    q_cap = q_params[2];
    q_num_items = q_params[3];

    if ( q_num_items == q_cap) {
        #pragma omp critical (MEM_ALLOC)
        {
            queue = (INT64 *) aligned_realloc(queue, 2 * q_cap * sizeof(INT64));
        }
        arr_pointers[3] = queue;
        q_cap *= 2;
        q_params[2] = q_cap;
    }

    queue[q_tail] = elem;

    q_tail += 1;
    if ( q_tail >= q_cap ) q_tail -= q_cap;

    if ( q_head == q_tail ) q_num_items = q_cap;
    else if ( q_tail > q_head ) q_num_items = q_tail - q_head;
    else q_num_items = (q_cap - q_head) + q_tail;

    q_params[1] = q_tail;
    q_params[3] = q_num_items;

    return;
}


INT64 dequeue(INT64** arr_pointers, INT64* q_params) {

    INT64* queue = arr_pointers[3];
    INT64 q_head, q_tail, q_cap, q_num_items, elem;
    q_head = q_params[0];
    q_tail = q_params[1];
    q_cap = q_params[2];

    elem = queue[q_head];
    q_head += 1;
    if ( q_head >= q_cap ) q_head -= q_cap;

    if ( q_head == q_tail ) q_num_items = 0;
    else if ( q_tail > q_head ) q_num_items = q_tail - q_head;
    else q_num_items = (q_cap - q_head) + q_tail;

    q_params[0] = q_head;
    q_params[3] = q_num_items;

    return elem;
}



#endif /* QUEUE_H */