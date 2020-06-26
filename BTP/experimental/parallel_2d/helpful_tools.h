#ifndef HELPFUL_TOOLS_H
#define HELPFUL_TOOLS_H

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#ifndef INT64
#define INT64 int64_t
#endif
#ifndef UINT64
#define UINT64 uint64_t
#endif


/* A fast way to exponentiate integers, courtesy :                           */
/* https://gist.github.com/orlp/3551590                                      */

static const uint8_t highest_bit_set[] = {
        0, 1, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 255, // anything past 63 is a guaranteed overflow with base > 1
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
};

INT64 ipow(INT64 base, uint8_t exp) {

    INT64 result = 1;

    switch (highest_bit_set[exp]) {
    case 255: // we use 255 as an overflow marker and return 0 on overflow/underflow
        if (base == 1) {
            return 1;
        }
        
        if (base == -1) {
            return 1 - 2 * (exp & 1);
        }
        
        return 0;
    case 6:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 5:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 4:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 3:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 2:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 1:
        if (exp & 1) result *= base;
    default:
        return result;
    }
}


/* the log2 functions below are taken from :                                 */
/* https://stackoverflow.com/a/11398748                                      */
/* They have been modified so that they instead of rounding down they return */
/* an approximately rounded off value                                        */

static const uint8_t tab64[64] = {
    63,  0, 58,  1, 59, 47, 53,  2,
    60, 39, 48, 27, 54, 33, 42,  3,
    61, 51, 37, 40, 49, 18, 28, 20,
    55, 30, 34, 11, 43, 14, 22,  4,
    62, 57, 46, 52, 38, 26, 32, 41,
    50, 36, 17, 19, 29, 10, 13, 21,
    56, 45, 25, 31, 35, 16,  9, 12,
    44, 24, 15,  8, 23,  7,  6,  5};

INT64 log2_64 (INT64 value) {
    INT64 res, mid_l;
    double mid_d;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;

    res = (INT64) tab64[((uint64_t)((value - (value >> 1))*0x07EDD5E59A4E28C2)) >> 58];
    mid_d = (double) (1l << res);
    mid_d *= 1.4142135624;
    mid_l = (INT64) mid_d;
    if ( value >= mid_l ) res += 1;
    return res;
}

static const uint8_t tab32[32] = {
     0,  9,  1, 10, 13, 21,  2, 29,
    11, 14, 16, 18, 22, 25,  3, 30,
     8, 12, 20, 28, 15, 17, 24,  7,
    19, 27, 23,  6, 26,  5,  4, 31};

INT64 log2_32 (INT64 value) {
    INT64 res, mid_l;
    double mid_d;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    
    res = (INT64) tab32[(uint32_t)(value*0x07C4ACDD) >> 27];
    mid_d = (double) (1l << res);
    mid_d *= 1.4142135624;
    mid_l = (INT64) mid_d;
    if ( value >= mid_l ) res += 1;
    return res;
}


/* `aligned_malloc` and `aligned_free` routines are courtesy :               */
/* https://gist.github.com/dblalock/255e76195676daa5cbc57b9b36d1c99a         */

// cache line size on most modern processors is 64 bytes
static const size_t ALIGNMENT = 64;

void* aligned_malloc(size_t size) {
    size_t request_size = size + ALIGNMENT;
    char *buf = (char *) malloc(request_size);

    size_t remainder = ((size_t)buf) % ALIGNMENT;
    size_t offset = ALIGNMENT - remainder;
    char *ret = buf + (unsigned char)offset;

    // store how many extra bytes we allocated in the byte just before the
    // pointer we return
    *(unsigned char *)(ret - 1) = offset;

    return (void *) ret;
}

/* Free memory allocated with aligned_alloc */
void aligned_free(void* aligned_ptr) {
    int offset = *(((char *) aligned_ptr) - 1);
    free(((char *) aligned_ptr) - offset);
}


void* aligned_realloc(void* ptr, size_t new_size) {
    size_t offset = *( ( (char *) ptr ) - 1 );
    char *actual_ptr = ( (char *) ptr ) - offset;
    char *new_ptr = (char *) realloc(actual_ptr, new_size + offset);

    if (new_ptr != actual_ptr) {
        printf("woops\n");
        char *temp_ptr = (char *) aligned_malloc(new_size);
        size_t temp_offset = *(temp_ptr - 1);
        new_ptr = new_ptr + offset;
        for (INT64 i = 0; i < new_size; i++ ) {
            temp_ptr[i] = new_ptr[i];
        }
        new_ptr = new_ptr - offset;
        free(new_ptr);
        new_ptr = temp_ptr;
    }

    return (void *) new_ptr;
}


#endif /* HELPFUL_TOOLS_H */