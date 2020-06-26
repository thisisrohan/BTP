#include "tetgen.h"
// #include <stdio.h>
// #include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <omp.h>



double uniformdoublerand()
{
  double result;
  long a, b;

  a = random();
  b = random();
  result = (double) (a - 1073741824) * 8388608.0 + (double) (b >> 8);
  return result;
}


int main(int argc, char const *argv[])
{
    
    int num_points = 100, i;

    tetgenio in, out;
    in.firstnumber = 1;

    in.numberofpoints = num_points;
    in.pointlist = new double[in.numberofpoints * 3];

    for ( i = 0; i < 3 * num_points; ++i ) {
        in.pointlist[i] = uniformdoublerand();
    }

    in.save_nodes("barin");

    double start, end;
    double cpu_time;

    start = omp_get_wtime();
    tetrahedralize("", &in, &out);
    end = omp_get_wtime();

    cpu_time = (double) (end - start);
    printf("time taken : %f\n", cpu_time);

    return 0;
}