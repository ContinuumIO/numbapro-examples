/*
Numba requires return value to be passed as
a pointer in the first argument.


To compile:

	nvcc -arch=sm_20 -dc jitlink.cu -o jitlink.o

*/
#include <cstdio>

extern "C" {

__device__
int bar(int* retval, int a, int b){
	/* Fill this function with anything */
	
	/* Return 0 to indicate success */
	return 0;
}

}
