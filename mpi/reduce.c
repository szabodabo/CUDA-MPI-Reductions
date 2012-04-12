#include "externalfunctions.h"
#include "constants.h"
#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <strings.h>

int main(int argc, char **argv) {
	int myRank, commSize;
	unsigned long long start, end;
	double reduce_time;
	int i, x;
	int *random_ints;
	int *reduced_ints;
	double *random_doubles;
	double *reduced_doubles;

	///////// The struct and array thereof makes things easier later ///////
	typedef struct {
		MPI_Op op;
		char *name;
	} OpStruct;

	const OpStruct operations[] = {{MPI_MAX, "MAX"}, 
	                               {MPI_MIN, "MIN"},
	                               {MPI_SUM, "SUM"}};
	///////////////////////////////////////////////////////////////////////
	
	//////////////////////// Init MPI ////////////////////////////////////
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &myRank );
	MPI_Comm_size( MPI_COMM_WORLD, &commSize );
	////////////////////////////////////////////////////////////////////

	//////////////// RANDOM NUMBER GENERATION /////////////////////////
	unsigned long rng_init_seeds[6]={0x0, 0x123, 0x234, 0x345, 0x456, 0x789};
	unsigned long rng_init_length=6;
	rng_init_seeds[0] = myRank;
	init_by_array(rng_init_seeds, rng_init_length);

	random_ints = (int *) malloc( NUM_INTS * sizeof(int) );
	random_doubles = (double *) malloc( NUM_DOUBLES * sizeof(double) );
	reduced_ints = (int *) malloc( NUM_INTS * sizeof(int) );
	reduced_doubles = (double *) malloc( NUM_DOUBLES * sizeof(double) );

	for (i = 0; i < NUM_INTS; i++) {
		random_ints[i] = genrand_int32();
	}

	for (i = 0; i < NUM_DOUBLES; i++) {
		random_doubles[i] = genrand_res53();
	}
	///////////////////////////////////////////////////////////////////
	
	//////////////// Start with a warm-up /////////////////////////////
	for (i = 0; i < 1; i++) {
		MPI_Reduce( random_ints, reduced_ints, NUM_INTS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD );
		MPI_Reduce( random_doubles, reduced_doubles, NUM_DOUBLES, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
	}
	///////////////////////////////////////////////////////////////////
	
	
	for (x = 0; x < RETRY_COUNT; x++) {
		//////////////////////// Do INT reductions ///////////////////////
		for (i = 0; i < 3; i++) {
			bzero( reduced_ints, NUM_INTS * sizeof(int) );
			start = rdtsc();
			MPI_Reduce( random_ints, reduced_ints, NUM_INTS, MPI_INT, operations[i].op, 0, MPI_COMM_WORLD );
			end = rdtsc();
			reduce_time = (end - start) / CLOCK_RATE;
			if (myRank == 0) {
				printf("Reduce time for Operation %s (INT) was %10.5lf seconds\n", operations[i].name, reduce_time);
			}
		}
		///////////////////////////////////////////////////////////////////
		
		////////////////////// Do DOUBLE reductions //////////////////////
		for (i = 0; i < 3; i++) {
			bzero( reduced_doubles, NUM_DOUBLES * sizeof(double) );
			start = rdtsc();
			MPI_Reduce( random_doubles, reduced_doubles, NUM_DOUBLES, MPI_DOUBLE, operations[i].op, 0, MPI_COMM_WORLD );
			end = rdtsc();
			reduce_time = (end - start) / CLOCK_RATE;
			if (myRank == 0) {
				printf("Reduce time for Operation %s (DOUBLE) was %10.5lf seconds\n", operations[i].name, reduce_time);
			}
		}
		//////////////////////////////////////////////////////////////////
	}

	free( random_ints );
	free( random_doubles );
	free( reduced_ints );
	free( reduced_doubles );

	MPI_Finalize();
	return EXIT_SUCCESS;
}
