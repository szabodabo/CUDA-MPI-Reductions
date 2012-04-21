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
	double reduce_time, bandwidth;
	int i, x;
	int *random_ints;
	int *reduced_ints;
	double *random_doubles;
	double *reduced_doubles;
	int ints_per_node, doubles_per_node;

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

	ints_per_node = NUM_INTS / commSize;
	doubles_per_node = NUM_DOUBLES / commSize;

	random_ints = (int *) malloc( ints_per_node * sizeof(int) );
	random_doubles = (double *) malloc( doubles_per_node * sizeof(double) );
	reduced_ints = (int *) malloc( ints_per_node * sizeof(int) );
	reduced_doubles = (double *) malloc( doubles_per_node * sizeof(double) );

	for (i = 0; i < ints_per_node; i++) {
		random_ints[i] = genrand_int32();
	}

	for (i = 0; i < doubles_per_node; i++) {
		random_doubles[i] = genrand_res53();
	}
	///////////////////////////////////////////////////////////////////
	
	//////////////// Start with a warm-up /////////////////////////////
	for (i = 0; i < 1; i++) {
		MPI_Reduce( random_ints, reduced_ints, ints_per_node, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD );
		MPI_Reduce( random_doubles, reduced_doubles, doubles_per_node, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
	}
	///////////////////////////////////////////////////////////////////
	//
	if (myRank == 0) {	
		printf("# DATATYPE OP NODES GB/sec\n");
	}

	for (x = 0; x < RETRY_COUNT; x++) {
		//////////////////////// Do INT reductions ///////////////////////
		for (i = 0; i < 3; i++) {
			bzero( reduced_ints, ints_per_node * sizeof(int) );
			start = rdtsc();
			MPI_Reduce( random_ints, reduced_ints, ints_per_node, MPI_INT, operations[i].op, 0, MPI_COMM_WORLD );
			end = rdtsc();
			reduce_time = (end - start) / CLOCK_RATE;
			bandwidth = ((double)(NUM_INTS * sizeof(int)) / (double)reduce_time) / 1073741824; // GB/sec
			if (myRank == 0) {
				printf("INT %s %d %10.3lf\n", operations[i].name, commSize, bandwidth);
			}
		}
		///////////////////////////////////////////////////////////////////
		
		////////////////////// Do DOUBLE reductions //////////////////////
		for (i = 0; i < 3; i++) {
			bzero( reduced_doubles, doubles_per_node * sizeof(double) );
			start = rdtsc();
			MPI_Reduce( random_doubles, reduced_doubles, doubles_per_node, MPI_DOUBLE, operations[i].op, 0, MPI_COMM_WORLD );
			end = rdtsc();
			reduce_time = (end - start) / CLOCK_RATE;
			bandwidth = ((double)(NUM_DOUBLES * sizeof(double)) / (double)reduce_time) / 1073741824; // GB/sec
			if (myRank == 0) {
				printf("DOUBLE %s %d %10.3lf\n", operations[i].name, commSize, bandwidth);
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
