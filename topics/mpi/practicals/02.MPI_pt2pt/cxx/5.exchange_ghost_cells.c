/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS Summer School.        *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS take no responsibility for the use of the enclosed      *
 * teaching material.                                           *
 *                                                              *
 * Purpose: Exchange ghost cell in 2 directions                 *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

/* Use only 16 processes for this exercise
 * Send the ghost cell in two directions: top to bottom and bottom to top
 * consider cyclic boundaries, for instance, rank 0 and rank 12 are connected together
 *
 * process decomposition on 4*4 grid
 *
 *  |-----------|
 *  | 0| 1| 2| 3|
 *  |-----------|
 *  | 4| 5| 6| 7|
 *  |-----------|
 *  | 8| 9|10|11|
 *  |-----------|
 *  |12|13|14|15|
 *  |-----------|
 *
 * Each process works on a 10*10 (SUBDOMAIN) block of data
 * the D corresponds to data, g corresponds to "ghost cells"
 * xggggggggggx
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * xggggggggggx
 */

/* Tasks:
 * A. each rank has to find its top and bottom neighbor
 * B. send them the data they need
 *    - top array goes to top neighbor
 *    - bottom array goes to bottom neighbor
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SUBDOMAIN 10
#define DOMAINSIZE (SUBDOMAIN+2)

#define MPI_B

int main(int argc, char *argv[])
{
    int rank, size, i, j, rank_bottom, rank_top;
    double data[DOMAINSIZE*DOMAINSIZE];
    MPI_Request request;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double sndbuf[DOMAINSIZE];
    double rcvbuf[DOMAINSIZE];

    if (size!=16) {
        printf("please run this with 16 processors\n");
        MPI_Finalize();
        exit(1);
    }

    for (i=0; i<DOMAINSIZE*DOMAINSIZE; i++) {
        data[i]=rank;
    }

    rank_bottom = (rank + 4) % 16;/* find the rank of the top neighbor */
    rank_top = (rank - 4 + size) % 16;/* find the rank of the bottom neighbor */


    //  ghost cell exchange with the neighbouring cells (cyclic) to the bottom and to the top using:
    //  a) MPI_Send, MPI_Irecv, MPI_Wait
    //  b) MPI_Isend, MPI_Recv, MPI_Wait
    //  c) MPI_Sendrecv



    // Get data from top into sndbuf
    for(i = 0; i < SUBDOMAIN; i++)
    	sndbuf[i] = data[i+1];
    //  to the top

    // a)
#ifdef MPI_A
    MPI_Send(&sndbuf, SUBDOMAIN, MPI_DOUBLE, rank_top, 0, MPI_COMM_WORLD);
    MPI_Irecv(&rcvbuf, SUBDOMAIN, MPI_DOUBLE, rank_bottom, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, &status);
#endif
    // b)
#ifdef MPI_B
    MPI_Isend(&sndbuf, SUBDOMAIN, MPI_DOUBLE, rank_top, 0, MPI_COMM_WORLD, &request);
    MPI_Recv(&rcvbuf, SUBDOMAIN, MPI_DOUBLE, rank_bottom, 0, MPI_COMM_WORLD, &status);
    MPI_Wait(&request, &status);
#endif
    // c)
#ifdef MPI_C
    MPI_Sendrecv(
    		&sndbuf, SUBDOMAIN, MPI_DOUBLE, rank_top, 0,
			&rcvbuf, SUBDOMAIN, MPI_DOUBLE, rank_bottom, 0,
			MPI_COMM_WORLD,  &status
    		);
#endif

    // get data from btm into sndbuf, then replace data to top
    for(i = 0; i < SUBDOMAIN; i++) {
    	sndbuf[i] = data[i+1 + (SUBDOMAIN+2)*(SUBDOMAIN+1)];
    	data[i+1 + (SUBDOMAIN+2)*(SUBDOMAIN+1)] = rcvbuf[i];
    }

#ifdef MPI_A
    MPI_Send(&sndbuf, SUBDOMAIN, MPI_DOUBLE, rank_bottom, 0, MPI_COMM_WORLD);
    MPI_Irecv(&rcvbuf, SUBDOMAIN, MPI_DOUBLE, rank_top, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, &status);
#endif
    // b)
#ifdef MPI_B
    MPI_Isend(&sndbuf, SUBDOMAIN, MPI_DOUBLE, rank_bottom, 0, MPI_COMM_WORLD, &request);
    MPI_Recv(&rcvbuf, SUBDOMAIN, MPI_DOUBLE, rank_top, 0, MPI_COMM_WORLD, &status);
    MPI_Wait(&request, &status);
#endif
    // c)
#ifdef MPI_C
    MPI_Sendrecv(
    		&sndbuf, SUBDOMAIN, MPI_DOUBLE, rank_bottom, 0,
			&rcvbuf, SUBDOMAIN, MPI_DOUBLE, rank_top, 0,
			MPI_COMM_WORLD,  &status
    		);
#endif

    // get data from btm into sndbuf, then replace data to top
       for(i = 0; i < SUBDOMAIN; i++) {
       	data[i+1] = rcvbuf[i];
       }

    //  to the bottom
    // a)

    // b)

    // c)

    if (rank==2) {
        printf("data of rank 2 after communication\n");
        for (j=0; j<DOMAINSIZE; j++) {
            for (i=0; i<DOMAINSIZE; i++) {
                printf("%.1f ", data[i+j*DOMAINSIZE]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
