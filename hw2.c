/*
 * Mat Mult
 * CS698 GPU cluster programming - MPI + CUDA 
 * Fall 2024
 * template for HW1 - 3
 * HW1 - point to point communication
 * HW2 - collective communication
 * HW3 - one-sided communication
 * Andrew Sohn
 */


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXDIM 1<<12    /* 4096 */
#define ROOT 0

int mat_mult(double *A, double *B, double *C, int n, int n_local);
void init_data();
int check_result();

int main(int argc, char *argv[]) {
    int i, n = 64,n_sq, flag, my_work;
    int my_rank, num_procs = 1;
    double *A, *B, *C, *D, *local_A, *local_C;
    int elms_to_comm;
    double start_time, end_time, elapsed;

    MPI_Comm world = MPI_COMM_WORLD;
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n > MAXDIM) n = MAXDIM;
    }
    n_sq = n * n;
    
    my_work = n / num_procs; // number of rows te each process
    elms_to_comm = my_work * n; // total number of elements a process deals with since n is number of 
    // elements in a column

    // Allocate memory
    if (my_rank == ROOT) {
        A = (double *)malloc(sizeof(double) * n_sq);
        C = (double *)malloc(sizeof(double) * n_sq);
    }
    B = (double *)malloc(sizeof(double) * n_sq);
    D = (double *)malloc(sizeof(double) * n_sq);

    // Local arrays for each process
    local_A = (double *)malloc(sizeof(double) * elms_to_comm);  /* Local rows of A */
    local_C = (double *)malloc(sizeof(double) * elms_to_comm);

   
    if (my_rank == ROOT) {
        printf("pid=%d: num_procs=%d n=%d my_work=%d\n", my_rank, num_procs, n, my_work);
        init_data(A, n_sq);  
        init_data(B, n_sq);  

        // printf("The root matrix\n");
        // for (i = 0 ; i < n_sq ; i ++) {
        //     printf("%f..", A[i]);
        // }
    }

    start_time = MPI_Wtime();

    // divide A into number of processes and send them using scatter not bcast
    MPI_Scatter(A, elms_to_comm, MPI_DOUBLE, local_A, elms_to_comm, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    // broadcast same B matrix to all processes
    MPI_Bcast(B, n_sq, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    // peform the local matrix multiplication
    mat_mult(local_A, B, local_C, n, my_work);

    // put a barrier for all processes to complete the multiplication
    MPI_Barrier(world);

    // Gather the results from all processes into matrix C on the root process
    MPI_Gather(local_C, elms_to_comm, MPI_DOUBLE, C, elms_to_comm, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);


    // if (my_rank != 0) {
    //     // process i received
    //     printf("process %d received..\n", my_rank);
    //     for (i = 0; i < elms_to_comm; i++) printf("%f..", local_A[i]);
    //     printf("\n");
    // }

    if (my_rank == ROOT) {
        end_time = MPI_Wtime();
        elapsed = end_time - start_time;

        /* Local computation for comparison: results in D */
        mat_mult(A, B, D, n, n);

        flag = check_result(C, D, n);
        if (flag) printf("Test: FAILED\n");
        else {
            printf("Test: PASSED\n");
            printf("Total time %d: %f seconds.\n", my_rank, elapsed);
        }

    }

    MPI_Finalize();
    return 0;
}

int mat_mult(double *a, double *b, double *c, int n, int my_work) {
    int i, j, k, sum = 0;
    for (i = 0; i < my_work; i++) {
        for (j = 0; j < n; j++) {
            sum = 0;
            for (k = 0; k < n; k++)
                sum = sum + a[i * n + k] * b[k * n + j];
            c[i * n + j] = sum;
        }
    }
    return 0;
}

/* Initialize an array with random data */
void init_data(double *data, int data_size) {
    for (int i = 0; i < data_size; i++)
        data[i] = rand() & 0xf;
}

/* Compare two matrices C and D */
int check_result(double *C, double *D, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (C[i * n + j] != D[i * n + j]) {
                printf("ERROR: C[%d][%d]=%f != D[%d][%d]=%f\n", i, j, C[i * n + j], i, j, D[i * n + j]);
                return 1;
            }
        }
    }
    return 0;
}

/*
  End of file: template.c
 */