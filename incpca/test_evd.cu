#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "magma.h"
// includes, cuda 
// matrix indexing convention
#define id(n, m, ld) (((n) * (ld) + (m)))
#define checkmem(x) do { \
        if (x == 0) {  \
            fprintf(stderr, "! host memory allocation error: T\n"); \
            return EXIT_FAILURE; \
        } \
    } while (0);
       

// declarations
int gs_pca_cublas(int, int, int, double *, double *, double *);

int print_results(int, int, int, double *, double *, double *, double *);
// main
int main(int argc, char** argv) {
    cudaError_t cudaStat;    
    cublasStatus_t stat;
    cublasHandle_t handle;

    magma_init();

    // initiallize some random test data X 
    int N = 3, M = 3;
    double *X = 0;
    X = (double*)malloc(N * M * sizeof(double));
    if(X == 0) {
        fprintf (stderr, "! host memory allocation error: X\n"); return EXIT_FAILURE;
    }
    for(int n = 0; n < N; n++) {
        for(int m = 0; m < M; m++) {
            X[id(n,m,M)] = (double) (n * M + m); //rand() / (double)RAND_MAX;
            /*printf("(%d, %d) ", n*M+m, id(n,m,M));*/
        } 
    }

    X[0] = 3;
    X[1] = 3;
    X[2] = 5;
    X[3] = 6;
    X[4] = 3;
    X[5] = 6;
    X[6] = 7;
    X[7] = 7;
    X[8] = 5;



    double *dX = 0;
    cudaMalloc((void**)&dX, N * M * sizeof(double));
    /*cudaMemcpy(dX, X, N * M * sizeof(double), cudaMemcpyHostToDevice);*/
    cublasSetMatrix (N, M, sizeof(double), X, N, dX, N);

    
    
    double *wr = 0;// (double*) malloc(N * sizeof(double));
    double *wi = 0;//(double*) malloc(N * sizeof(double));
    double *V = 0;//(double*) malloc(N * N * sizeof(double));
    int nb = magma_get_dgehrd_nb(N);
    printf("nb: %d\n", nb);
    int lwork = N*(2 + 2*nb);
    double *work = 0;//(double*) malloc(lwork * sizeof(double));
    magma_int_t info;
    magma_malloc_cpu( (void**) &wr, (N)*sizeof(double) );
    magma_malloc_cpu( (void**) &wi, (N)*sizeof(double) );
    magma_malloc_cpu( (void**) &V, (N * N)*sizeof(double) );
    magma_malloc_cpu( (void**) &work, (lwork)*sizeof(double) );
    
    


    magma_dgeev_m(MagmaNoVec, MagmaVec, N,
            X, N, wr, wi,
            NULL, N, V, N,
            work, lwork, &info);
    
    if (info != 0) {
        printf("info != 0\n");
    }
    double *evs = (double*) malloc(N * sizeof(double));
    cudaMemcpy(evs, wr, N * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf("%lf\n", wr[i]);
    }    

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%lf ", V[j*N + i]);
        }
        puts("");
    }



    /*for (int i = 0; i < M; i++) {*/
        /*printf("%lf ", X[id(0, i, M)]);*/
    /*}*/
    /*puts("aa");*/
    
    /*double *dY = 0;*/
    /*double *Y = 0;*/
    /*[>Y = (double*)malloc(M * sizeof(double));<]*/
    /*cudaStat = cudaMalloc((void**)&dY, M * sizeof(double));*/
    /*puts("sadfddf");*/
    /*if (cudaStat != cudaSuccess) {*/
        /*printf ("device memory allocation failed");*/
        /*return EXIT_FAILURE;*/
    /*}*/
    /*puts("gg");*/
    /*stat = cublasCreate(&handle);*/
    /*if (stat != CUBLAS_STATUS_SUCCESS) { */
        /*printf ("CUBLAS initialization failed\n");*/
        /*return EXIT_FAILURE;*/
    /*}*/

    
    /*[>double *dY2 = 0<]*/
    /*[>cudaMalloc((void**)&dY2, (N-1) * (M-1) * sizeof(double));<]*/
    /*[>cudaDcopy(handle, M)<]*/




    /*fprintf(stderr, "ggg\n");*/

    /*Y = (double*) malloc(M * sizeof(double));*/
    
    /*[>cublasDcopy(M, &dR[0], 1, dU, 1);<]*/
    /*printf("before cublasDcopy\n");*/
    /*stat = cublasDcopy(handle, M, dX, 1, dY, 1);*/
    /*printf("done cublasDcopy \n");*/
    /*[>cudaMemcpy(&Y[0], &dY[0], M * sizeof(double), cudaMemcpyDeviceToHost);<]*/
    /*cublasGetMatrix (1, M, sizeof(double), dY, 1, Y, 1);*/
    /*if(stat != CUBLAS_STATUS_SUCCESS) {*/
        /*printf("cublas fail 33\n");*/
    /*}*/
    /*for (int i = 0; i < M; i++) {*/
        /*printf("%lf ",Y[i]);*/
    /*}*/
    /*puts("");*/
    

    /*[>free(Y);<]*/
    /*[>free(X);<]*/
    
}

