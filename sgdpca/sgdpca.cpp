// C/C++ example for the CUBLAS (NVIDIA)
// implementation of SGD-PCA algorithm //
#include <cstdio>
#include <cstdlib>
#include <string.h>
// includes, blas 
#include "cblas.h"
#include <lapacke.h>
#include <time.h>

#define id(m, n, ld) (((n) * (ld) + (m)))
#define ytran CblasTrans
#define ntran CblasNoTrans
#define order CblasColMajor
static int d = 10000;
static int n = 10000;
static int k = 10;
static int max_iter = 20;
static int lr = 0.01;
static int batch_size = 50;

void print_matrix(float* mat, int d1, int d2)
{

    for(int i=0 ; i<d1 ; i++)
    {
        for(int j=0 ; j<d2 ; j++)
        {
            
            printf("%d, %d, :%lf |", i, j, mat[id(j, i, d2)]);
        }
        printf("\n");
    }
}

void sum_diagnal(float* final_output, int d1, int d2)
{
    float tr = 0.;
    for(int i=0 ; i<d1 ; i++)
    {
        for(int j=0 ; j<d2 ; j++)
        {
            if(i==j)
            {
                tr+=final_output[id(j,i,d2)];
            }
        }
    }
    printf("tr: %lf\n", tr);
}

void Calc_trace(float* U, float* X)
{      
    const float one= 1.0;
    const float zero = 0.0;
    float* output = 0;//Matrix for U^TX
    output = (float*)malloc(k*n*sizeof(float));
    float* final_output = 0;
    final_output = (float*)malloc(sizeof(final_output[0])*k*k);

    cblas_sgemm(order, ytran, ntran, k, n, d, one, U, d, X, d, zero, output, k);
    cblas_sgemm(order, ntran, ytran, k, k, n, one, output, k, output, k, zero, final_output, k);
    sum_diagnal(final_output, k, k);
}
void initZero(float* Mat, int d1, int d2)
{
    for(int i=0 ; i<d1 ; i++)
    {
        for(int j=0 ; j<d2 ; j++)
        {
            Mat[id(j, i, d2)] = 0.0;
        }
    }
}


void initIdentity(float* Mat, int d1, int d2)
{
    for(int i=0 ; i<d1 ; i++)
    {
        for(int j=0 ; j<d2 ; j++)
        {
            if(i==j)
                Mat[id(j, i, d2)] = 1.0;
            else
                Mat[id(j, i, d2)] = 0.0;
        }
    }
}

void mat_add(float* mat1, float* mat2, float* mat3, int d1, int d2, float alpha, float beta)
{
    //mat3 = mat1 + mat2
    for(int i=0 ; i<d1 ; i++)
    {
        for(int j=0 ; j<d2 ; j++)
        {
            mat3[id(j,i,d)] = alpha*mat1[id(j,i,d)] + beta*mat2[id(j,i,d)];
        }
    }
}

void sgd_pca(float* X, float* U, float* IdMat)
{
    float *zero_mat = 0;
    zero_mat = (float*)malloc(sizeof(zero_mat[0])*d*k); 
    initZero(zero_mat,k,d);

    // Alloc delta_U
    float *delta_U = 0;
    delta_U = (float*)malloc(d*k*sizeof(float));
    
    float* part_delta_U = 0;
    part_delta_U = (float*)malloc(d*k*sizeof(float));

    float* x_2 = 0;
    x_2 = (float*)malloc(d*d*sizeof(float));
    float* x = 0;
    x = (float*)malloc(d*sizeof(float));
    float* tau = 0;
    tau = (float*)malloc(d*sizeof(float));
    float* new_dU = 0;
    new_dU = (float*)malloc(d*k*sizeof(float));
    float alpha = 1.0;
    float beta = 0.;
    float base_lr = lr;
    float one_batch = 1./batch_size; //for divison of delta_U

    float* work;
    int work_size = 0;
    int* info = 0;
    int SelectID = 0;
    //cudaMalloc((void**)&dev_info, sizeof(int));

    double dtime;
    for(int i=0 ; i<max_iter ; i++)
    {
       clock_t start=clock();
       printf("iter:%d\n", i);     

       // [...d...] [...d...] [...d...] ...n

       memcpy(delta_U, zero_mat, sizeof(zero_mat[0])*d*k);
       memcpy(part_delta_U, zero_mat, sizeof(zero_mat[0])*d*k);
       memcpy(new_dU, zero_mat, sizeof(zero_mat[0])*d*k);

       //rand for mini-batch 
       SelectID = rand() % (n/batch_size);
       
       for(int j=0 ; j<batch_size ; j++)
       {
            //fetch data
            memcpy(x, X+SelectID*batch_size*d + j*d, d);
            
            //delta_U = base_lr * x * x^T * U
            //x_2 = x * x
            cblas_sgemm(order, ntran, ytran, d, d, 1, 
                        alpha, x, d, x, d, beta, x_2, d); 
            //delta_U = base_lr * x_2 * U
            cblas_sgemm(order, ntran, ytran, d, k, d, 
                        base_lr, x_2, d, U, d, beta, part_delta_U, d);
            //device_delta_U = device_delta_U + d_part_delta_U
            mat_add(part_delta_U, delta_U, delta_U, k, d, alpha, alpha);
            //cblas_sgeam(order, ntran, ntran, d, k, alpha, part_delta_U, d, 
            //            alpha, delta_U, d, delta_U, d);
       }


       //U = U + delta_U
       mat_add(U, delta_U, U, k, d, alpha, one_batch);
       //cblas_sgeam(order, ntran, ntran, d, k, alpha, U, d, 
       //            one_batch, delta_U, d, U, d);


       //QR decomposition buffer query
       //LAPACKE_sgeqrfp_work(LAPACK_COL_MAJOR, d, k, U, d, tau, work, work_size);
       //cusolverDnSgeqrf_bufferSize(solver_handle, d, k, device_U, d, &work_size);

       //cudaMalloc(&d_work, work_size * sizeof(float));
       //work = (float*)malloc(work_size*sizeof(float));
       //QR decomposition
       //checkCudaErrors3(cusolverDnSgeqrf(solver_handle, d, k, device_U, d, d_tau, d_work, work_size, dev_info));
       LAPACKE_sgeqrf(LAPACK_COL_MAJOR, d, k, U, d, tau);
       //Set new_dU to identity matrix
       //cudaMemcpy(new_dU, IdMat, sizeof(float)*d*k, cudaMemcpyHostToDevice);
       memcpy(new_dU, IdMat, sizeof(float)*d*k);

       //Get Q into new_dU
       //checkCudaErrors3(cusolverDnSormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, d, k, k, device_U, d, 
       //            d_tau, new_dU, d, d_work, work_size, dev_info));
       LAPACKE_sormqr(LAPACK_COL_MAJOR, 'L', 'N', d, k, k, U, d, tau, new_dU, d);

       //Copy new_dU to device_U 
       memcpy(U, new_dU, sizeof(float)*d*k);
       dtime = ((double)clock()-start)/CLOCKS_PER_SEC; 
       printf("\nTime for single iteration: %f\n", dtime); // call gs_pca_cublas
       if(i>0 && i % 10 == 0)
            Calc_trace(U, X);

    }

    free(delta_U);
    free(x_2);
    free(x);
    free(new_dU);
    free(IdMat);

}

void ReadInMat(const char* FileName, float* X)
{
    FILE* f = fopen(FileName, "r");
    char line[1000000] = "";
    char* pch;
    int j=0;
    for(int i=0 ; i<d ; i++)
    {
        j=0;
        fprintf(stderr,"col: %d\n", i);
        //fscanf(f, "%s", line);
        fgets(line , 1000000, f);
        pch = strtok(line, " ");
        while(pch!=NULL)
        {
            X[id(j, i, d)] = atof(pch);
            pch = strtok(NULL, " ");

            j++;

        }
        //fprintf(f, "\n");
    }
    fclose(f);
}

int main(int argc, char** argv) {
    //SGD model for PCA
    // U = U + delta_U

    printf("Problem dimensions: MxN=%dx%d, K=%d\n", d, n, k); // initialize srand and clock
    //Data generation, random normalized
    float *X;
    X = (float*)malloc(d*n * sizeof(X[0]));
    ReadInMat("/home/extra/pcjeff/GPGPU/gpu_final/sgdpca/random_mat.txt", X);
    
    if(X == 0) {
        fprintf (stderr, "! host memory allocation error: X\n"); return EXIT_FAILURE;
    }
    double av = 0.;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < d; j++) {
            av+=X[id(j, i, d)];//rand() / (float)RAND_MAX;
        } 
    }
    av = av/(d*n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < d; j++) {
            X[id(j, i, d)] -= av;//rand() / (float)RAND_MAX;
        } 
    }

    float* U;
    U = (float*)malloc(d*k * sizeof(U[0]));
    if(U == 0)
    {
        fprintf(stderr, "! host memory allocation error: T\n"); return EXIT_FAILURE;
    }
    for(int i = 0; i < k; i++) {
        for(int j = 0; j < d; j++) {
            U[id(j, i, d)] = rand() / (float)RAND_MAX;
        } 
    }
    
    float* IdMat = 0;
    IdMat = (float*)malloc(d*k*sizeof(IdMat[0]));
    initIdentity(IdMat, k, d);
    
    double dtime;
    clock_t start=clock();
    sgd_pca(X, U, IdMat);   
    dtime = ((double)clock()-start)/CLOCKS_PER_SEC; 
    printf("\nTime for PCA: %f\n", dtime); // call gs_pca_cublas

    free(IdMat);
    free(X);
    free(U);

    return 0;

}
