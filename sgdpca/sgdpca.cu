// C/C++ example for the CUBLAS (NVIDIA)
// implementation of SGD-PCA algorithm //
#include <cstdio>
#include <cstdlib>
#include <string.h>
// includes, cuda 
#include <cublas.h>
#include <cusolverDn.h>
#include <time.h>
#include <omp.h>

#define id(m, n, ld) (((n) * (ld) + (m)))
#define d 10000
#define n 10000
#define k 10
#define max_iter 20
#define lr 0.01
#define batch_size 50

void checkCudaErrors(cublasStatus status)
{
    if(status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "! CUDA error: %d\n", status);
    }
}

void checkCudaErrors2(cublasStatus status)
{
    //for cublas
    if(status == CUBLAS_STATUS_NOT_INITIALIZED)
        printf("CUBLAS_STATUS_NOT_INITIALIZED\n");
    else if(status == CUBLAS_STATUS_INVALID_VALUE)
        printf("CUBLAS_STATUS_INVALID_VALUE\n");
    else if(status == CUBLAS_STATUS_ARCH_MISMATCH)
        printf("CUBLAS_STATUS_ARCH_MISMATCH\n");
    else if(status == CUBLAS_STATUS_EXECUTION_FAILED)
        printf("CUBLAS_STATUS_EXECUTION_FAILED\n");
    else if(status == CUBLAS_STATUS_SUCCESS)
        ;//printf("CUBLAS_STATUS_SUCCESS\n");
    else
        printf("unkonwn error\n");

}

void print_matrix(float* dmat, int d1, int d2)
{
    float* mat = (float*)malloc(sizeof(float)*d1*d2);
    cudaMemcpy(mat, dmat, sizeof(float)*d1*d2, cudaMemcpyDeviceToHost);

    for(int i=0 ; i<d1 ; i++)
    {
        for(int j=0 ; j<d2 ; j++)
        {
            
            printf("%d, %d, :%lf |", i, j, mat[id(j, i, d2)]);
        }
        printf("\n");
    }
}

void checkCudaErrors3(cusolverStatus_t status)
{
    //for cusolver 
    if(status == CUSOLVER_STATUS_NOT_INITIALIZED)
        printf("CUSOLVER_STATUS_NOT_INITIALIZED\n");
    else if(status == CUSOLVER_STATUS_INVALID_VALUE)
        printf("CUSOLVER_STATUS_INVALID_VALUE\n");
    else if(status == CUSOLVER_STATUS_ARCH_MISMATCH)
        printf("CUSOLVER_STATUS_ARCH_MISMATCH\n");
    else if(status == CUSOLVER_STATUS_INTERNAL_ERROR)
        printf("CUSOLVER_STATUS_INTERNAL_ERROR\n");
    else if(status == CUSOLVER_STATUS_SUCCESS)
        ;//printf("CUBLAS_STATUS_SUCCESS\n");
    else
        printf("unkonwn error\n");

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

void Calc_trace(cublasHandle_t handle, float* device_U, float* device_X)
{    
    const float one= 1.0;
    const float zero = 0.0;
    float* output = 0;//Matrix for U^TX
    checkCudaErrors(cublasAlloc(k*n, sizeof(float), (void**)&output));
    float* d_final_output = 0; //Matrix for calc trace
    checkCudaErrors(cublasAlloc(k*k, sizeof(float), (void**)&d_final_output));
    float* final_output = 0;
    final_output = (float*)malloc(sizeof(final_output[0])*k*k);

    
    checkCudaErrors2(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, n, d, 
                        &one, device_U, d, device_X, d, &zero, output, k));
    checkCudaErrors2(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, k, k, n, &one, output, k, output, k, &zero, d_final_output, k));
    cudaMemcpy(final_output, d_final_output, sizeof(float)*k*k, cudaMemcpyDeviceToHost);    
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


void sgd_pca(float* X, float* U, float* IdMat)
{
    float *zero_mat = 0;
    zero_mat = (float*)malloc(sizeof(zero_mat[0])*d*k); 
    initZero(zero_mat,k,d);
    // Alloc device matrix X
    float *device_X = 0;
    checkCudaErrors(cublasAlloc(d*n, sizeof(device_X[0]), (void**)&device_X));
    // copy X to device_X
    cudaMemcpy(device_X, X, sizeof(float)*d*n, cudaMemcpyHostToDevice);

    // Alloc device matrix U
    float *device_U = 0;
    checkCudaErrors(cublasAlloc(d*k, sizeof(device_U[0]), (void**)&device_U));

    // copy U to device_U
    cudaMemcpy(device_U, U, sizeof(U[0])*d*k, cudaMemcpyHostToDevice);

    // Alloc delta_U
    float *device_delta_U = 0;
    checkCudaErrors(cublasAlloc(d*k, sizeof(device_delta_U[0]), (void**)&device_delta_U));
    
    float* d_part_delta_U = 0;
    checkCudaErrors(cublasAlloc(d*k, sizeof(d_part_delta_U[0]), (void**)&d_part_delta_U));

    float* dev_idmat = 0;
    checkCudaErrors(cublasAlloc(d*k, sizeof(dev_idmat[0]), (void**)&dev_idmat));
    cudaMemcpy(dev_idmat, IdMat, sizeof(IdMat[0])*d*k, cudaMemcpyHostToDevice);

    float* x_2 = 0;
    checkCudaErrors(cublasAlloc(d*d, sizeof(float), (void**)&x_2));
    float* x = 0;
    checkCudaErrors(cublasAlloc(d, sizeof(float), (void**)&x));
    float* d_tau = 0;
    checkCudaErrors(cublasAlloc(d, sizeof(float), (void**)&d_tau));
    float* new_dU = 0;
    checkCudaErrors(cublasAlloc(d*k, sizeof(float), (void**)&new_dU));
    float alpha = 1.0;
    float beta = 0.;
    float base_lr = lr;
    float one_batch = 1./batch_size; //for divison of delta_U
    cublasHandle_t handle;  
    cublasCreate(&handle);
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    float* d_work;
    int work_size = 0;
    int* dev_info = 0;
    int SelectID = 0;
    cudaMalloc((void**)&dev_info, sizeof(int));

    for(int i=0 ; i<max_iter ; i++)
    {
       printf("iter:%d\n", i);     

       // [...d...] [...d...] [...d...] ...n

       cudaMemcpy(device_delta_U, zero_mat, sizeof(zero_mat[0])*d*k, cudaMemcpyHostToDevice);
       cudaMemcpy(d_part_delta_U, zero_mat, sizeof(zero_mat[0])*d*k, cudaMemcpyHostToDevice);
       cudaMemcpy(new_dU, zero_mat, sizeof(zero_mat[0])*d*k, cudaMemcpyHostToDevice);

       //rand for mini-batch 
       SelectID = rand() % (n/batch_size);
       
       for(int j=0 ; j<batch_size ; j++)
       {
            //fetch data
            checkCudaErrors(cublasScopy(handle, d, device_X+SelectID*batch_size*d + j*d, 1, x, 1));
            
            //delta_U = base_lr * x * x^T * U
            //x_2 = x * x
            checkCudaErrors2(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, d, d, 1, 
                        &alpha, x, d, x, d, &beta, x_2, d)); 
            //delta_U = base_lr * x_2 * U
            checkCudaErrors2(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, k, d, 
                        &base_lr, x_2, d, device_U, d, &beta, d_part_delta_U, d));
            //device_delta_U = device_delta_U + d_part_delta_U
            checkCudaErrors2(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, k, &alpha, d_part_delta_U, d, 
                        &alpha, device_delta_U, d, device_delta_U, d));
       }


       //U = U + delta_U
       checkCudaErrors2(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, k, &alpha, device_U, d, 
                   &one_batch, device_delta_U, d, device_U, d));

       //QR decomposition buffer query
       checkCudaErrors3(cusolverDnSgeqrf_bufferSize(solver_handle, d, k, device_U, d, &work_size));
       cudaMalloc(&d_work, work_size * sizeof(float));
       //QR decomposition
       checkCudaErrors3(cusolverDnSgeqrf(solver_handle, d, k, device_U, d, d_tau, d_work, work_size, dev_info));
       //Set new_dU to identity matrix
       cudaMemcpy(new_dU, IdMat, sizeof(float)*d*k, cudaMemcpyHostToDevice);

       //Get Q into new_dU
       checkCudaErrors3(cusolverDnSormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, d, k, k, device_U, d, 
                   d_tau, new_dU, d, d_work, work_size, dev_info));

       //Copy new_dU to device_U 
       cudaMemcpy(device_U, new_dU, sizeof(float)*d*k, cudaMemcpyDeviceToDevice);

       Calc_trace(handle, device_U, device_X);

    }

    cudaMemcpy(U, device_U, sizeof(U[0])*d*k, cudaMemcpyDeviceToHost);

    cublasFree(device_U);
    cublasFree(device_X);
    cublasFree(device_delta_U);
    cublasFree(x_2);
    cublasFree(x);
    cublasFree(new_dU);
    cublasFree(dev_idmat);

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
    
    cublasStatus status; 
    status = cublasInit();
    if(status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "! CUBLAS initialization error\n"); return EXIT_FAILURE;
    }

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
