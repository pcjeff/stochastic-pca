// C/C++ example for the CUBLAS (NVIDIA)
// implementation of PCA-GS algorithm //
// M. Andrecut (c) 2008
// includes, system
#include <stdio.h> 
#include <cstdlib>
#include <iostream>
#include <string.h> 
#include <time.h>
#include <float.h>
#include <cmath>
#include <ctime>
#include <sys/time.h>
// includes, cuda 
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "magma.h"

using namespace std;

// matrix indexing convention
#define id(i, j, ld) (((j) * (ld) + (i)))
#define checkmem(x) do { \
        if (x == 0) {  \
            fprintf(stderr, "! host memory allocation error: T\n"); \
            return EXIT_FAILURE; \
        } \
    } while (0);
       
#ifdef DEBUGMODE
    #define DEBUG(x) do { std::cerr << "[DEBUG] " << x << endl;  } while (0)
#else
    #define DEBUG(x)
#endif

// declarations
int gs_pca_cublas(int, int, int, double *, double *, double *);
int inc_pca_cublas(cublasHandle_t &handle, double *X, int M, int N, int K, int T, double *Y);
void subtract_mean(double *X, int m, int n);
int identity(double *X, int d);
int argmin(const double *vec, int n);
void print_mat(double *X, int d, int n, const char *name);
void print_mat_gpu(double *dX, int d, int n, const char *name);
double calc_trace(cublasHandle_t handle, int M, int N, int K, double *device_U, double *device_X);
void checkCublas(cublasStatus_t stat);

int print_results(int, int, int, double *, double *, double *, double *);
// main
int main(int argc, char** argv) {
    // PCA model: X = TP’ + R
    // input: X, MxN matrix (data)
    // input: M = number of rows in X
    // input: N = number of columns in X
    // input: K = number of components (K<=N)
    // output: T, MxK scores matrix // output: P, NxK loads matrix // output: R, MxN residual matrix
    /*int M = 5000, m; int N = 5000, n; int K = 10;*/
    /*int M = 1000, m; int N = M/2, n; int K = 10;*/
    int M = 10000, N = 10000, K = 10, T = 50000;
    cublasHandle_t handle;

    printf("\nProblem dimensions: MxN=%dx%d, K=%d", M, N, K); // initialize srand and clock
    srand (time(NULL));
    magma_init();
    checkCublas(cublasCreate(&handle));

    // initiallize some random test data X 
    double *X;
    X = (double*)malloc(M * N * sizeof(X[0]));
    if(X == 0) {
        fprintf (stderr, "! host memory allocation error: X\n"); return EXIT_FAILURE;
    }
    for(int n = 0; n < N; n++) {
        for(int m = 0; m < M; m++) {
            X[id(m, n, M)] = rand() / (double)RAND_MAX;
        } 
    }
    X[0] = 1;
    X[1] = 3;
    X[2] = 5;
    X[3] = 2;
    X[4] = 3;
    X[5] = 1;
    X[6] = 3;
    X[7] = 1;
    X[8] = 2;
    X[9] = 1;
    X[10] = 2;
    X[11] = 1;
    
    print_mat(X, M, N, "\nX");
     
    double *Y;
    Y = (double*)malloc(K * N * sizeof(Y[0]));

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    inc_pca_cublas(handle, X, M, N, K, T, Y);

    double norm = 0;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            norm += Y[id(i, j, K)] * Y[id(i, j, K)];
        }
    }
    norm = sqrt(norm);
    cout << "Y norm2:" << norm << endl;
    
    print_mat(Y, K, N, "\nY");

    gettimeofday(&end_time, NULL);
    float seconds = end_time.tv_sec - start_time.tv_sec +
            (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    
    cout << "Time elapsed:" << seconds << " seconds." << endl;

    // clean up memory 
    free(X);
    free(Y);
    
    return EXIT_SUCCESS; 
}



int inc_pca_cublas(cublasHandle_t &handle, double *X, int M, int N, int K, int T, double *Y) {
    cudaError_t cudaStat;    
    cublasStatus_t stat;

    double *dX = 0;
    cudaMalloc((void**)&dX, M * N * sizeof(double));
    cudaMemset(dX, 0, M * N * sizeof(double));
    
    double *dY = 0;
    cudaMalloc((void**)&dY, K * N * sizeof(double));
    cudaMemset(dY, 0, K * N * sizeof(double));

    double *dU = 0;
    cudaMalloc((void**)&dU, M * K * sizeof(double));
    cudaMemset(dU, 0, M * K * sizeof(double));

    double *hU_ = (double*) calloc((K+1) * K, sizeof(double));

    double *dU_ = 0;
    cudaMalloc((void**)&dU_, (K+1) * K * sizeof(double));
    cudaMemset(dU_, 0, M * K * sizeof(double));
    
    double *dS = 0;
    cudaMalloc((void**)&dS, K * K * sizeof(double));
    cudaMemset(dS, 0, K * K * sizeof(double));

    double *hS = 0;
    hS = (double*) calloc(K * K, sizeof(double)); // zeros

    double *dP = 0;
    cudaMalloc((void**)&dP, (K+1) * (K+1) * sizeof(double));
    cudaMemset(dP, 0, (K+1) * (K+1)  * sizeof(double));
    
    double *dTMP = 0;
    cudaMalloc((void**)&dTMP, M * (K+1) * sizeof(double));
    cudaMemset(dTMP, 0, M * (K+1)  * sizeof(double));

    double *dx = 0;
    cudaMalloc((void**)&dx, M * sizeof(double));
    cudaMemset(dx, 0, M * sizeof(double));
    
    double *dx_h = 0;
    cudaMalloc((void**)&dx_h, K * sizeof(double));
    cudaMemset(dx_h, 0, K * sizeof(double));
    
    double *dx_o = 0;
    cudaMalloc((void**)&dx_o, M * sizeof(double));
    cudaMemset(dx_o, 0, M * sizeof(double));

    double *dxh_xht = 0;
    cudaMalloc((void**)&dxh_xht, M * M * sizeof(double));
    cudaMemset(dxh_xht, 0, M * M * sizeof(double));

    double *dI_K = 0;
    cudaMalloc((void**)&dI_K, K * K * sizeof(double));
    identity(dI_K, K);

    double *dI_M = 0;
    cudaMalloc((void**)&dI_M, M * M * sizeof(double));
    identity(dI_M, M);
    

    double *hP = 0;
    hP = (double*) malloc((K+1) * (K+1) * sizeof(double));

    DEBUG("Initing magma variables..");

    // eigen decomposition with magma
    double *wr = 0;// (double*) malloc(N * sizeof(double));
    double *wi = 0;//(double*) malloc(N * sizeof(double));
    double *V = 0;//(double*) malloc(N * N * sizeof(double));
    double *dV = 0;
    int nb = magma_get_dgehrd_nb(K+1);
    int lwork = (K+1)*(2 + 2*nb);
    double *work = 0;//(double*) malloc(lwork * sizeof(double));
    magma_int_t info;
    magma_malloc_cpu( (void**) &wr, (K+1)*sizeof(double) );
    magma_malloc_cpu( (void**) &wi, (K+1)*sizeof(double) );
    magma_malloc_cpu( (void**) &V, (K+1) * (K+1)*sizeof(double) );
    magma_malloc_cpu( (void**) &work, (lwork)*sizeof(double) );
    cudaMalloc((void**)&dV, (K+1) * (K+1) * sizeof(double));
    cudaMemset(dV, 0, (K+1) * (K+1) * sizeof(double));

    // X = X - X.mean(axis=0)
    subtract_mean(X, M, N);

    double alpha(1.0), beta(0.0);
    cublasSetMatrix (N, M, sizeof(double), X, N, dX, N);

    
    DEBUG("Starting iterations...");


    for (int t = 0; t < T; t++) {
        /*if (t % 100 == 0) {*/
            /*cout << "" << t << endl;*/
        /*}*/
        DEBUG("================================");
        DEBUG("===========" << t << "============");
        DEBUG("================================");

        int ind = rand() % N;        
        /*int ind = t % N;*/
        stat = cublasDcopy(handle, M, dX + (M*ind), 1, dx, 1);
            
        DEBUG("random index:" << ind);
        print_mat_gpu(dx, M, 1, "x");
        
        // x_h = Ut * x
        DEBUG("x_h = Ut * x");
        alpha = 1.0;
        beta = 0.0;
        stat = cublasDgemv(handle, CUBLAS_OP_T, M, K, 
                &alpha, dU, M, dx, 1, &beta, dx_h, 1);
        print_mat_gpu(dx_h, K, 1, "dx_h");
        
        // x_o = x - U Ut x
        DEBUG("x_o = x - U Ut x");
        alpha = -1.0;
        beta = 1.0;
        cublasDcopy(handle, M, dx, 1, dx_o, 1);
        stat = cublasDgemv(handle, CUBLAS_OP_N, M, K, 
                &alpha, dU, M, dx_h, 1, &beta, dx_o, 1);
        
        print_mat_gpu(dx_o, M, 1, "dx_o");
        
        // xh * xh_t
        DEBUG("xh * xh_t");
        alpha = 1.0;
        beta = 0.0;
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, K, 1,
                &alpha, dx_h, K, dx_h, K, &beta, dxh_xht, K);
        print_mat_gpu(dxh_xht, K, K, "dx_h * dx_hT");

        print_mat_gpu(dS, K, K, "dS");

        // P[:k, :k] = S + xh_xht
        cudaMemset(dP, 0, (K+1) * (K+1)  * sizeof(double));
        
        DEBUG("P[:k, :k]");
        alpha = 1.0;
        beta = 1.0;
        cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, K,
                &alpha, dS, K, &beta, dxh_xht, K, dP, K+1);

        DEBUG("xo_norm");
        double xo_norm = 0.0;
        stat = cublasDnrm2(handle, M, dx_o, 1, &xo_norm);
        DEBUG("xo_norm:" << xo_norm);

        // M[-1, :k] = (norm(x_o)) * xh_t
        alpha = xo_norm;
        stat = cublasDaxpy(handle, K, &alpha, dx_h, 1, dP + K, K+1);
        
        // M[:k, -1] = (norm(x_o)) * xh_t
        alpha = xo_norm;
        stat = cublasDaxpy(handle, K, &alpha, dx_h, 1, dP + K*(K+1), 1);

        // M[-1, -1] = norm(x_o) ^ 2
        double xo_norm_sq = xo_norm * xo_norm;
        DEBUG("xo_norm_sq:" << xo_norm_sq);
        cudaMemcpy(dP + (K+1)*(K+1) - 1, &xo_norm_sq, sizeof(double), cudaMemcpyHostToDevice);
        print_mat_gpu(dP, K+1, K+1, "dP");
        
        cudaMemcpy(hP, dP, (K+1) * (K+1) * sizeof(double), cudaMemcpyDeviceToHost);
        print_mat(hP, K+1, K+1, "hP");
        
        
        // eigen decomposition
        // wr: real eigenvalues
        // V: right eigenvectors
        magma_dgeev_m(MagmaNoVec, MagmaVec, K+1,
                hP, K+1, wr, wi,
                NULL, K+1, V, K+1,
                work, lwork, &info);
        
        print_mat(V, K+1, K+1, "eigen vectors");
        print_mat(wr, K+1, 1, "wigenvalues");

        
        int rm_idx = argmin(wr, K+1);
        for (int i = 0, cnt = 0; i < K+1; i++) {
            if (i != rm_idx) {
                DEBUG("V[" << i << "]");
                hS[id(cnt, cnt, K)] = wr[i];
                for (int j = 0; j < K+1; j++) {
                    /*cout << " " << V[id(j,i,K+1)];*/
                    hU_[id(j, cnt, K+1)] = V[id(j, i, K+1)];
                }
                /*cout << endl;*/
                cnt ++;
            }
        }
        print_mat(hU_, K+1, K, "hU_");
        cudaMemcpy(dU_, hU_, (K+1)*K*sizeof(double), cudaMemcpyHostToDevice);
        print_mat_gpu(dU_, K+1, K, "dU_");
        cudaMemcpy(dS, hS, (K * K) * sizeof(double), cudaMemcpyHostToDevice);

        // TMP[:, :k] = U
        // TMP[:, :k] = 1.0 * U * I_k + 0.0 * TMP[:, :k] , gemm
        // 這是不是可以用SetMatrix  //TODO
        cudaMemset(dTMP, 0, M * (K+1) * sizeof(double));
        alpha = 1.0;
        beta = 0.0;
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, K, 
                &alpha, dU, M, dI_K, K,
                &beta, dTMP, M);
        
        // TMP[:, -1] = (x_o / norm(x_o))
        alpha = 1.0 / xo_norm;
        stat = cublasDaxpy(handle, M, &alpha, dx_o, 1, &dTMP[id(0, K, M)], 1);

        print_mat_gpu(dTMP, M, K+1, "dTMP");

        // U{M,K} = TMP{M,K+1} * U_{K+1,K}
        alpha = 1.0;
        beta = 0.0;
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, K+1,
                &alpha, dTMP, M, dU_, K+1,
                &beta, dU, M);
        print_mat_gpu(dU, M, K, "dU");
        
        if (t % 100 == 0) {
            double objective = calc_trace(handle, M, N, K, dU, dX);
            cout << t << "," << objective << endl;
        }
    }
    double objective = calc_trace(handle, M, N, K, dU, dX);
    cout << "finish," << objective << endl;

    cout << "Finish iterations." << endl;

    // Y{K,N} = U_t{K, M}.dot(X{M, N})
    alpha = 1.0;
    beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, N, M,
            &alpha, dU, M, dX, M,
            &beta, dY, K);  

    cublasGetMatrix(K, N, sizeof(double), dY, K, Y, K);
    
   
    // clean up
    magma_free_cpu(work);
    magma_free_cpu(V);
    magma_free_cpu(wi);
    magma_free_cpu(wr);
    cerr << "finish clearning up magmas" << endl;

    cublasDestroy(handle);
    cudaFree(dV);
    cudaFree(dI_M);
    cudaFree(dI_K);
    cudaFree(dxh_xht);
    cudaFree(dx_o);
    cudaFree(dx_h);
    cudaFree(dx);
    cudaFree(dTMP);
    cudaFree(dP);
    cudaFree(dS);
    cudaFree(dU_);
    cudaFree(dU);
    cudaFree(dY);
    cudaFree(dX);
    cerr << "finish clearning up cuda memories" << endl;

    free(hP);
    free(hS);
    free(hU_);
    cerr << "finish clearning up host memories" << endl;

    return 0;
}

void subtract_mean(double *X, int m, int n) {
    for (int i = 0; i < m; i++) {
        double mean = 0.0;
        for (int j = 0; j < n; j++) {
            mean += X[id(i, j, m)];
        }
        mean /= n;
        for (int j = 0; j < n; j++) {
            X[id(i, j, m)] -= mean;
        }
    }
}

int argmin(const double *vec, int n) {
    int min_idx = -1;
    double min = DBL_MAX;
    for (int i = 0; i < n; i++) {
        if (vec[i] < min) {
            min_idx = i;
            min = vec[i];
        }
    }

    return min_idx;
}


__global__ void initIdentityGPU(double *devMatrix, int dim) {
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if(y < dim && x < dim) {
          if(x == y)
              devMatrix[x * dim + y] = 1.0;
          else
              devMatrix[x * dim + y] = 0.0;
    }
}

int identity(double *X, int d) {
    dim3 griddim(64, 64);
    dim3 blocksize(32,32);
    initIdentityGPU<<<griddim, blocksize>>>(X, d);
    return 0;
}

void print_mat(double *X, int d, int n, const char *name) {
#ifdef DEBUGMODE
    cout << endl << "MAT:" << name << endl;
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            cout << X[id(i, j, d)] << " ";
        }
        cout << endl;
    }
#endif
}

void print_mat_gpu(double *dX, int d, int n, const char *name) {
#ifdef DEBUGMODE
    double *hX = (double*) malloc(d * n * sizeof(dX[0]));
    cudaMemcpy(hX, dX, d * n * sizeof(dX[0]), cudaMemcpyDeviceToHost);

    cout << endl << "MAT on GPU:" << name << endl;
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            cout << hX[id(i, j, d)] << " ";
        }
        cout << endl;
    }
    
    free(hX);
#endif
}

double sum_diagnal(double* final_output, int d1, int d2) {
    double tr = 0.;
    for(int i=0 ; i<d1 ; i++) {
        for(int j=0 ; j<d2 ; j++) {
            if(i==j){
                tr+=final_output[id(j,i,d2)];
            }
        }
    }
    return tr;
}

// X: MxN, U:MxK
double calc_trace(cublasHandle_t handle, int M, int N, int K, double* device_U, double* device_X) {    
    const double one= 1.0;
    const double zero = 0.0;
    double* output = 0;//Matrix for U^TX
    cudaMalloc((void**)&output, K*N*sizeof(double));
    double* d_final_output = 0; //Matrix for calc trace
    cudaMalloc((void**)&d_final_output, K*K * sizeof(double));
    double* final_output = 0;
    final_output = (double*) malloc(sizeof(final_output[0])*K*K);

    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, N, M, 
                        &one, device_U, M, device_X, M, &zero, output, K);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, K, N, &one, output, K, output, K, &zero, d_final_output, K);
    cudaMemcpy(final_output, d_final_output, sizeof(double)*K*K, cudaMemcpyDeviceToHost);
    return sum_diagnal(final_output, K, K);
}

static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

void checkCublas(cublasStatus_t stat) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "CUBLAS ERROR:" << _cublasGetErrorEnum(stat) << endl;
    }
}

#if 0

int gs_pca_cublas(int M, int N, int K, double *T, double *P,
        double *R) { // PCA model: X = TP’ + R
    // input: X, MxN matrix (data)
    // input: M = number of rows in X
    // input: N = number of columns in X
    // input: K = number of components (K<=N)
    // output: T, MxK scores matrix // output: P, NxK loads matrix // output: R, MxN residual matrix
    cublasStatus_t status;
    // maximum number of iterations

    int J = 10000;
    // max error 
    double er = 1.0e-7; 
    int n, j, k;
    // transfer the host matrix X to device matrix dR 
    double *dR = 0;
    cudaMalloc(M*N, sizeof(dR[0]), (void**)&dR);
    status = cublasSetMatrix(M, N, sizeof(R[0]), R, M, dR, M);

    // allocate device memory for T, P 
    double *dT = 0;
    cudaMalloc(M*K, sizeof(dT[0]), (void**)&dT);

    double *dP = 0;
    status = cublasAlloc(N*K, sizeof(dP[0]), (void**)&dP);
    if(status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "! device memory allocation error (dP)\n");

        return EXIT_FAILURE; 
    }

    // allocate memory for eigenvalues 
    double *L;
    L = (double*)malloc(K * sizeof(L[0]));;
    if(L == 0) {
        fprintf (stderr, "! host memory allocation error: T\n"); return EXIT_FAILURE;
    }

    // mean center the data 
    double *dU = 0;
    status = cublasAlloc(M, sizeof(dU[0]), (void**)&dU); if(status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf (stderr, "! device memory allocation error (dU)\n"); return EXIT_FAILURE;
    }
    cublasDcopy(M, &dR[0], 1, dU, 1);
    for(n=1; n<N; n++) {
        cublasDaxpy (M, 1.0, &dR[n*M], 1, dU, 1);
    }
    for(n=0; n<N; n++) {
        cublasDaxpy (M, -1.0/N, dU, 1, &dR[n*M], 1); 
    }
    // GS-PCA
    double a;
    for(k=0; k<K; k++) {
        cublasDcopy (M, &dR[k*M], 1, &dT[k*M], 1);

        a = 0.0;
        for(j=0; j<J; j++) {
            cublasDgemv ('t', M, N, 1.0, dR, M, &dT[k*M], 1, 0.0, &dP[k*N], 1);

            if(k>0) {
                cublasDgemv ('t', N, k, 1.0, dP, N, &dP[k*N], 1, 0.0, dU, 1);
                cublasDgemv ('n', N, k, -1.0, dP, N, dU, 1, 1.0, &dP[k*N], 1);
            }
            cublasDscal (N, 1.0/cublasDnrm2(N, &dP[k*N], 1), &dP[k*N], 1);
            cublasDgemv ('n', M, N, 1.0, dR, M, &dP[k*N], 1, 0.0, &dT[k*M], 1);
            if(k>0) {
                cublasDgemv ('t', M, k, 1.0, dT, M, &dT[k*M], 1, 0.0, dU, 1);
                cublasDgemv ('n', M, k, -1.0, dT, M, dU, 1, 1.0, &dT[k*M], 1 );
            }
            L[k] = cublasDnrm2(M, &dT[k*M], 1); cublasDscal(M, 1.0/L[k], &dT[k*M], 1);
            if(fabs(a - L[k]) < er*L[k]) break;
            a = L[k];
        }
        cublasDger (M, N, - L[k], &dT[k*M], 1, &dP[k*N], 1, dR, M); 
    }
    for(k=0; k<K; k++) {
        cublasDscal(M, L[k], &dT[k*M], 1); 
    }
    // transfer device dT to host T
    cublasGetMatrix (M, K, sizeof(dT[0]), dT, M, T, M);
    // transfer device dP to host P
    cublasGetMatrix (N, K, sizeof(dP[0]), dP, N, P, N);
    // transfer device dR to host R
    cublasGetMatrix (M, N, sizeof(dR[0]), dR, M, R, M);
    // clean up memory
    free(L);
    status =cublasFree(dP); 
    status =cublasFree(dT);
    status =cublasFree(dR);
    return EXIT_SUCCESS;
}

int print_results(int M, int N, int K, double *X, double *T, double *P, double *R) {
    int m, n, k;
    // If M < 13 print the results on screen if(M > 12) return EXIT_SUCCESS; printf("\nX\n");
    for(m=0; m<M; m++) {
        for(n=0; n<N; n++) {
            printf("%+f ", X[id( m, n,M)]);
        }
        printf("\n");
    }
    printf("\nT\n");
    for(m=0; m<M; m++) {
        for(n=0; n<K; n++) {
            printf("%+f ", T[id(m, n, M)]); 
        }
        printf("\n");
    }
    double a;
    printf("\nT’ * T\n");
    for(m = 0; m<K; m++) {
        for(n=0; n<K; n++) {
            a=0;
            for(k=0; k<M; k++)
            {
                a = a + T[id(k, m, M)] * T[id(k, n, M)]; }
            printf("%+f ", a);
        }
        printf("\n");
    }
    printf("\nP\n");
    for(m=0; m<N; m++) {
        for(n=0; n<K; n++){
                printf("%+f ", P[id(m, n, N)]);
        } 
        printf("\n");
    }
    printf("\nP' * P\n");
    

    for(m = 0; m<K; m++) {
        for(n=0; n<K; n++) {
            a=0; 
            for(k=0; k<N; k++) a = a + P[id(k, m, N)] * P[id(k, n, N)]; 
            printf("%+f ", a);
        }
        printf("\n");
    }

    printf("\nR\n");
    for(m=0; m<M; m++) {
        for(n=0; n<N; n++) {
            printf("%+f ", R[id( m, n,M)]);
        }
        printf("\n");
    }
    return EXIT_SUCCESS;
}
#endif

