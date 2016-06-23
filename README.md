# sgdpca
sgd version for pca

Implementation SGD PCA in paper
http://ttic.uchicago.edu/~klivescu/papers/arora_etal_allerton2012.pdf

CPU version using lapack and cblas

CUDA version using cublas and cusolver

testing input matrix : 10000 samples each is 10000 dimensions
        
|              | time/iter(k=5) |
|:-------------|---------------:|
| CPU version  | 45sec          |
| CUDA version | 0.25sec        |


# incpca
Incremental PCA implementation of: http://ttic.uchicago.edu/~klivescu/papers/arora_etal_allerton2012.pdf

CPU version using Eigen3.

GPU version using CUDA, CUBLAS, Magma, Eigen3.

testing input matrix : 10000 samples, 10000 dim, #principle components (K) = 5

|              | time/100iter(k=5) |
|:-------------|------------------:|
| CPU version  | 0.000428sec       |
| CUDA version | 0.000372sec       |

