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
