import sklearn
import numpy as np
import random
from joblib import Parallel, delayed
from sklearn import decomposition

d=1000
n=1000
k=5

max_iter = 50
base_lr = 0.01
batch_size = 300

#X = np.matrix( np.random.uniform(-1, 1, (d, n))) #data
X = np.matrix(np.full((d,n), 0.5)) #data


def sk_pca():
    pca = decomposition.IncrementalPCA(n_components=k, batch_size=10)
    return pca.fit_transform(np.transpose(X)) #fitting data martix X
    

def get_data():
    #only support d,1 matrix data
    return X[ :, random.randint(0, n-1) ] # d, 1 matrix

def Meausrement(U):

    sgd_pca_output = np.transpose( U ) * X
    sk_pca_output = np.transpose( sk_pca() )
    
    print 'sklearn optimal trace'
    print np.matrix.trace( np.dot(sk_pca_output , np.transpose(sk_pca_output)) )

    print 'sklear pca , sgd pca'
    print np.linalg.norm(sk_pca_output), np.linalg.norm(sgd_pca_output)
    print np.linalg.norm(sk_pca_output-sgd_pca_output)
    print 'sklearn pca'
    print sk_pca_output
    print '================================'
    print 'sgd pca'
    print sgd_pca_output

def Calc_delta(U):
    rand = random.randint(0, n-1)
    x = X[:, rand]

    print 
    return base_lr * ( x * np.transpose(x) ) * U

def sgd_pca():
    U = np.matrix(np.random.rand(d,k)) #projection matrix
    delta_U = np.matrix(np.random.rand(d,k)) #projection matrix
    for iter_num in xrange(max_iter):
        #x = get_data(batch_size) #d, x(batch_size) matrix
        for ind in xrange(batch_size):
            rand = random.randint(0, n-1)
            x = X[:, rand]
            delta_U += base_lr * ( x * np.transpose(x) ) * U
        U = U + delta_U
        Q, R = np.linalg.qr(U)
        U = Q
        print iter_num
        print np.matrix.trace(np.transpose(U) * X * np.transpose(X) * U )
    
        if iter_num%50 == 0:
            Meausrement(U)
        
    return U 

def main():

    sk_pca_output = np.transpose( sk_pca() )
    out = np.matrix(sk_pca_output)
    print np.matrix.trace( out * np.transpose(out) )


if __name__ == '__main__':
    main()
