import numpy as np
from mnist import MNIST
from sklearn.decomposition import RandomizedPCA, PCA
from numpy.linalg import norm as norm
from numpy.linalg import eig as eig

MNIST_PATH = '/home/master/04/weitang114/GPUProgramming/final/python-mnist/data'

data = np.array(MNIST(MNIST_PATH).load_testing()[0])


def baseline_randomizedpca(data):
    pca = RandomizedPCA(n_components=20).fit(data)
    x = pca.transform(data)
    return x

def baseline_pca(data, k):
    pca1 = PCA(n_components=k).fit(data)
    x1 = pca1.transform(data)
    return x1


def vec2mat(vec):
    return np.matrix(vec).transpose()

def mat2arr(mat):
    return np.array(mat)

def inc_pca(X, k):
    mean = X.mean(axis=0)
    X = X - mean
    
    n, d = X.shape
    U = np.zeros((d,k))
    S = np.zeros((k,k))
    M = np.zeros((k+1, k+1))
    TMP = np.zeros((d, k+1))
    rank = k
    
    cnt = 0
    while True:
        if cnt % 100 == 0:
            print 'iter', cnt
        Ut = U.transpose()
        x = X[np.random.randint(n)]
        x_h = vec2mat(Ut.dot(x))
        x_o = vec2mat(x - U.dot(Ut.dot(x)))
        # print 'x_h', x_h
        # print 'x_o', x_o
        M[:k, :k] = S + x_h * x_h.transpose()
        M[-1, :k] = (norm(x_o) * x_h.transpose()).A1
        M[:k, -1] = (norm(x_o) * x_h).A1
        M[-1, -1] = norm(x_o) ** 2
        
        evs, U_ = eig(M)

        rm_idx = evs.argmin()
        evs = np.delete(evs, rm_idx)
        U_ = np.delete(U_, rm_idx, 1)


        # idx = evs.argsort()[::-1]
        # evs, U_ = evs[idx], U_[:,idx]
        # # evs = evs[:-1]
        # # U_ = U_[:, :-1]
        # evs = evs[1:]
        # U_ = U_[:, 1:]


        # print evs
        # print 'U_', U_
        S_ = np.diag(evs)
        # print S_.shape
        
        TMP[:, :k] = U
        TMP[:, -1] = (x_o / norm(x_o)).A1
        # print 'TMP[:,-1]', TMP[:,-1]
        # print TMP
        U = TMP.dot(U_)
        # print 'U', U.mean()
        S = S_ 
        # print S
        # print U.shape, S.shape
    
        cnt += 1
        if cnt == 10000:
            break

    # print U
    print U.shape
    # print X.dot(U)
    # result = U.transpose().dot(X.transpose())
    result = X.dot(U)

    return result

# data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float)
# data = np.random.rand(10, 30)

k = 10 

a = baseline_pca(data, k)
b = inc_pca(data, k)


print a
print b
print norm(a)
print norm(b)
