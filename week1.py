import numpy as np
from util.optim import *
mode = 'direct'

# 最小二乘法

class labelGenerator:
    def __init__(self, M, N):
        np.random.seed(1)
        self.A = np.random.random([M, 1])
        self.B = np.random.random()
        self.N = N
        self.M = M

    def __call__(self):
        # 输出的label带有噪声

        X = np.random.random([self.N, self.M]) / np.random.random([self.N, self.M])
        #               (M, 1)
        return X, self.pred(X) + np.random.standard_normal([self.N, 1])/10

    def pred(self, ip):
        "ip: np array of (B,self.M)"
        return (ip) @ self.A + self.B


def grad_linear(Y, X, B):
    '''gradient of B with (Y - XB)T (Y - XB)
    Arg:
        Y numpy array of (B, 1)
        X numpy array of (B, M)
        B numpy array of (M, 1)
    Return:
        grad with shape of (M, 1)
    '''
    return -2 * np.transpose(X) @ (Y - X @ B)

def Hessian_linear(X):
    '''Hessian of B with (Y - XB)T (Y -XB)
     Arg:
        X numpy array of (B, M)
     return:
        Hessian array with shape of (M, M)
    '''
    return 2 * np.transpose(X) @ X

if __name__ == '__main__':
    M = 100     # 线性乘积参数个数
    N = 1000    # 数据量

    gen = labelGenerator(M, N)
    X, label = gen()
    # (N,M) -> (N,M+1)
    X_ = np.concatenate([np.ones([N, 1], dtype=X.dtype), X], axis=1)
    modes = ('direct', 'grad', 'newton')
    A_ = None
    B_ = None
    if mode == modes[0]:
        # B_ = (XT X)-1 XT y

        # (M+1, N)
        X_T_ = np.transpose(X_)

        #               (M+1, M+1)      (M+1, N)  (N, 1) -> (M+1, 1)
        C_ = np.linalg.inv(X_T_ @ X_) @ X_T_ @ label
        # (1)
        B_ = C_[0]
        # (M, 1)
        A_ = C_[1:]

    elif mode == modes[1]:
        optimizer = Adam(1e-3)
        C_ = np.zeros([M + 1, 1], dtype=X.dtype)
        e_stop = 1e-2
        step_stop = 10e5
        n = len(X_)

        error = (np.transpose(label - X_ @ C_) @ (label - X_ @ C_) / n)[0,0]
        i=0
        while error > e_stop and i < step_stop:
            grad = grad_linear(label, X_, C_)
            detla = optimizer(grad)
            C_ = C_ + detla
            error = (np.transpose(label - X_ @ C_) @ (label - X_ @ C_) / n)[0,0]
            i += 1
            print(f'step{i}\t error {error}\t detla {detla[:,0].tolist()}')

        # (1)
        B_ = C_[0]
        # (M, 1)
        A_ = C_[1:]
    elif mode == modes[2]:
        C_ = np.zeros([M + 1, 1], dtype=X.dtype)
        e_stop = 1e-2
        step_stop = 10e5
        n = len(X_)

        error = (np.transpose(label - X_ @ C_) @ (label - X_ @ C_) / n)[0, 0]
        i = 0
        while error > e_stop and i < step_stop:
            grad = grad_linear(label, X_, C_)
            H = Hessian_linear(X_)
            detla = -1 * np.linalg.inv(H) @ grad
            C_ = C_ + detla
            error = (np.transpose(label - X_ @ C_) @ (label - X_ @ C_) / n)[0,0]
            i += 1
            print(f'step{i}\t error {error}\t detla {detla[:,0].tolist()}')
        # (1)
        B_ = C_[0]
        # (M, 1)
        A_ = C_[1:]
    for i in range(100):
        ip_test = np.random.random([1, M]) / np.random.random([1, M])
        ture = gen.pred(ip_test)
        pred = ip_test @ A_ + B_

        print(f'test{i}:\n\tpred\t{pred[0][0]}\n\tture\t{ture[0][0]}')





