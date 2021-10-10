import numpy as np

class Adam:
    def __init__(self, lr, u0 = 0.9, u1=0.999, e=1e-8):
        self.lr = lr
        self.u0 = u0
        self.u1 = u1
        self.e = e
        self.mt = 0
        self.vt = 0
    def __call__(self, grad):
        self.mt = self.u0 * self.mt + (1 - self.u0) * grad
        self.vt = self.u1 * self.vt + (1 - self.u1) * grad ** 2
        return -1 * self.lr * self.mt / (self.vt ** 0.5 + self.e)
