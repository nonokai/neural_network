import numpy as np

class Sigmoid:

    @classmethod
    def f(cls, x):
        return 1.0 / (1.0+np.exp(-x))

    @classmethod
    def prime(cls, x):
        return cls.f(x) * (1-cls.f(x))

class Tanh:

    @classmethod
    def f(cls, x):
        return np.tanh(x)

    @classmethod
    def prime(cls, x):
        return 1 - cls.f(x)**2

class ReLU:

    @classmethod
    def f(cls, x):
        return np.maximum(0, x)

    @classmethod
    def prime(cls, x):
        return np.where(x >= 0, 1, 0)

class Linear:

    @classmethod
    def f(cls, x):
        return x

    @classmethod
    def prime(cls, x):
        return np.ones_like(x)

class Softmax:

    @classmethod
    def f(cls, x):
        exp = np.exp(x)
        return exp / sum(exp)
