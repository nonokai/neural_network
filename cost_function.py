import numpy as np

class CrossEntropyCost:

    @staticmethod
    def fn(a, y):
        return np.nan_to_num(np.sum(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def nabla(z, a, y):
        return (a-y) / (np.exp(-z) / (1 + np.exp(-z))**2)

class QuadraticCost:

    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a-y)**2

    @staticmethod
    def nabla(z, a, y):
        return (a-y)
