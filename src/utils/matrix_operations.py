import numpy as np
class MatrixOperations:
    @staticmethod
    def normalize(matrix):
        return matrix / np.linalg.norm(matrix, ord=2)
