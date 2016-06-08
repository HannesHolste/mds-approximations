import numpy as np
from scipy.linalg import eigh

from mdsa.algorithm import Algorithm


class Eigh(Algorithm):
    def __init__(self):
        super(Eigh, self).__init__(algorithm_name='eigh')

    def run(self, distance_matrix, num_dimensions_out=10):
        super(Eigh, self).run(distance_matrix, num_dimensions_out)
        distance_matrix = distance_matrix.data

        # Find eigenvalues and eigenvectors v of distance matrix
        eigenvalues, eigenvectors = eigh(distance_matrix)

        # these are already normalized such that
        # vi'vi = 1 where vi' is the transpose of eigenvector i

        percentages = ((eigenvalues / np.sum(eigenvalues)) * 100)[
                      :num_dimensions_out]

        # NOTE: numpy produces transpose of Numeric!
        return eigenvectors.transpose(), eigenvalues, percentages
