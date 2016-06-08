from numpy import dot
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, qr
from numpy.random import standard_normal

from mdsa.algorithm import Algorithm


class Ssvd(Algorithm):
    def __init__(self):
        super(Ssvd, self).__init__(algorithm_name='ssvd')

    def run(self, distance_matrix, num_dimensions_out=10):
        """
          takes a distance matrix and returns eigenvalues and eigenvectors
          of the ssvd method
          Based on algorithm described in 'Finding Structure with Randomness:
          Probabilistic Algorithms for Constructing Approximate Matrix
           Decompositions'
          by N. Halko, P.G. Martinsson, and J.A. Tropp
          Code adapted from R:
          https://goo.gl/gSPNZh

          distance_matrix: distance matrix
          num_dimensions_out: dimensions

          Constants:
          p: oversampling parameter - this is added to k to boost accuracy
          qiter: iterations to go through - to boost accuracy

          NOTE: the lower the num_dimensions_out (dimensions),
           the __worse__ the resulting matrix is
          """
        super(Ssvd, self).run(distance_matrix, num_dimensions_out)

        distance_matrix = distance_matrix.data

        # constants for algorithm
        p = 10
        qiter = 0

        m, n = distance_matrix.shape
        p = min(min(m, n) - num_dimensions_out,
                p)  # an mxn matrix M has at most p = min(m,n) unique
        # singular values
        r = num_dimensions_out + p  # rank plus oversampling parameter p

        omega = standard_normal(size=(n, r))  # generate random matrix omega
        # compute a sample matrix Y: apply distance_matrix to random
        y = dot(distance_matrix, omega)
        # vectors to identify part of its range corresponding
        # to largest singular values
        Q, R = qr(y)  # find an ON matrix st. Y = QQ'Y
        # multiply distance_matrix by Q whose columns form
        b = dot(Q.transpose(), distance_matrix)
        # an orthonormal basis for the range of Y

        # often, no iteration required to small error in eqn. 1.5
        for i in xrange(1, qiter):
            y = dot(distance_matrix, b.transpose())
            Q, R = qr(y)
            b = dot(Q.transpose(), distance_matrix)

        # compute eigenvalues of much smaller matrix bbt
        bbt = dot(b, b.transpose())
        if globals().get(eigsh, True):
            # eigsh faster, returns sorted eigenvals/vecs
            eigenvalues, eigenvectors = eigsh(bbt, num_dimensions_out)
        else:
            # eigsh faster, returns sorted eigenvals/vecs
            eigenvalues, eigenvectors = eigh(bbt, num_dimensions_out)
        U_ssvd = dot(Q, eigenvectors)  # [:,1:k]

        # don't need to compute V
        # V_ssvd = dot(transpose(b),dot(eigvecs, diag(1/eigvals,0))) [:, 1:k]
        eigenvectors = U_ssvd.real

        percentages = ((eigenvalues / np.sum(eigenvalues)) * 100)[
                      :num_dimensions_out]

        return eigenvectors.transpose(), eigenvalues, percentages
