import numpy as np

from scipy.linalg import sqrtm
from scipy.linalg import inv
from sklearn.cluster import KMeans


class SignedNetworkSpectralClustering:

    def __init__(self, positive_weight_matrix, negative_weight_matrix, num_clusters):
        self.positive_weight_matrix = positive_weight_matrix
        self.negative_weight_matrix = negative_weight_matrix
        self.num_clusters = num_clusters

    def computer_clusters(self, laplacian_operator):
        if laplacian_operator == 'L_gm':
            laplacian = self.__compute_L_gm()
        elif laplacian_operator == 'L_sym':
            laplacian = self.__compute_L_sym()
        elif laplacian_operator == 'Q_sym':
            laplacian = self.__compute_Q_sym()
        elif laplacian_operator == 'L_sn':
            laplacian = self.__compute_L_sn()
        elif laplacian_operator == 'L_bn':
            laplacian = self.__compute_L_bn()
        elif laplacian_operator == 'L_am':
            laplacian = self.__compute_L_am()
        else:
            raise ValueError("Unknown Laplacian operator {}".format(laplacian_operator))

        return self.__spectral_clustering(laplacian, self.num_clusters)

    def __compute_geometric_mean(self, matrix_a, matrix_b):
        return np.dot(matrix_a, sqrtm(np.dot(np.linalg.inv(matrix_a), matrix_b)))

    def __diagonal_shift(self, matrix, epsilon):
        I = np.diag(np.ones(matrix.shape[0]))
        return matrix + epsilon * I

    def __spectral_clustering(self, laplacian, k):
        eigvals, eigvec = np.linalg.eig(laplacian)
        indices = eigvals.argsort()[:k]
        eigvec = eigvec[:, indices]

        clusters = KMeans(n_clusters=k).fit(eigvec)
        return clusters.labels_

    def __compute_L_sym(self):
        """
        L = D - positive_weight_matrix
        L_sym = D^{-1/2} L D^{-1/2}
        """

        D = np.diag(np.dot(self.positive_weight_matrix, np.ones(self.positive_weight_matrix.shape[0])))
        D_sqrt_inv = np.linalg.inv(sqrtm(D))
        L = D - self.positive_weight_matrix
        L_sym = np.dot(D_sqrt_inv, np.dot(L, D_sqrt_inv))
        return L_sym

    def __compute_Q_sym(self):
        """
        Q = D + negative_weight_matrix
        Q_sym = D^{-1/2} Q D^{-1/2}
        """
        D = np.diag(np.dot(self.negative_weight_matrix, np.ones(self.negative_weight_matrix.shape[0])))
        D_sqrt_inv = inv(sqrtm(D))
        Q = D + self.negative_weight_matrix
        Q_sym = np.dot(D_sqrt_inv, np.dot(Q, D_sqrt_inv))
        return Q_sym


    def __compute_L_gm(self):
        """
        L_gm = L_sym # Q_sym
        """
        L_sym = self.__diagonal_shift(self.__compute_L_sym(), 0.1)
        Q_sym = self.__diagonal_shift(self.__compute_Q_sym(), 0.1)

        L_gm = self.__compute_geometric_mean(L_sym, Q_sym)

        return L_gm

    def __compute_L_bn(self):
        """
        L_br = D - positive_weight_matrix + negative_weight_matrix
        L_bn = D^{-1} L_br
        """
        D = np.diag(np.dot(self.positive_weight_matrix, np.ones(self.positive_weight_matrix.shape[0])))
        D_bar = np.diag(np.dot(self.positive_weight_matrix, np.ones(self.positive_weight_matrix.shape[0])) +
                        np.dot(self.negative_weight_matrix, np.ones(self.negative_weight_matrix.shape[0])))
        D_bar_inv = np.linalg.inv(D_bar)
        L_br = D - self.positive_weight_matrix + self.negative_weight_matrix
        L_bn = np.dot(D_bar_inv, L_br)

        return L_bn

    def __compute_L_sn(self):
        """
        L_sr = D_bar - positive_weight_matrix + negative_weight_matrix
        L_sn = D^{-1/2} L_sr D^{-1/2}
        """
        D_bar = np.diag(np.dot(self.positive_weight_matrix, np.ones(self.positive_weight_matrix.shape[0])) +
                        np.dot(self.negative_weight_matrix, np.ones(self.negative_weight_matrix.shape[0])))
        D_bar_sqrt_inv =  np.linalg.inv(sqrtm(D_bar))
        L_sr = D_bar - self.positive_weight_matrix + self.negative_weight_matrix
        L_sn = np.dot(D_bar_sqrt_inv, np.dot(L_sr, D_bar_sqrt_inv))

        return L_sn

    def __compute_L_am(self):
        """
        L_am = L_sym + Q_sym
        """
        L_sym = self.__diagonal_shift(self.__compute_L_sym(), 0.1)
        Q_sym = self.__diagonal_shift(self.__compute_Q_sym(), 0.1)
        L_am = L_sym + Q_sym

        return L_am





