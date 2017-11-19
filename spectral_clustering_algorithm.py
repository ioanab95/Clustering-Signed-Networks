import numpy as np

from scipy.linalg import sqrtm
from scipy.linalg import inv
from sklearn.cluster import KMeans


def compute_L_sym(weight_matrix):
    D = np.diag(np.dot(weight_matrix, np.ones(weight_matrix.shape[0])))
    D_sqrt_inv = np.linalg.inv(sqrtm(D))

    laplacian = D - weight_matrix

    L_sym = np.dot(D_sqrt_inv, np.dot(laplacian, D_sqrt_inv))

    return L_sym


def compute_Q_sym(weight_matrix):

    D = np.diag(np.dot(weight_matrix, np.ones(weight_matrix.shape[0])))
    D_sqrt_inv = inv(sqrtm(D))

    signless_laplacian = D + weight_matrix

    symmetric_signless_laplacian = np.dot(D_sqrt_inv,
                                          np.dot(signless_laplacian, D_sqrt_inv))

    return symmetric_signless_laplacian


def compute_geometric_mean(matrix_a, matrix_b):
    return np.dot(matrix_a, sqrtm(np.dot(np.linalg.inv(matrix_a), matrix_b)))


def diagonal_shift(matrix, epsilon):
    I = np.diag(np.ones(matrix.shape[0]))
    return matrix + epsilon * I


def spectral_clustering(laplacian, k):
    eigvals, eigvec = np.linalg.eig(laplacian)
    indices = eigvals.argsort()[:k]
    eigvec = eigvec[:, indices]

    clusters = KMeans(n_clusters=k).fit(eigvec)
    return clusters.labels_


def spectral_clustering_lgm(positive_weight_matrix, negative_weight_matrix, k):
    L_sym = diagonal_shift(compute_L_sym(positive_weight_matrix), 0.7)
    Q_sym = diagonal_shift(compute_Q_sym(negative_weight_matrix), 0.7)

    L_gm = compute_geometric_mean(L_sym, Q_sym)

    return spectral_clustering(L_gm, k)


def spectral_clustering_lbn(positive_weight_matrix, negative_weight_matrix, k):
    D = np.diag(np.dot(positive_weight_matrix, np.ones(positive_weight_matrix.shape[0])))

    L_br = D - positive_weight_matrix + negative_weight_matrix

    D_bar = np.diag(np.dot(positive_weight_matrix, np.ones(positive_weight_matrix.shape[0])) +
                    np.dot(negative_weight_matrix, np.ones(negative_weight_matrix.shape[0])))

    D_bar_inv = np.linalg.inv(D_bar)

    L_bn = np.dot(D_bar_inv, L_br)

    return spectral_clustering(L_bn, k)


def spectral_clustering_lsn(positive_weight_matrix, negative_weight_matrix, k):
    D_bar = np.diag(np.dot(positive_weight_matrix, np.ones(positive_weight_matrix.shape[0])) +
                    np.dot(negative_weight_matrix, np.ones(negative_weight_matrix.shape[0])))

    L_sr = D_bar - positive_weight_matrix + negative_weight_matrix

    D_bar_sqrt_inv =  np.linalg.inv(sqrtm(D_bar))

    L_sn = np.dot(D_bar_sqrt_inv, np.dot(L_sr, D_bar_sqrt_inv))

    return spectral_clustering(L_sn, k)


def spectral_clustering_lam(positive_weight_matrix, negative_weight_matrix, k):
    L_sym = diagonal_shift(compute_L_sym(positive_weight_matrix), 0.01)
    Q_sym = diagonal_shift(compute_Q_sym(negative_weight_matrix), 0.01)

    L_am = L_sym + Q_sym

    return spectral_clustering(L_am, k)





