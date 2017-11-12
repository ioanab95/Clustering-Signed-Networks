import numpy as np

from scipy.linalg import sqrtm
from scipy.linalg import inv
from sklearn.cluster import KMeans


def compute_symmetric_laplacian(weight_matrix):
    D = np.diag(np.dot(weight_matrix, np.ones(weight_matrix.shape[0])))
    D_sqrt_inv = np.linalg.inv(sqrtm(D))

    laplacian = D - weight_matrix

    symmetric_laplacian = np.dot(D_sqrt_inv, np.dot(laplacian, D_sqrt_inv))

    return symmetric_laplacian


def compute_symmetric_signless_laplacian(weight_matrix):

    D = np.diag(np.dot(weight_matrix, np.ones(weight_matrix.shape[0])))
    print D
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


def spectral_clustering(positive_weight_matrix, negative_weight_matrix, k):

    symetric_laplacian = diagonal_shift(compute_symmetric_laplacian(positive_weight_matrix), 0.1)
    signless_symetric_laplacian = diagonal_shift(compute_symmetric_signless_laplacian(negative_weight_matrix), 0.2)
    L_gm = compute_geometric_mean(symetric_laplacian, signless_symetric_laplacian)

    eigvals, eigvec = np.linalg.eig(L_gm)
    indices = eigvals.argsort()[:k]
    eigvec = eigvec[:, indices]

    clusters = KMeans(n_clusters=k).fit(eigvec)
    return clusters.labels_



def spectral_clustering_lbr(positive_weight_matrix, negative_weight_matrix, k):

    D = np.diag(np.dot(positive_weight_matrix, np.ones(positive_weight_matrix.shape[0])))

    L_br = D - positive_weight_matrix + negative_weight_matrix

    eigvals, eigvec = np.linalg.eig(L_br)
    indices = eigvals.argsort()[:k]
    eigvec = eigvec[:, indices]

    clusters = KMeans(n_clusters=k).fit(eigvec)
    return clusters.labels_



"""
def compute_eigenspace(A, B):

    A_eigval, A_eigvec = np.linalg.eigh(A)
    B_eigval, B_eigvec = np.linalg.eigh(B)

    eigenvalues = []
    eigenspace = []

    for index_A, A_vec in enumerate(A_eigvec):
        for index_B, B_vec in enumerate(B_eigvec):
            if (np.allclose(A_vec, B_vec, 0.001)):
                eigenspace.append(A_vec)
                eigenvalues.append(np.sqrt(A_eigval[index_A] * B_eigval[index_B]))
                break

    #eigenspace.append(np.zeros(A.shape[0]))
    return eigenvalues, eigenspace


def project_subspace(u, eigenspace):
    projection = np.zeros_like(u)

    for v in eigenspace:
        projection += np.vdot(u, v) * v

    return projection


def IPM_algorithm(A, B, eigenspace):

    x_new = np.ones(A.shape[0])/np.sqrt(A.shape[0])
    for i in range(100):
        x = x_new
        u = np.linalg.solve(A, x)
        v = np.linalg.solve(sqrtm(np.dot(np.linalg.inv(A), B)), u)
        y = v - project_subspace(v, eigenspace)
        x_new = y/np.linalg.norm(y)

    return np.vdot(x, x_new), x_new




def spectral_clustering_improved(positive_weight_matrix, negative_weight_matrix, k):

    L_sym = diagonal_shift(compute_symmetric_laplacian(positive_weight_matrix), 0.1)
    Q_sym = diagonal_shift(compute_symmetric_signless_laplacian(negative_weight_matrix), 0.2)

    eigenvalues, eigenspace =  compute_eigenspace(L_sym, Q_sym)

    while(len(eigenvalues) < k):
        new_eigenvalue, new_eigenvector = IPM_algorithm(L_sym, Q_sym, eigenspace)




matrix_a = np.array([[0, 2, 3], [2, 0, 1], [3, 1, 0]])
matrix_b = np.array([[0, 2, 2], [2, 0, 1], [2, 1, 0]])

lsim = compute_symmetric_laplacian(matrix_a)
qsim = compute_symmetric_signless_laplacian(matrix_a)








x = np.ones(matrix_a.shape[0])/np.sqrt(matrix_a.shape[0])
print np.linalg.solve(matrix_a, x)

spectral_clustering(matrix_a, matrix_b, 2)
"""