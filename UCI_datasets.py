
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import euclidean, minkowski

from sklearn import datasets
from sklearn import metrics

import matplotlib.pyplot as plt

from spectral_clustering_algorithm import spectral_clustering_lgm, spectral_clustering_lbn, \
    spectral_clustering_lsn, spectral_clustering_lam

def distance(x, y):
    return float(euclidean(x, y))

def inverse_distance(x, y):
    return float(1/(euclidean(x, y) + 0.001))


def signed_network(dataset, k_pos, k_neg):

    nearest_neighbours = kneighbors_graph(dataset, n_neighbors=k_pos, metric=distance, mode='connectivity')
    positive_weight_matrix = nearest_neighbours.toarray()

    furthest_neighbours = kneighbors_graph(dataset, n_neighbors=k_neg, metric=inverse_distance, mode='connectivity')
    negative_weight_matrics = furthest_neighbours.toarray()

    return positive_weight_matrix, negative_weight_matrics



iris_data, iris_target = datasets.load_digits(return_X_y=True)
pos_weights, neg_weights = signed_network(iris_data, 30, 40)

print pos_weights
print neg_weights

plt.matshow(pos_weights)
plt.matshow(neg_weights)
plt.show()


cluster_labels=spectral_clustering_lgm(pos_weights, neg_weights, 10)
print cluster_labels
print metrics.adjusted_rand_score(cluster_labels, iris_target)

cluster_labels=spectral_clustering_lbn(pos_weights, neg_weights, 10)
print cluster_labels
print metrics.adjusted_rand_score(cluster_labels, iris_target)

cluster_labels=spectral_clustering_lsn(pos_weights, neg_weights, 10)
print cluster_labels
print metrics.adjusted_rand_score(cluster_labels, iris_target)


cluster_labels=spectral_clustering_lam(pos_weights, neg_weights, 10)
print cluster_labels
print metrics.adjusted_rand_score(cluster_labels, iris_target)
