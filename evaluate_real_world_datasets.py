
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt
from spectral_clustering import SignedNetworkSpectralClustering
from sklearn.model_selection import ParameterGrid
from real_word_datasets import signed_network
import numpy as np


iris_data, iris_target = datasets.load_iris(return_X_y=True)
num_clusters = 3


k_pos = np.arange(3, 100, step=1)
k_neg = np.arange(3, 100, step=1)

parameters_grid = [{'k_pos':k_pos, 'k_neg':k_neg}]
max = 0
for pair in ParameterGrid(parameters_grid):
    pos_weights, neg_weights = signed_network(iris_data, pair['k_pos'], pair['k_neg'])
    clustering = SignedNetworkSpectralClustering(pos_weights, neg_weights, num_clusters)
    cluster_labels = clustering.computer_clusters(laplacian_operator="L_gm")
    score = metrics.normalized_mutual_info_score(cluster_labels, iris_target)
    if score > max:
        max=score
        best_k_pos =






cluster_labels=clustering.computer_clusters(laplacian_operator="L_sn")
print cluster_labels
print metrics.normalized_mutual_info_score(cluster_labels, iris_target)

cluster_labels=clustering.computer_clusters(laplacian_operator="L_bn")
print cluster_labels
print metrics.normalized_mutual_info_score(cluster_labels, iris_target)


cluster_labels=clustering.computer_clusters(laplacian_operator="L_am")
print cluster_labels
print metrics.normalized_mutual_info_score(cluster_labels, iris_target)
