from spectral_clustering_algorithm import spectral_clustering, spectral_clustering_lbr
import numpy as np
from sklearn import metrics


import matplotlib.pyplot as plt
#import networkx as nx


def same_cluster(i, j, k):
    return i%k == j%k


def stochastic_block_method(k, cluster_size):

    p_in_pos = 0.08
    p_in_neg = 0.01
    p_out_pos = 0.075
    p_out_neg = 0.09

    n = k * cluster_size

    positive_weight_matrix = np.zeros((n, n))
    negative_weight_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if same_cluster(i, j, k):
                positive_weight_matrix[i][j] = np.random.choice([1, 0], p=[p_in_pos, 1 - p_in_pos])
                negative_weight_matrix[i][j] = np.random.choice([1, 0], p=[p_in_neg, 1 - p_in_neg])
            else:
                positive_weight_matrix[i][j] = np.random.choice([1, 0], p=[p_out_pos, 1 - p_out_pos])
                negative_weight_matrix[i][j] = np.random.choice([1, 0], p=[p_out_neg, 1 - p_out_neg])

            positive_weight_matrix[j][i] = positive_weight_matrix[i][j]
            negative_weight_matrix[j][i] = negative_weight_matrix[i][j]

    return positive_weight_matrix, negative_weight_matrix

pos_matrix, neg_matrix = stochastic_block_method(3, 100)

print pos_matrix
print neg_matrix

cluster_labels = spectral_clustering(pos_matrix, neg_matrix, 3)
print cluster_labels


correct_labels = [i%3 for i in range(300)]
print metrics.adjusted_rand_score(correct_labels, cluster_labels)


cluster_labels = spectral_clustering_lbr(pos_matrix, neg_matrix, 3)
print cluster_labels


correct_labels = [i%3 for i in range(300)]
print metrics.adjusted_rand_score(correct_labels, cluster_labels)



"""
file = open("dataset", "r")

pos_matrix = []
neg_matrix = []

for i in range(16):
    line = file.readline().rstrip().split(' ')
    line = [int(i) for i in line]
    pos_matrix.append(line)


for i in range(16):
    line = file.readline().rstrip().split(' ')
    line = [int(i) for i in line]
    neg_matrix.append(line)

pos_matrix = np.array(pos_matrix)
print pos_matrix
neg_matrix = np.array(neg_matrix)
print neg_matrix

cluster_labels = spectral_clustering_lbr(pos_matrix, neg_matrix, 2)

print cluster_labels
"""
