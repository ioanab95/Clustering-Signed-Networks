import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import ParameterGrid

from real_word_datasets import signed_network
from spectral_clustering import SignedNetworkSpectralClustering


def grid_search_parameters(data, targets, num_clusters, laplacian_operators):
    k_pos = np.array([3, 5, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    k_neg = np.array([3, 5, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])

    parameters_grid = [{'k_pos': k_pos, 'k_neg': k_neg}]

    best_clustering_score = dict()
    for operator in laplacian_operators:
        best_clustering_score[operator] = 0

    for pair in ParameterGrid(parameters_grid):
        pos_weights, neg_weights = signed_network(data, pair['k_pos'], pair['k_neg'])
        max = 0
        operator = "Lam"
        for laplacian_operator in laplacian_operators:
            clustering = SignedNetworkSpectralClustering(pos_weights, neg_weights, num_clusters)
            cluster_labels = clustering.computer_clusters(laplacian_operator=laplacian_operator)
            score = metrics.normalized_mutual_info_score(cluster_labels, targets)
            if score >=max:
                max = score
                operator = laplacian_operator
        best_clustering_score[operator] += 1

    print best_clustering_score


def plot_clustering_scores(laplacian_scores, x_datapoints):
    fig = plt.figure(figsize=(12, 8.5), dpi=150)
    ax = plt.subplot(111)
    ax.set_color_cycle(['r', 'g', 'b', 'm', 'y', 'c', 'pink', 'orange', 'indigo', 'k'])
    plt.title("Olivetti faces, $k^{+} = 10$", fontsize=22)

    print laplacian_scores["L_gm"]
    plt.plot(x_datapoints, laplacian_scores["L_gm"], label='$L_{GM}$')
    plt.plot(x_datapoints, laplacian_scores["L_gm"], '*r')
    plt.plot(x_datapoints, laplacian_scores["L_am"], label='$L_{AM}$')
    plt.plot(x_datapoints, laplacian_scores["L_sn"], label='$L_{SN}$')
    plt.plot(x_datapoints, laplacian_scores["L_bn"], label='$L_{BN}$')
    plt.plot(x_datapoints, laplacian_scores["L_sym"], label='$L_{sym}^{+}$')
    plt.plot(x_datapoints, laplacian_scores["Q_sym"], label='$Q_{sym}^{-}$')

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.75, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.2, 0.7), shadow=False, ncol=1, prop={'size': 22})
    plt.xlabel('$k^{-}$', size=22)
    plt.ylabel('Clustering score', size=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fig.savefig("olivetti_1.pdf")


def vary_k_neg(data, target, num_clusters, laplacian_operators):

    operators = ['L_sym', 'Q_sym', 'L_sn', 'L_bn', 'L_am', 'L_gm']

    laplacian_scores = dict()
    for operator in operators:
        laplacian_scores[operator] = []

    k_neg_values = np.array([3, 5, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    for k in k_neg_values:
        pos_weights, neg_weights = signed_network(data, 10, k)
        for laplacian_operator in laplacian_operators:
            clustering = SignedNetworkSpectralClustering(pos_weights, neg_weights, num_clusters)
            cluster_labels = clustering.computer_clusters(laplacian_operator=laplacian_operator)
            score = metrics.normalized_mutual_info_score(cluster_labels, target)
            laplacian_scores[laplacian_operator] += [score]

    print laplacian_scores
    plot_clustering_scores(laplacian_scores, k_neg_values)


laplacian_operators = ['L_sn', 'L_bn', 'L_am', 'L_gm', 'L_sym', 'Q_sym']
iris_data, iris_target = datasets.load_iris(return_X_y=True)
num_clusters = 3
vary_k_neg(iris_data, iris_target, num_clusters, laplacian_operators)
grid_search_parameters(iris_data, iris_target, num_clusters, laplacian_operators)

faces = datasets.fetch_olivetti_faces()
faces_data, faces_target = faces.data, faces.target
num_clusters = 40
vary_k_neg(faces_data, faces_target, num_clusters, laplacian_operators)
