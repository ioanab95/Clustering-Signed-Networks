from scipy.spatial.distance import euclidean
from sklearn.neighbors import kneighbors_graph


def distance(x, y):
    return float(euclidean(x, y))

def inverse_distance(x, y):
    return float(1/(euclidean(x, y) + 0.001))

def signed_network(dataset, k_pos, k_neg):

    nearest_neighbours = kneighbors_graph(dataset, n_neighbors=k_pos, metric=distance, mode='connectivity')
    positive_weight_matrix = nearest_neighbours.toarray()

    for i in range(positive_weight_matrix.shape[0]):
        for j in range(positive_weight_matrix.shape[0]):
            if positive_weight_matrix[i][j] == 1:
                positive_weight_matrix[j][i] = 1

    furthest_neighbours = kneighbors_graph(dataset, n_neighbors=k_neg, metric=inverse_distance, mode='connectivity')
    negative_weight_matrics = furthest_neighbours.toarray()

    for i in range(negative_weight_matrics.shape[0]):
        for j in range(negative_weight_matrics.shape[0]):
            if negative_weight_matrics[i][j] == 1:
                negative_weight_matrics[j][i] = 1

    return positive_weight_matrix, negative_weight_matrics
