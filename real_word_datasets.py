from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import euclidean

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
