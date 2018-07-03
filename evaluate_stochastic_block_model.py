import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from spectral_clustering import SignedNetworkSpectralClustering
from stochastic_block_model import StochasticBlockModel


def plot_clustering_scores(laplacian_scores, x_datapoints, x_label):
    fig = plt.figure(figsize=(12, 8.5), dpi=150)
    ax = plt.subplot(111)
    ax.set_color_cycle(['r', 'g', 'b', 'm', 'y', 'c', 'pink', 'orange', 'indigo', 'k'])
    plt.title("Clustering scores with varying {}".format(x_label), fontsize=22)

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
    plt.xlabel(x_label, size=22)
    plt.ylabel('Median clustering score', size=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fig.savefig("num_clusters_1.pdf")


def vary_p_in_neg():
    "G+ is assortative and G- changes from being dissassortative to not having any clustering structure"
    p_in_pos = 0.3
    p_out_pos = 0.05
    p_out_neg = 0.2

    num_clusters = 4
    cluster_size = 50
    num_data_points = num_clusters * cluster_size

    operators = ['L_sym', 'Q_sym', 'L_sn', 'L_bn', 'L_am', 'L_gm']

    laplacian_scores = dict()
    for operator in operators:
        laplacian_scores[operator] = []

    for i in range(51):
        step = 0.02
        p_in_neg = (i) * step
        stochastic_block_model = StochasticBlockModel(
            p_in_pos, p_in_neg, p_out_pos, p_out_neg, num_clusters, cluster_size)
        correct_labels = [stochastic_block_model.cluster_assignmnet(i) for i in range(num_data_points)]
        clustering_score, _, _ = run_sthocastic_block_method(stochastic_block_model, operators, num_clusters,
                                                             correct_labels)
        print clustering_score
        for operator in operators:
            laplacian_scores[operator] += [clustering_score[operator]]

    x_datapoints = [i * 0.02 for i in range(51)]
    print laplacian_scores
    plot_clustering_scores(laplacian_scores, x_datapoints, x_label="$p_{in}^{-}$")


def vary_p_out_pos():
    "G+ is assortative and G- changes from being dissassortative to not having any clustering structure"
    p_in_pos = 0.4
    p_out_neg = 0.4
    p_in_neg = 0.05

    num_clusters = 4
    cluster_size = 50
    num_data_points = num_clusters * cluster_size

    operators = ['L_sym', 'Q_sym', 'L_sn', 'L_bn', 'L_am', 'L_gm']

    laplacian_scores = dict()
    for operator in operators:
        laplacian_scores[operator] = []

    for i in range(51):
        step = 0.02
        p_out_pos = (i) * step
        stochastic_block_model = StochasticBlockModel(
            p_in_pos, p_in_neg, p_out_pos, p_out_neg, num_clusters, cluster_size)
        correct_labels = [stochastic_block_model.cluster_assignmnet(i) for i in range(num_data_points)]
        clustering_score, _, _ = run_sthocastic_block_method(stochastic_block_model, operators, num_clusters,
                                                             correct_labels)
        print clustering_score
        for operator in operators:
            laplacian_scores[operator] += [clustering_score[operator]]

    x_datapoints = [i * 0.02 for i in range(51)]
    print laplacian_scores
    plot_clustering_scores(laplacian_scores, x_datapoints, x_label="$p_{out}^{+}$")


def vary_num_clusters():
    p_in_pos = 0.4
    p_in_neg = 0.1
    p_out_pos = 0.05
    p_out_neg = 0.2

    cluster_size = 30

    operators = ['L_sym', 'Q_sym', 'L_sn', 'L_bn', 'L_am', 'L_gm']

    laplacian_scores = dict()
    for operator in operators:
        laplacian_scores[operator] = []

    for num_clusters in range(2, 15):
        num_data_points = num_clusters * cluster_size
        stochastic_block_model = StochasticBlockModel(
            p_in_pos, p_in_neg, p_out_pos, p_out_neg, num_clusters, cluster_size)
        correct_labels = [stochastic_block_model.cluster_assignmnet(i) for i in range(num_data_points)]
        clustering_score, _, _ = run_sthocastic_block_method(stochastic_block_model, operators, num_clusters,
                                                             correct_labels)
        print clustering_score
        for operator in operators:
            laplacian_scores[operator] += [clustering_score[operator]]

    x_datapoints = [i for i in range(2, 15)]
    print laplacian_scores
    plot_clustering_scores(laplacian_scores, x_datapoints, x_label="$k$")


def run_sthocastic_block_method(stochastic_block_model, laplacian_operators, num_clusters, correct_labels):
    num_runs = 50

    clustering_score = dict()
    for operator in laplacian_operators:
        clustering_score[operator] = []

    exp_pos, exp_neg = stochastic_block_model.generate_signed_graph()
    for i in range(num_runs):
        pos_weights, neg_weights = stochastic_block_model.generate_signed_graph()
        exp_pos += pos_weights
        exp_neg += neg_weights
        spectral_clustering = SignedNetworkSpectralClustering(pos_weights, neg_weights, num_clusters)
        for operator in laplacian_operators:
            cluster_labels = spectral_clustering.computer_clusters(laplacian_operator=operator)
            clustering_score[operator] += [metrics.adjusted_mutual_info_score(cluster_labels, correct_labels)]

    for operator in laplacian_operators:
        clustering_score[operator] = np.median(clustering_score[operator])

    exp_pos = exp_pos / (num_runs + 1)
    exp_neg = exp_neg / (num_runs + 1)

    return clustering_score, exp_pos, exp_neg


def plot_expectation_matrices(exp_pos, exp_neg, exp_pos_1, exp_neg_1):
    fig = plt.figure(figsize=(12, 12), dpi=150)
    ax = fig.add_subplot(221)
    ax.matshow(exp_pos)
    plt.title('a) $\mathcal{W}^{+}$ with $p_{in}^{-}$ = 0.1', size=20, y=-0.1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax = fig.add_subplot(222)
    ax.matshow(exp_neg)
    plt.title('b) $\mathcal{W}^{-}$ with $p_{in}^{-}$ = 0.1', size=20, y=-0.1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax = fig.add_subplot(223)
    ax.matshow(exp_pos_1)
    plt.title('c) $\mathcal{W}^{+}$ with $p_{in}^{-}$ = 0.3', size=20, y=-0.1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax = fig.add_subplot(224)
    ax.matshow(exp_neg_1)
    plt.title('d) $\mathcal{W}^{-}$ with $p_{in}^{-}$ = 0.3', size=20, y=-0.1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    fig.savefig("dissassortative_matrix.pdf")


def plot_dissassortative_matrices():
    num_clusters = 4
    cluster_size = 50
    num_data_points = num_clusters * cluster_size

    p_in_pos = 0.3
    p_out_pos = 0.05
    p_out_neg = 0.2

    p_in_neg = 0.1
    stochastic_block_model = StochasticBlockModel(p_in_pos, p_in_neg, p_out_pos, p_out_neg, num_clusters, cluster_size)
    correct_labels = [stochastic_block_model.cluster_assignmnet(i) for i in range(num_data_points)]
    operators = ['L_sym', 'Q_sym', 'L_sn', 'L_bn', 'L_am', 'L_gm']
    clustering_score, exp_pos, exp_neg = run_sthocastic_block_method(stochastic_block_model, operators, num_clusters,
                                                                     correct_labels)

    p_in_neg = 0.3
    stochastic_block_model = StochasticBlockModel(p_in_pos, p_in_neg, p_out_pos, p_out_neg, num_clusters, cluster_size)
    correct_labels = [stochastic_block_model.cluster_assignmnet(i) for i in range(num_data_points)]
    operators = ['L_sym', 'Q_sym', 'L_sn', 'L_bn', 'L_am', 'L_gm']
    clustering_score, exp_pos_1, exp_neg_1 = run_sthocastic_block_method(stochastic_block_model, operators,
                                                                         num_clusters, correct_labels)

    plot_expectation_matrices(exp_pos, exp_neg, exp_pos_1, exp_neg_1)


def plot_assortative_matrices():
    num_clusters = 4
    cluster_size = 50
    num_data_points = num_clusters * cluster_size

    p_in_pos = 0.4
    p_out_neg = 0.4
    p_in_neg = 0.05

    p_out_pos = 0.3
    stochastic_block_model = StochasticBlockModel(p_in_pos, p_in_neg, p_out_pos, p_out_neg, num_clusters, cluster_size)
    correct_labels = [stochastic_block_model.cluster_assignmnet(i) for i in range(num_data_points)]
    operators = ['L_sym', 'Q_sym', 'L_sn', 'L_bn', 'L_am', 'L_gm']
    clustering_score, exp_pos, exp_neg = run_sthocastic_block_method(stochastic_block_model, operators, num_clusters,
                                                                     correct_labels)

    p_out_pos = 0.5
    stochastic_block_model = StochasticBlockModel(p_in_pos, p_in_neg, p_out_pos, p_out_neg, num_clusters, cluster_size)
    correct_labels = [stochastic_block_model.cluster_assignmnet(i) for i in range(num_data_points)]
    operators = ['L_sym', 'Q_sym', 'L_sn', 'L_bn', 'L_am', 'L_gm']
    clustering_score, exp_pos_1, exp_neg_1 = run_sthocastic_block_method(stochastic_block_model, operators,
                                                                         num_clusters, correct_labels)

    plot_expectation_matrices(exp_pos, exp_neg, exp_pos_1, exp_neg_1)


vary_p_in_neg()
vary_p_out_pos()
vary_num_clusters()
plot_assortative_matrices()
plot_dissassortative_matrices()
