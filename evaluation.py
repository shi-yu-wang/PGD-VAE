# import tensorflow as tf
# tf.disable_v2_behavior()
from sklearn.metrics import mean_squared_error

# flags = tf.compat.v1.flags
# flags = tf.app.flags
# FLAGS = flags.FLAGS
import sys
from tqdm import tqdm
import numpy as np
import sklearn
import networkx as nx
import scipy
import os
import pickle
from random import sample
import matplotlib.pyplot as plt

import networkx as nx
import math
import eval.stats
import scipy.io as scio
import scipy.stats
import scipy.sparse
from queue import Queue

from math import gcd
from functools import reduce

def make_discretizer(target, num_bins=2):
    """Wrapper that creates discretizers."""
    return np.digitize(target, np.histogram(target, num_bins)[1][:-1])


def MI(z, f):
    z = np.array(z)
    f = np.array(f)
    m = []
    for i in range(z.shape[-1]):
        discretized_z = make_discretizer(z[:, :, i].reshape(-1))
        m.append(sklearn.metrics.normalized_mutual_info_score(discretized_z, f))
    return np.max(m)


def MI_factor(f):
    res = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            res[i][j] = sklearn.metrics.normalized_mutual_info_score(f[i].reshape(-1), f[j].reshape(-1))
    print('factor mutual info matrix', res)
    return res

def compute_kld_(generated, real):
    generated = [i / generated.count(i) for i in generated]
    real = [i / real.count(i) for i in real]
    return scipy.stats.entropy(generated, real)

# Generated vs test
def compute_KLD_from_graph(metric, generated_graph_list, real_graph_list):
    generated_l = list()
    real_l = list()

    for i in tqdm(range(len(generated_graph_list))):
        generated_G = generated_graph_list[i]
        real_G = real_graph_list[i]
        if metric == 'avg_clustering_dist':
            try:
                temp_value = nx.average_clustering(generated_G)
                generated_l.append(temp_value)
            except:
                generated_l.append(1.0)
            real_l.append(nx.average_clustering(real_G))
        if metric == 'density':
            generated_l.append(nx.density(generated_G))
            real_l.append(nx.density(real_G))
        if metric == 'avg_node_connectivity':
            generated_l.append(nx.average_node_connectivity(generated_G))
            real_l.append(nx.average_node_connectivity(real_G))

            # transfer to descrete:
    generated_discrete = make_discretizer(generated_l, num_bins=50)
    real_discrete = make_discretizer(real_l, num_bins=50)
    kld = compute_kld_(list(generated_discrete), list(real_discrete))
    return kld

# Generated vs test
def compute_kld(metric_name, generate_graph_list, real_graph_list):
    # generate_graph_list = generate_graph_list[: test_size]
    # real_graph_list = real_graph_list[: test_size]
    generate_graph_list = generate_graph_list
    real_graph_list = real_graph_list

    fake = list()
    real = list()
    if metric_name == 'degree':
        return eval.stats.degree_stats(generate_graph_list,real_graph_list)
    elif metric_name == 'cluster':
        return eval.stats.clustering_stats(generate_graph_list,real_graph_list)
    elif metric_name == 'orbit':
        return eval.stats.orbit_stats_all(generate_graph_list,real_graph_list)
    # if metric_name == 'degree':
    #     for item in generate_graph_list:
    #         deg = eval.stats.degree_stats(item)
    #         fake.append(deg)
    #     for item in real_graph_list:
    #         deg = eval.stats.degree_stats(item)
    #         real.append(deg)
    # elif metric_name == 'cluster':
    #     for item in generate_graph_list:
    #         deg = eval.stats.clustering_stats(item)
    #         fake.append(deg)
    #     for item in real_graph_list:
    #         deg = eval.stats.clustering_stats(item)
    #         real.append(deg)
    # elif metric_name == 'orbit':
    #     for item in generate_graph_list:
    #         deg = eval.stats.orbit_stats_all(item)
    #         fake.append(deg)
    #     for item in real_graph_list:
    #         deg = eval.stats.orbit_stats_all(item)
    #         real.append(deg)

    # if metric_name == 'temporal':
    #     for item in generate_graph_list:
    #         tc, tc_vec = temporal_correlation(item)
    #         fake.append(tc)
    #     fake = np.nan_to_num(fake, nan=0)
    #     for item in real_graph_list:
    #         tc, tc_vec = temporal_correlation(item)
    #         real.append(tc)
    #     real = real[:len(fake)]
    #     real = np.nan_to_num(real, nan=0)
    # elif metric_name == 'between':
    #     for item in tqdm(generate_graph_list):
    #         graph_sum = 0
    #         for i in range(item.shape[0] - 1):
    #             for j in range(item.shape[1]):
    #                 temp_closeness = betweenness_centrality(item, i, j)
    #                 graph_sum += temp_closeness
    #         fake.append(graph_sum)

    #     for item in tqdm(real_graph_list):
    #         graph_sum = 0
    #         for i in range(item.shape[0] - 1):
    #             for j in range(item.shape[1]):
    #                 temp_closeness = betweenness_centrality(item, i, j)
    #                 graph_sum += temp_closeness
    #         real.append(graph_sum)
    # elif metric_name == 'close':
    #     for item in tqdm(generate_graph_list):
    #         graph_sum = 0
    #         for j in range(item.shape[1]):
    #             temp_closeness = closeness_centrality(item, 0, j)
    #             graph_sum += temp_closeness
    #
    #         fake.append(graph_sum)
    #
    #     for item in tqdm(real_graph_list[:len(fake)]):
    #         graph_sum = 0
    #         for j in range(item.shape[1]):
    #             temp_closeness = closeness_centrality(item, 0, j)
    #             graph_sum += temp_closeness
    #
    #         real.append(graph_sum)

    result = compute_KLD_from_graph(fake, real)
    # print(metric_name, result)

    return result

# start = time.time()
# cycles = list(nx.algorithms.cycles.simple_cycles(graphs_whole[0].to_directed()))
# print("Cycle time: ", time.time() - start)

# cycles_sizes = [len(c) for c in cycles]
# cycles_gcd = reduce(gcd, cycles_sizes)
# is_periodic = cycles_gcd > 1



# Generated
def check_uniqueness(graphs):
    all_graphs = set()
    for i in range(len(graphs)):
        all_graphs.add(tuple(np.reshape(graphs[i], (-1))))
    # graph = [tuple(np.reshape(graphs[i], (-1))) for i in range(len(graphs))]
    original_num = len(graphs)
    # all_smiles=set(graph)
    new_num = len(all_graphs)
    return new_num/original_num

# Generated vs training
def novelty_metric(graphs_all, graphs_pred):
    graph_all = set()
    for i in range(len(graphs_all)):
        graph_all.add(tuple(np.reshape(graphs_all[i], (-1))))
    # graph_all = set([tuple(graphs_all[i]) for i in range(len(graphs_all))])
    graph_pred = set()
    for j in range(len(graphs_pred)):
        graph_pred.add(tuple(np.reshape(graphs_pred[i], (-1))))
    # graph_pred = set([tuple(graphs_pred[i]) for i in range(len(graphs_pred))])
    total_new_graphs=0
    for g in graph_pred:
        if g not in graph_all:
            total_new_graphs+=1
            
    return float(total_new_graphs)/len(graph_pred)

