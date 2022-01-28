import torch
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

import networkx as nx
import pickle as pkl
import scipy.sparse as sp
import logging

import numpy as np
import random
from random import shuffle

import shutil
import os
import time
from model import *
from utils import *

# Set seeds
random.seed(10)

def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    return output



def encode_adj(adj, max_prev_node=10, is_full = False):
    '''

    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0]-1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i,:] = adj_output[i,:][::-1] # reverse order

    return adj_output

def encode_adj_flexible(adj):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end-len(adj_slice)+np.amin(non_zero)

    return adj_output


########## use pytorch dataloader
class Graph_sequence_sampler_pytorch(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            print('calculating max previous node, total iteration: {}'.format(iteration))
            self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            print('max previous node: {}'.format(self.max_prev_node))
        else:
            self.max_prev_node = max_prev_node

        # self.max_prev_node = max_prev_node

        # # sort Graph in descending order
        # len_batch_order = np.argsort(np.array(self.len_all))[::-1]
        # self.len_all = [self.len_all[i] for i in len_batch_order]
        # self.adj_all = [self.adj_all[i] for i in len_batch_order]
    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        start_idx = np.random.randint(adj_copy.shape[0])
        x_idx = np.array(bfs_seq(G, start_idx))
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        return {'x':x_batch, 'y':y_batch, 'len':len_batch, 'max_prev_node': self.max_prev_node}

    def calc_max_prev_node(self, iter=20000,topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print('iter {} times'.format(i))
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-1*topk:]
        return max_prev_node

class Make_batch_data:
    def __init__(self, num_pad, batch_size, batch_share):
        """
        num_pad: padding size
        """
        self.batch_size = batch_size
        self.batch_share = batch_share
        self.num_pad = num_pad
    
    def makeAdj(self, graphs, num_pad):
        adj = torch.zeros(len(graphs), num_pad, num_pad)
        for i in range(len(graphs)):
            graph_tmp = torch.from_numpy(nx.to_numpy_matrix(graphs[i]))
            adj[i, :, :] = F.pad(graph_tmp, (0, num_pad - graph_tmp.shape[1], 0, num_pad - graph_tmp.shape[0]), "constant", 0)
        
        return adj
    
    def makeTrain(self, dataset, unit_cell, num_per_unit_cell):
        '''
        dataset: list of networkx graphs
        unit_cell: Numpy array of unit cells identifier
        '''
        # Number of graphs
        num_graph = len(dataset)
        
        # Number of unit cells
        num_unit_cell = len(set(unit_cell))
        
        # # Detect if the training set is too large
        # if num_train_per_unit_cell * num_unit_cell >= num_graph:
        #     print("Training set is larger than the input data!")
        #     return
        
        # Extract training and testing data
        data_train = []
        data_test = []
        unit_cell_train = []
        unit_cell_test = []
        
        uc = list(set(unit_cell))
        
        for i in range(num_unit_cell):

            dataset_tmp = [dataset[k] for k in np.where(unit_cell == uc[i])[0]]
            random.shuffle(dataset_tmp)

            data_train_tmp = dataset_tmp[:num_per_unit_cell]
            data_train.append(data_train_tmp)
            unit_cell_tmp_train = uc[i] * num_per_unit_cell
            unit_cell_train.append(unit_cell_tmp_train)
            
            data_test_tmp = dataset_tmp[num_per_unit_cell:num_per_unit_cell*2]
            data_test.append(data_test_tmp)
            unit_cell_tmp_test = uc[i] * num_per_unit_cell
            unit_cell_test.append(unit_cell_tmp_test)
            
        # Extract shared data
        adj_share = []
        for i in range(num_unit_cell):
            adj_share_tmp = random.sample(data_train[i], self.batch_share)
            adj_share.append(self.makeAdj(adj_share_tmp, self.num_pad))
        
        # Make batches for training data
        batch_remain = self.batch_size - self.batch_share
        adj_train = []
        unit_cell_identifier_train = []
        for i in range(num_unit_cell):
            data_train_unit_cell_tmp = data_train[i]
            for j in range(num_per_unit_cell // self.batch_size):
                adj_train_unit_cell_tmp = torch.cat((adj_share[i], self.makeAdj(data_train_unit_cell_tmp[j * batch_remain:(j+1) * batch_remain], self.num_pad)), dim = 0)
                adj_train.append(adj_train_unit_cell_tmp)
                unit_cell_identifier_train.append(i)
        
        # Make batches for testing data
        adj_test = []
        unit_cell_identifier_test = []
        for i in range(num_unit_cell):
            data_test_unit_cell_tmp = data_test[i]
            for j in range(num_per_unit_cell // self.batch_size):
                adj_test_unit_cell_tmp = self.makeAdj(data_test_unit_cell_tmp[j * self.batch_size:(j+1) * self.batch_size], self.num_pad)
                adj_test.append(adj_test_unit_cell_tmp)
                unit_cell_identifier_test.append(i)
        
        # Finalize batched training and testing data
        num_graphs_each_unit_cell = int(len(unit_cell_identifier_train) / num_unit_cell)
        adj_train_output = []
        adj_test_output = []
        for i in range(num_graphs_each_unit_cell):
            adj_unit_cell_train = []
            adj_unit_cell_test = []
            for j in range(num_unit_cell):
                adj_unit_cell_train.append(adj_train[j * num_graphs_each_unit_cell + i])
                adj_unit_cell_test.append(adj_train[j * num_graphs_each_unit_cell + i])
            adj_train_output.append(adj_unit_cell_train)
            adj_test_output.append(adj_unit_cell_test)
            
        # Finalize batched test data
        # num_batch_test = int(len(unit_cell_identifier_train) / num_unit_cell)
        # adj_test_output = []
        # for i in range(num_batch_test):
        #     adj_test_tmp = []
        #     for j in range(num_unit_cell):
        #         adj_test_tmp.append(self.makeAdj(graphs = data_test[j * num_batch_test + i], num_pad = self.num_pad))
        #     adj_test_output.append(adj_test_tmp)
        
        return adj_train_output, adj_test_output

        