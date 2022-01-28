import ase
import os
from ase.io import read
import numpy as np
import csv
from ase.io.jsonio import read_json
import json
from scipy.stats import rankdata
from ase.visualize import view
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import torch_geometric.transforms as T
from torch_geometric.utils import degree

import glob, os

import networkx as nx

import trimesh
import pickle

mofs = ase.io.read('qmof-geometries.xyz',index=':')
refcodes = np.genfromtxt('qmof-refcodes.csv',delimiter=',',dtype=str)
properties = np.genfromtxt('qmof-bandgaps.csv',delimiter=',',dtype=str)

def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    if reverse == False:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    elif reverse == True:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if adj == False:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(
                temp,
                pad_width=(0, neighbors + 1 - len(temp)),
                mode="constant",
                constant_values=0,
            )
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed, adj_list, adj_attr

os.chdir("./MOF_data")
mof_uc = []
i = 1
for file in glob.glob("*.json"):
    structure = ase.io.read(os.path.join("./MOF_data", file))
    distance_matrix = structure.get_all_distances(mic=True)
    num_of_nodes = distance_matrix.shape[0]
    if num_of_nodes <= 20:
        mof_uc.append(structure)
    
    if i % 1000 == 0:
        print("Processed", i, "files")
        print("================================================")
    
    i = i + 1

unit_cell = []
mof_graph = []
for i in range(len(mof_uc)):
    print("Processing ",i,"-th graph", sep="")
    s1 = mof_uc[i]
    distance_matrix = s1.get_all_distances(mic=True)
    distance_matrix_trimmed = threshold_sort(distance_matrix,8,12,adj=False)
    distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
    distance_matrix_trimmed[distance_matrix_trimmed != 0] = 1
    # graph_tmp = nx.convert_matrix.from_numpy_matrix(distance_matrix_trimmed.numpy())
    mof_graph.extend([nx.convert_matrix.from_numpy_matrix(distance_matrix_trimmed.numpy())])
    unit_cell.extend([i])


nu = [nx.number_of_nodes(g) for g in mof_graph]

sum(nu)/len(nu)

unit_cell = []
mof_graph = []
for i in range(len(mof_uc)):
    print("Processing ",i,"-th graph", sep="")
    s1 = mof_uc[i]
    for j in range(3):
        for k in range(3):
            for l in range(3):
                distance_matrix = s1.repeat((j + 1, k + 1, l + 1)).get_all_distances(mic=True)
                distance_matrix_trimmed = threshold_sort(distance_matrix,8,12,adj=False)
                distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
                distance_matrix_trimmed[distance_matrix_trimmed != 0] = 1
                # graph_tmp = nx.convert_matrix.from_numpy_matrix(distance_matrix_trimmed.numpy())
                mof_graph.extend([nx.convert_matrix.from_numpy_matrix(distance_matrix_trimmed.numpy())] * 10)
                unit_cell.extend([i] * 10)



with open('MOFGraphs.p', 'wb') as f:
    pickle.dump(mof_graph, f) 

with open('MOFUnitCells.p', 'wb') as f:
    pickle.dump(unit_cell, f) 



