from __future__ import (division, print_function)
# import time

import networkx as nx
import pickle

import pandas as pd
import random
# import matplotlib.pyplot as plt
# from scipy import stats
# from scipy.spatial import distance

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

from arg_helper import *
from model import *
# from args import *
from data import *
from data_parallel import *
from evaluation import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="synthetic")
parser.add_argument('--train', type=str, default="True")
parser.add_argument('--eval', type=str, default="False")
# parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)

args = parser.parse_args()
print(f'Agrs: {args}')
file_name = args.data + "_train.pt"
# Load data and configurations
if args.data == "qmof":
    graphs_whole = pd.read_pickle("MOFGraphs.p")
    unit_cell_whole = pd.read_pickle("MOFUnitCells.p")
    max_num_nodes = 360
    batch_size = 100
    batch_share = 20
    max_num_nodes_l = 30
    max_num_nodes_g = 12

elif args.data == "mesh":
    graphs_whole = pd.read_pickle("MeshSegGraphs.p")
    unit_cell_whole = pd.read_pickle("MeshSegUnitCells.p")
    max_num_nodes = 360
    batch_size = 100
    batch_share = 20
    max_num_nodes_l = 3
    max_num_nodes_g = 120
    
graphs_whole = pd.read_pickle("SynGraphs.p")
unit_cell_whole = pd.read_pickle("SynUnitCells.p")
max_num_nodes = 360
batch_size = 100
batch_share = 20
max_num_nodes_l = 3
max_num_nodes_g = 120

epochs = args.epoch
config = get_config("gran_grid.yaml")

# Make train and test data
graph_loader = Make_batch_data(num_pad = max_num_nodes, batch_size = batch_size, batch_share = batch_share)
graph_train, graph_test = graph_loader.makeTrain(dataset = graphs_whole, unit_cell = unit_cell_whole, num_per_unit_cell = 5000)

# Initialize model
if args.train == "True":
    seed = 666
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
        
    model = GRANMixtureBernoulli(config = config, max_num_nodes = max_num_nodes, max_num_nodes_l = max_num_nodes_l, max_num_nodes_g = max_num_nodes_g, num_cluster = 4, num_layer = 3, batch_size = batch_size, dim_l = 512, dim_g = 512)
    model = DataParallel(model, device_ids=config.gpus).to(config.device)
    
    ################################ Training process #############################
    # Set up optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=0)
    
    # Adjust learning rate
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[10, 30, 50, 70, 100, 130, 160, 200, 250, 350, 500],
            gamma=0.1)
    
    # Save loss values
    ## Total loss
    total_loss_record = []
    ## Reconstruction loss
    adj_loss_record = []
    ## KL loss
    kl_loss_record = []
    ## Contrastive loss
    reg_loss_record = []
    
    # Training iteration
    for epoch in range(epochs):
        model.train()
        lr_scheduler.step()
        
        # z_l_mu_record = []
        # z_g_mu_record = []
        # A_pred_record = []
        for i in range(len(graph_train)):
            optimizer.zero_grad()
            total_loss, adj_loss, kl_loss, reg, A_tmp, zl, zg = model(*[(graph_train[i],)])
            
            total_loss_record.append(total_loss.detach().cpu().numpy())
            adj_loss_record.append(adj_loss.detach().cpu().numpy())
            kl_loss_record.append(kl_loss.detach().cpu().numpy())
            reg_loss_record.append(reg.detach().cpu().numpy())
    
            print("epoch: ", epoch, "iter: ", i, "total loss: ", total_loss, "adj loss: ", adj_loss, "kl loss: ", kl_loss, "reg loss: ", reg)
    
            total_loss.backward()
            optimizer.step()
    

        if (epoch + 1) % 50 == 0:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, file_name)

if args.eval == "True":
    print(file_name)
    model = GRANMixtureBernoulli(config = config, max_num_nodes = max_num_nodes, max_num_nodes_l = max_num_nodes_l, max_num_nodes_g = max_num_nodes_g, num_cluster = 4, num_layer = 3, batch_size = batch_size, dim_l = 512, dim_g = 512)
    model.load_state_dict(torch.load(file_name)['model_state_dict'])
    model.eval()
    
    graph_gen_test = []
    
    with torch.no_grad():
        for i in range(len(graph_test)):
            total_loss, adj_loss, kl_loss, reg, A_tmp, zl, zg = model(*[(graph_test[i],)])
            graph_gen_test.append(A_tmp)
            
    graph_test_true = []
    graph_test_pred = []
    print("Evaluating generated graphs:")
    for i in range(2):
        print("Processing ", i, "-th graph")
        for j in range(len(graph_gen_test[0])):
            for k in range(graph_gen_test[0][0].shape[0]):
                graph_test_true.append(nx.convert_matrix.from_numpy_matrix(graph_test[i][j][k, :, :].detach().cpu().numpy()))
                graph_test_pred.append(nx.convert_matrix.from_numpy_matrix(graph_gen_test[i][j][k, :, :]))
    print("Degree: ")
    print(compute_kld(metric_name = 'degree', generate_graph_list = graph_test_pred, real_graph_list = graph_test_true))
    
    print("Cluster: ")
    print(compute_kld(metric_name = 'cluster', generate_graph_list = graph_test_pred, real_graph_list = graph_test_true))

    print("Average clustering distance: ")
    print(compute_KLD_from_graph(metric = 'avg_clustering_dist', generated_graph_list = graph_test_pred, real_graph_list = graph_test_true))

    print("Density: ")
    print(compute_KLD_from_graph(metric = 'density', generated_graph_list = graph_test_pred, real_graph_list = graph_test_true))
    print("Evaluation completed!")