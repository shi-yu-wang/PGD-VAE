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

# Load data and configurations
graphs_whole = pd.read_pickle("SynGraphs.p")
unit_cell_whole = pd.read_pickle("SynUnitCells.p")
config = get_config("gran_grid.yaml")

# Set up model settings
max_num_nodes = 360
batch_size = 100
batch_share = 20
max_num_nodes_l = 3
max_num_nodes_g = 120
epochs = 300

# Make train and test data
graph_loader = Make_batch_data(num_pad = max_num_nodes, batch_size = batch_size, batch_share = batch_share)
graph_train, graph_test = graph_loader.makeTrain(dataset = graphs_whole, unit_cell = unit_cell_whole, num_per_unit_cell = 5000)


# Initialize model
model = GRANMixtureBernoulli(config = config, max_num_nodes = max_num_nodes, max_num_nodes_l = max_num_nodes_l, max_num_nodes_g = max_num_nodes_g, num_cluster = 4, num_layer = 3, batch_size = batch_size, dim_l = 512, dim_g = 512)
model = DataParallel(model, device_ids=config.gpus).to(config.device)

################################ Training process #############################
# Set up optimizer
params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(params, lr=0.001, weight_decay=0)

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

        # A_pred_record.append(A_tmp)
        # z_l_mu_record.append(zl)
        # z_g_mu_record.append(zg)
    if (epoch + 1) % 50 == 0:
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, "model_syn.pt")

