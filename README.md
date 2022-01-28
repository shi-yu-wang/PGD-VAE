# PGD-VAE
The source code of PGD-VAE
The code for PGD-VAE is adapted from the code of GRAN: https://github.com/lrjconan/GRAN 
(Liao, R., Li, Y., Song, Y., Wang, S., Nash, C., Hamilton, W. L., ... & Zemel, R. S. (2019). Efficient graph generation with graph recurrent attention networks. arXiv preprint arXiv:1910.00760.)

Running environment:
Python 3.9; PyTorch 1.8.1, networkx, 2.5

Data: QMOF, MeshSeg and synthetic datasets have been processed and provided in the package as: 
QMOF: MOFGraphs.p, MOFUnitCells.p
MeshSeg: MeshSeqGraphs.p, MeshSegUnitCells.p
Synthetic: SynGraphs.p, SynUnitCells.p

Original QMOF data can be downloaded from: https://github.com/arosen93/QMOF
Original MeshSeg data can be downloaded from: https://segeval.cs.princeton.edu/

The code to extract and generate data from original QMOF and MeshSeg datasets have been provided in the package as:
QMOF: MOFDataGen.py
MeshSeg: MeshSegDataGen.py

To train the model, run:
python train.py
or:
directly run code in train.py
This will train the model with the synthetic data and returns the trained model as model_syn.pt.
To train the model on QMOF or MeshSeg datasets, modify the input file in train.py and train the model in the same way

For evaluation purpose, please use functions in evaluation.py:
compute_kld(): degree, cluster, orbit
check_uniqueness(): uniqueness
novelty_metric(): novelty
