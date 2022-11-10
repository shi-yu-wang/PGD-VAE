# This is the official Pytorch implementation of [Deep Generative Model for Periodic Graphs](https://arxiv.org/pdf/2201.11932.pdf) accepted by NeurIPS 2022:
```
@article{wang2022deep,
  title={Deep Generative Model for Periodic Graphs},
  author={Wang, Shiyu and Guo, Xiaojie and Zhao, Liang},
  journal={arXiv preprint arXiv:2201.11932},
  year={2022}
}
```
The code for PGD-VAE is adapted from the code of GRAN: [Efficient graph generation with graph recurrent attention networks](https://github.com/lrjconan/GRAN).

## Running environment:
Python 3.9; PyTorch 1.8.1, networkx 2.5, scipy, numpy, pyyaml

## Datasets
Data: QMOF, MeshSeg and synthetic datasets have been processed and provided in the repository as: 

- QMOF: MOFGraphs.p, MOFUnitCells.p

- MeshSeg: MeshSeqGraphs.p, MeshSegUnitCells.p

- Synthetic: SynGraphs.p, SynUnitCells.p

To download the original datasets:

- Original QMOF data can be downloaded from [here](https://github.com/arosen93/QMOF)
- 
- Original MeshSeg data can be downloaded from [here](https://segeval.cs.princeton.edu/)

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
