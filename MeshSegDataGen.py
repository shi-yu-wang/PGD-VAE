import trimesh
import networkx as nx
import numpy as np
import glob
import pickle

meshdata = []
i_data = 0

for name in glob.glob("./data/off/*.off"):
    i_data = i_data + 1
    mesh = trimesh.load(name)
    mg = trimesh.graph.vertex_adjacency_graph(mesh)
    mesh_adj = np.array(nx.adjacency_matrix(mg).todense())
    
    id_delete = []
    id_keep = []
    for i in range(mesh_adj.shape[0] // 2):
        if mesh_adj[i*2, i*2 + 1] == 0:
            if len(np.where(mesh_adj[i*2, (i*2 + 1):] == 1)[0]) == 0:
                id_delete.extend([i*2, i*2 + 1])
            else: 
                id_col = np.where(mesh_adj[i*2, (i*2 + 1):] == 1)[0][0]
                id_keep.extend([i*2, i*2 + 1])
                mesh_adj[:, [i*2 + 1, id_col + i*2 + 1]] = mesh_adj[:, [id_col + i*2 + 1, i*2 + 1]]
                mesh_adj[[i*2 + 1, id_col + i*2 + 1], :] = mesh_adj[[id_col + i*2 + 1, i*2 + 1], :]    
                
    mesh_adj = np.delete(mesh_adj, np.array(id_delete), 0)
    mesh_adj = np.delete(mesh_adj, np.array(id_delete), 1)
    
    meshdata.append(nx.convert_matrix.from_numpy_matrix(mesh_adj))
    print("Processed", i_data, "graphs")
    

unitcell = [0] * len(meshdata)

for i in range(len(meshdata)):
    print(nx.number_of_nodes(meshdata[i]))


with open('MeshSegGraphs.p', 'wb') as f:
    pickle.dump(meshdata, f) 

with open('MeshSegUnitCells.p', 'wb') as f:
    pickle.dump(unitcell, f) 
