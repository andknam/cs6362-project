#Marco Barbero Mota
#November 2022


# %%
#Generate plot for the Ground truth DAG
import networkx as nx
from utils import *
import matplotlib.pyplot as plt
#%%
#Obtain the adjacency matrix
ground_truth_matrix,edges, nodes_names = get_supplemented_ground_truth_adj_matrix()
#%%
#Plotting
G = nx.DiGraph() 
G.add_edges_from(edges)
pos = nx.circular_layout(G)
pos_labels = {val:l for l,val in zip(nodes_names,np.arange(11))}
nx.draw_networkx_nodes(G, pos = pos,node_color='blue', node_size=1200, alpha = 0.5)
nx.draw_networkx_edges(G, pos = pos, width = 2, edge_color='black', arrows = True, arrowsize=20, 
    alpha = 0.75, node_size=1200)
nx.draw_networkx_labels(G, pos=pos, labels=pos_labels, font_color='white', font_family='courier new', 
    font_weight='heavy')
plt.savefig('Ground truth.pdf', dpi = 150)
