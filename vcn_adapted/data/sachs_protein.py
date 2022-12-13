import numpy as np
import torch
from .generator import Generator
import networkx as nx
import graphical_models

class PT():
    """ Generate protein graph using hashmaps of protein mappings and edge connections 
    """

    def __init__(self, mapping, connections, num_nodes, protein_names):
        self.PROTEIN_MAPPING = mapping
        self.connections = connections
        self.num_nodes = num_nodes
        self.protein_names = protein_names
        self.build_graph()

    def build_graph(self):
        adj_matrix, edges_list = self.get_supplemented_ground_truth_adj_matrix()
        self.adjacency_matrix = adj_matrix

        self.graph = nx.Graph()
        self.graph.add_edges_from(edges_list)

    def get_supplemented_ground_truth_adj_matrix(self):
        matrix = np.zeros((self.num_nodes, self.num_nodes))
        edges_list = []

        for protein_1, protein_list in self.connections.items():
            for protein_2 in protein_list:
                # choosing only specific proteins if dimensionality < 11 nodes
                if protein_1 in self.protein_names and protein_2 in self.protein_names:
                    index_1, index_2 = self.PROTEIN_MAPPING[protein_1], self.PROTEIN_MAPPING[protein_2]

                    # building ground truth matrix
                    matrix[index_1][index_2] = 1

                    # edges for building graph
                    edges_list.append((index_1, index_2))

        return matrix, edges_list