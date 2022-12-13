import numpy as np

PROTEIN_MAPPING = {
    'raf': 0,
    'mek': 1,
    'plc': 2,
    'pip2': 3,
    'pip3': 4,
    'erk': 5,
    'akt': 6,
    'pka': 7,
    'pkc': 8,
    'p38': 9,
    'jnk': 10
}

def generate_edges(graph, mapping):
    edges = []

    # for each node in graph
    for node in graph:

        # for each neighbour node of a single node
        for neighbour in graph[node]:
            # if edge exists then append
            edges.append((mapping[node],mapping[neighbour]))
    return edges

def get_sachs_model_adj_matrix():
    matrix = [[0 for _ in range(11)] for _ in range(11)]

    connections = {
        'raf': ['mek'],
        'mek': ['erk'],
        'plc': ['pip2', 'pip3'],
        'pip2': [],
        'pip3': ['pip2'],
        'erk': ['akt'],
        'akt': [],
        'pka': ['raf', 'akt', 'erk', 'jnk', 'mek', 'p38'],
        'pkc': ['raf', 'jnk', 'mek', 'p38'],
        'p38': [],
        'jnk': []
    }
    
    # building sachs model matrix
    for protein_1, protein_list in connections.items():
        for protein_2 in protein_list:
            index_1, index_2 = PROTEIN_MAPPING[protein_1], PROTEIN_MAPPING[protein_2]

            matrix[index_1][index_2] = 1

    return np.array(matrix) 

def get_supplemented_ground_truth_adj_matrix():
    matrix = np.zeros((11,11))

    connections = {
        'raf': ['mek','akt'],
        'mek': ['erk','jnk','plc'],
        'plc': ['pip2','mek','pkc', 'pka', 'akt'],
        'pip2': ['pkc'],
        'pip3': ['pip2', 'akt','plc'],
        'erk': ['akt'],
        'akt': ['mek','pkc','raf','plc'],
        'pka': ['raf', 'akt', 'erk', 'jnk', 'mek', 'p38','plc','pkc'],
        'pkc': ['raf', 'jnk', 'mek', 'p38','akt','pka'],
        'p38': ['jnk'],
        'jnk': ['p38','mek']
    }
    
    #get the nodes labels
    nodes = connections.keys()


    # building ground truth matrix
    for protein_1, protein_list in connections.items():
        for protein_2 in protein_list:
            index_1, index_2 = PROTEIN_MAPPING[protein_1], PROTEIN_MAPPING[protein_2]

            matrix[index_1][index_2] = 1

    return matrix, generate_edges(connections, PROTEIN_MAPPING),  nodes