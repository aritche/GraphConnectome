from glob import glob
import numpy as np

def get_graph_structure(adj_matrix, threshold=None):
    if adj_matrix.ndim != 3 or adj_matrix.shape[0] != adj_matrix.shape[1] or adj_matrix.shape[2] != 1:
        raise ValueError("Input matrix must be of shape N x N x 1")

    adj_matrix = adj_matrix[:, :, 0]  # Convert the input matrix to 2D (N x N)

    # Only keep the top N entries
    if threshold != None:
        sorted_arr = np.sort(adj_matrix.flatten())
        threshold = sorted_arr[-threshold]
        adj_matrix[adj_matrix<threshold] = 0
    
    non_zero_indices = np.argwhere(adj_matrix != 0)  # Get the row and column indices of non-zero elements

    new_adj_matrix_size = len(non_zero_indices)
    new_adj_matrix = np.zeros((new_adj_matrix_size, new_adj_matrix_size))

    for i, (row_i, col_i) in enumerate(non_zero_indices):
        for j, (row_j, col_j) in enumerate(non_zero_indices):
            if i != j and (row_i == row_j or col_i == col_j):
                new_adj_matrix[i, j] = 1

    # NOTE: non_zero_indices is gauranteed to be in the order of the rows/cols in the adj_matrix
    #       hence it is useful for knowing the corresponding coordinates in the original connectivity matrix
    return new_adj_matrix, non_zero_indices

def get_features(raw_matrix, node_indices):
    node_features = []
    for node in node_indices:
        node_features.append(raw_matrix[node[0], node[1]])
    node_features = np.array(node_features)

    return node_features

def get_matrix(adj_matrix, raw_matrix, threshold=None):
    if adj_matrix.ndim != 3 or adj_matrix.shape[0] != adj_matrix.shape[1] or adj_matrix.shape[2] != 1:
        raise ValueError("Input matrix must be of shape N x N x 1")

    adj_matrix = adj_matrix[:, :, 0]  # Convert the input matrix to 2D (N x N)

    # Only keep the top N entries
    if threshold != None:
        sorted_arr = np.sort(adj_matrix.flatten())
        threshold = sorted_arr[-threshold]
        adj_matrix[adj_matrix<threshold] = 0
    
    non_zero_indices = np.argwhere(adj_matrix != 0)  # Get the row and column indices of non-zero elements

    new_adj_matrix_size = len(non_zero_indices)
    new_adj_matrix = np.zeros((new_adj_matrix_size, new_adj_matrix_size))

    for i, (row_i, col_i) in enumerate(non_zero_indices):
        for j, (row_j, col_j) in enumerate(non_zero_indices):
            if i != j and (row_i == row_j or col_i == col_j):
                new_adj_matrix[i, j] = 1

    node_features = []
    for node in non_zero_indices:
        node_features.append(raw_matrix[node[0], node[1]])
    node_features = np.array(node_features)

    return new_adj_matrix, node_features

def get_edges(adj_matrix):
    # NOTE: Self-edges are not included in the list, i.e. the ID for a given node will not be in
    #       it's own list. This may cause an error with Yuqian's code.

    num_nodes = adj_matrix.shape[0]

    # Calculate the maximum number of connections (for cropping unecessary padding zeros)
    connection_list = []
    for i in range(num_nodes):
        connected_nodes = np.nonzero(adj_matrix[i])[0]
        connected_nodes.sort()
        connection_list.append(connected_nodes)
    max_connections = max([len(conn) for conn in connection_list])
    
    # Create the edge matrix
    connections = np.zeros((num_nodes, max_connections), dtype=int)
    for i, connected_nodes in enumerate(connection_list):
        connections[i, :len(connected_nodes)] = connected_nodes
    
    return connections

# Compute the mean connectivity matrix
print('Computing mean connectivity matrix...')
mean_adj = np.float64(np.zeros((84,84)))
num_subjects = 0
for fn in glob('84x84_dataset_with_adj_matrices/adj_matrices/*.npy'):
    mean_adj += np.load(fn)[:,:,0]
    num_subjects += 1
mean_adj /= num_subjects
mean_adj = np.expand_dims(mean_adj,-1)

# Compute the graph structure for the mean connectivity matrix
graph_structure, node_indices = get_graph_structure(mean_adj)
np.save('mean_structure.npy', graph_structure)
e = get_edges(graph_structure)
np.save('edge_define.npy', e)

# Compute the features for each subject
all_features = []
i = 0
for fn in glob('84x84_dataset_with_adj_matrices/adj_matrices/*.npy'):
    adj = np.load(fn) # 84 x 84 x 2
    f = get_features(adj, node_indices)
    all_features.append(f) # store the features
    i += 1
    print(i)

all_features = np.array(all_features)
print(all_features.shape)
np.save('feature_array.npy', all_features)
