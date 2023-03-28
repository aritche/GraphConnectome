from glob import glob
import numpy as np

i = 0
for fn in glob('../Fan_Files/results/*.npy'):
    x = np.load(fn)

    rows = []
    cols = []
    edge_features = []
    node_features = []

    for row in range(len(x)):
        node_features.append([0])
        for col in range(len(x[row])):
            if x[row][col][0] != 0 or x[row][col][1] != 0:
                rows.append(row)
                cols.append(col)
                edge_features.append(x[row][col])


    edge_index = np.array([rows,cols])
    edge_features = np.array(edge_features)
    node_features = np.array(node_features)

    subject_id = fn.split('/')[-1].split('.')[0]
    np.save('./node_features/' + subject_id + '.npy', node_features)
    #np.save('./edge_features/' + subject_id + '.npy', edge_features)
    #np.save('./edge_index/' + subject_id + '.npy', edge_index)

    print(i)
    i += 1
