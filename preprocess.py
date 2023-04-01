from glob import glob
import numpy as np

# Get a mapping of GM region name->IDs for all GM regions, but merge regions prefixed by WM and CTX
def get_region_names(ids_fn):
    with open(ids_fn) as f:
        data = [x.strip('\n') for x in f.readlines()]
    gm_mapping = {}
    for line in data:
        num, name = line.split(',')
        name = name.replace('ctx-', '')
        name = name.replace('wm-', '')

        name = name.lower()
        if name not in gm_mapping:
            gm_mapping[name] = [num]
        else:
            gm_mapping[name].append(num)

    return sorted(gm_mapping.keys())

def get_mrtrix_regions(fn):
    with open(fn) as f:
        regions = [x.strip('\n') for x in f.readlines()]
    regions = [x.lower().replace('ctx-', '').replace('wm-', '') for x in regions]
    return sorted(regions)

def remove_regions(x, regions, allowed_regions_set):
    keep = []
    for i in range(len(regions)):
        if regions[i] in allowed_regions_set:
            keep.append(i)

    x = np.take(x, keep, axis=0) # remove rows
    x = np.take(x, keep, axis=1) # remove cols

    return x 

regions = get_region_names('GM_ids.txt')
allowed_regions = get_mrtrix_regions('mrtrix_regions.txt')

regions_set = set(regions)
allowed_regions_set = set(allowed_regions)
remove_regions_set = regions_set - allowed_regions_set

i = 0
for fn in glob('./111x111_connectomes/*.npy'):
    # Only retain GM regions
    x = np.load(fn)
    x = remove_regions(x, regions, allowed_regions_set)

    # Save intermediate result
    out_fn = './84x84_connectomes/' + fn.split('/')[-1]
    np.save(out_fn, x)

    # Convert to edge indexes, edge features, node features
    rows = []
    cols = []
    edge_features = []
    node_features = []

    # temporarily remove self edges 
    no_self_edges = x.copy()
    for row in range(len(x)):
        no_self_edges[row][row][0] = 0
        no_self_edges[row][row][1] = 0

    for row in range(len(x)):
        #node_features.append([0]) # constant node features
        node_features.append(no_self_edges[row,:,0]) # SL profile as node features
        for col in range(len(x[row])):
            if row == col: # exclude self-edges
                continue
            # if at least 1 feature is non-zero, count it
            if x[row][col][0] != 0 or x[row][col][1] != 0:
                rows.append(row)
                cols.append(col)
                edge_features.append(x[row][col])

    edge_index = np.array([rows,cols])
    edge_features = np.array(edge_features)
    node_features = np.array(node_features)

    subject_id = fn.split('/')[-1].split('.')[0]

    np.save('./84x84_dataset/node_features/' + subject_id + '.npy', node_features)
    np.save('./84x84_dataset/edge_features/' + subject_id + '.npy', edge_features)
    np.save('./84x84_dataset/edge_index/' + subject_id + '.npy', edge_index)

    print(i)
    i += 1
