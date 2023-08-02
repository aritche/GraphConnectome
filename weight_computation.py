import sys
import numpy as np
import os
from glob import glob
import vtk
import time
import csv
import json
import pickle

def streamline_count(filename):
    # Create a reader for .vtp files
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    # Get the output data from the reader
    polydata = reader.GetOutput()

    # Extract the number of streamlines
    num_streamlines = polydata.GetNumberOfLines()

    return num_streamlines

def point_count(filename):
    # Create a reader for .vtp files
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    # Get the output data from the reader
    polydata = reader.GetOutput()

    # Extract the number of streamlines
    num_points = polydata.GetNumberOfPoints()

    return num_points

def getEndpointsString(ids):
    if len(ids) == 1:
        endpoint_string_A = "endpoints_in(%s)" % (ids[0])
    elif len(ids) == 2:
        endpoint_string_A = "(endpoints_in(%s) or endpoints_in(%s))" % (ids[0], ids[1])
    else:
        print('ERROR: Invalid number of IDs for a given GM region (maximum should be 2).')
        sys.exit()

    return endpoint_string_A

# Get a mapping of GM region name->IDs for all GM regions, but merge regions prefixed by WM and CTX
def get_gm_mapping(ids_fn):
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

    return gm_mapping


# Build a dictionary for fn->label associations
def get_labels(labels_fn):
    labels = {}
    with open(labels_fn) as f:
        data = [x.strip('\n') for x in f.readlines()]
    for item in data[1:]:
        fn, label = item.split('\t')
        label = label.strip('\r')
        fn = fn.split('.')[0]
        labels[fn] = label

    return labels

# Build a dictionary for fn->wm tract associations
def get_tracts(tract_fn):
    tracts = {}
    with open(tract_fn) as f:
        data = [x.strip('\n') for x in f.readlines()]
    for item in data[1:]:
        fn, tract = item.split(',')
        tract = tract.strip('\r')
        tracts[fn] = tract
    return tracts

# Build a mapping
def get_subject_mapping(fn):
    with open(fn, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        data = [row for row in reader]

    header = data[0]
    data = data[1:]

    # Load column names
    props = set([])
    clusters = []
    col_to_vals = [[]]
    for item in header[1:]:
        location, cluster = item.split('.')[:2]
        cluster = 'tracts_' + location + '/' + cluster

        prop = '.'.join(item.split('.')[2:])

        props.add(prop)
        clusters.append(cluster)

        col_to_vals.append({'cluster': cluster, 'prop': prop})

    # Load subject values
    subjects = {}
    for row in data:
        subject_id = row[0]
        col_num = 1
        subjects[subject_id] = {}
        for val in row[1:]:
            cluster = col_to_vals[col_num]['cluster']
            prop = col_to_vals[col_num]['prop']

            if 'NAN' in val:
                val = 0
            if cluster not in subjects[subject_id]:
                subjects[subject_id][cluster] = {prop: float(val)}
            else:
                subjects[subject_id][cluster][prop] = float(val)
            
            col_num += 1

    return subjects

t0 = time.time()
gm_mapping = get_gm_mapping('GM_ids.txt')                    # region name  -> [ID1, ID2, ...]
labels = get_labels('cluster_hemisphere_location.txt')       # cluster name -> 'h' or 'c'
tracts = get_tracts('FiberClusterAnnotation_k0800_v1.0.csv') # cluster name -> tract or falsepositive or partial or unkown
subjects = get_subject_mapping('HCP_n1065_allDWI_fiber_clusters.csv') # subjects[subject_id][hemisphere/cluster_ID][property], e.g.  subjects['397760']['tracts_right_hemisphere/cluster_00778']['Num_Fibers']

# Aggregrate a list of filenames for clusters
clusters = tracts.keys()
cluster_fns = []
for cluster_id in clusters:
    # Exclude clusters for false positive or partial tracts
    if tracts[cluster_id] == 'FalsePositive' or tracts[cluster_id] == 'Partial':
        continue

    # Determine which directories to get tract files from
    if labels[cluster_id] == 'h':
        target_dirs = ['tracts_left_hemisphere', 'tracts_right_hemisphere']
    elif labels[cluster_id] == 'c':
        target_dirs = ['tracts_commissural']
    else:
        print('ERROR: Invalid tract label: %s' % (labels[cluster_id]))
        sys.exit()

    # Iterate over the individual directories
    for target_dir in target_dirs:
        cluster_fn = target_dir + '/' + cluster_id
        cluster_fns.append(cluster_fn)

# Aggregrate the regions that each cluster connects
# E.g. subsamples['tracts_right_hemisphere/cluster_00778'] 
#      = ('rhsuperiorparietal_rhrostralmiddlefrontal', 'rhlingual_rhlateralorbitofrontal', ...)
subsamples = {}
for fn in glob('./subsampled_clusters/*.vtp'):
    cluster_dir, subsample_info = fn.split('\\')[-1].split('-')[1:]
    cluster_id = '_'.join(subsample_info.split('_')[:2])
    region_id = '_'.join(subsample_info.split('_')[2:4]).split('.')[0]
    if cluster_dir + '/' + cluster_id in subsamples:
        subsamples[cluster_dir + '/' + cluster_id].add(region_id)
    else:
        subsamples[cluster_dir + '/' + cluster_id] = set([region_id])

"""
Generate an adjacency matrix, where each region is a node, and each edge stores these stats:
    - a list of all clusters that contain the cooresponding region pair
    - a list of the TOTAL NoS of each cluster that contains the corresponding region pair
    - a list of the NoS between the corresponding region pair for each cluster that contains the coresponding region pair
    - same as previous 2 lists, but for number of points instead of number of streamlines
"""
regions = sorted(gm_mapping.keys())
i = 0
atlas_weights = {} # adjaceny matrix
ids = [] # column IDs
times = []
for regionA in regions:
    regionA = regionA.replace('-', '')
    time_region_start = time.time()
    for regionB in regions:
        regionB = regionB.replace('-', '')

        region_pair_id = regionA + '_' + regionB

        cluster_list = []
        cluster_NoS = []
        cluster_NoS_R12 = []
        cluster_NoP= []
        cluster_NoP_R12 = []

        for cluster_fn in cluster_fns:
            if region_pair_id in subsamples[cluster_fn]:

                cluster_list.append(cluster_fn)
                cluster_NoS.append(streamline_count('./separated_atlas/' + cluster_fn + '.vtp'))
                subsample_fn = './subsampled_clusters/separated_atlas-' + cluster_fn.replace('/', '-') + '_' + region_pair_id + '.vtp'
                cluster_NoS_R12.append(streamline_count(subsample_fn))

                cluster_NoP.append(point_count('./separated_atlas/' + cluster_fn + '.vtp'))
                cluster_NoP_R12.append(point_count(subsample_fn))
        if regionA not in atlas_weights:
            atlas_weights[regionA] = {}
        atlas_weights[regionA][regionB] = [cluster_list, cluster_NoS, cluster_NoS_R12, cluster_NoP, cluster_NoP_R12]

    times.append(time.time() - time_region_start)
    i+=1
    #print("%d/%d = (%.2f mins on average, i.e. %.2f for all remaining regions)" % (i,len(regions),sum(times)/len(times)/60, sum(times)/len(times)/60*(111-i)))

with open('weights.json', 'w') as f:
    json.dump(atlas_weights, f)
with open('weights.pkl', 'wb') as f:
    pickle.dump(atlas_weights, f)
    

t1 = time.time()
print('Time for weight computation: %.2f minutes' % ((t1 - t0)/60))


"""
Now using the stats we've computed, which we will refer to as the 'atlas'. We want to estimate 
a connectome for each subject. To estimate the NoS between regionA and regionB for a specific 
subject, we iterate over all clusters that connect regionA and regionB in our cluster data, and
compute a sum of the subject's cluster's total NoS, weighted by the ratio of the NoS connecting 
regionA and regionB in the atlas's cluster to the total NoS in the atlas's cluster.
"""
t2 = time.time()
connectome = {}
times = []
for subject in subjects.keys():
    tx = time.time()
    connectome[subject] = {}
    i = 0
    time_start = time.time()
    for regionA in regions:
        regionA = regionA.replace('-', '')
        for regionB in regions:
            regionB = regionB.replace('-', '')

            region_pair_id = regionA + '_' + regionB
            connecting_clusters, cluster_NoS, cluster_NoS_R12, cluster_NoP, cluster_NoP_R12 = atlas_weights[regionA][regionB]

            sl_weight = 0
            points_weight = 0
            for i in range(len(connecting_clusters)):
                cluster = connecting_clusters[i]

                # Compute streamline weight
                cluster_total_sl_atlas = cluster_NoS[i]
                cluster_connecting_sl_atlas = cluster_NoS_R12[i]
                cluster_total_sl_subject = subjects[subject][cluster]['Num_Fibers']
                cluster_level_weight = cluster_connecting_sl_atlas / cluster_total_sl_atlas
                region_level_weight = cluster_connecting_sl_atlas / sum(cluster_NoS_R12)
                #sl_weight += cluster_total_sl_subject * cluster_level_weight * region_level_weight
                sl_weight += cluster_total_sl_subject * cluster_level_weight

                # Compute FA weight
                cluster_total_points_atlas = cluster_NoP[i]
                cluster_connecting_points_atlas = cluster_NoP_R12[i]
                cluster_FA_subject = subjects[subject][cluster]['FA1.Mean']
                cluster_level_weight = cluster_connecting_points_atlas / cluster_total_points_atlas
                region_level_weight = cluster_connecting_points_atlas / sum(cluster_NoP_R12)
                #points_weight += cluster_FA_subject * cluster_level_weight * region_level_weight
                points_weight += cluster_FA_subject * region_level_weight
        
            if regionA not in connectome[subject]:
                connectome[subject][regionA] = {}

            if len(connecting_clusters) == 0:
                connectome[subject][regionA][regionB] = [0, 0]
            else:
                connectome[subject][regionA][regionB] = [sl_weight, points_weight]
            #else:
            #    connectome[subject][regionA][regionB] = [sl_weight / len(connecting_clusters), points_weight / len(connecting_clusters)]

    i += 1
    times.append(time.time() - time_start)
    #print("%d/%d = (%.2f mins on average, i.e. %.2f for all remaining regions)" % (i,len(regions),sum(times)/len(times)/60, sum(times)/len(times)/60*(111-i)))

t3 = time.time()
print('Time for connectome computation: %.2f minutes' % ((t3 - t2)/60))

# Generate matrix
print('Creating result matrix...')
for subject in subjects:
    result = []
    for regionA in regions:
        regionA = regionA.replace('-', '')
        result.append([])
        for regionB in regions:
            regionB = regionB.replace('-', '')
            if len(result[-1]) == 0:
                result[-1] = [connectome[subject][regionA][regionB]]
            else:
                result[-1].append(connectome[subject][regionA][regionB])

    # Converting to numpy
    #print('Converting to numpy...')
    result = np.array(result)
    np.save('./results/' + subject + '.npy', result)
