import torch
import sys
import csv
import random
import numpy as np
from torch_geometric.data import Data

import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from glob import glob
import matplotlib.pyplot as plt

def CustomLoss(output, target):
    criterion = nn.MSELoss()
    return criterion(output,target)

# Return a subject->label mapping
def get_labels_mapping(f):
    result = {}
    with open(f, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            result[row[0]] = int(row[1])
    return result

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='/Users/aritchetchenian/Desktop/graph_cnn'):
        self.subjects = []

        subject_ids = [x.split('/')[-1].split('.')[0] for x in sorted(glob(dataset_dir + '/edge_features/*.npy'))]
        self.labels = get_labels_mapping(dataset_dir + '/ages.csv')

        for subject in subject_ids:
            edge_features = np.load(dataset_dir + '/edge_features/' + subject + '.npy')
            node_features = np.load(dataset_dir + '/node_features/' + subject + '.npy')
            edge_index = np.load(dataset_dir + '/edge_index/' + subject + '.npy')
            label = self.labels[subject]
            self.subjects.append(Data(x=torch.from_numpy(node_features).float(), edge_index=torch.from_numpy(edge_index).long(), edge_attr=torch.from_numpy(edge_features).float(), y=label))
            #self.subjects.append([torch.from_numpy(node_features), torch.from_numpy(edge_index), torch.from_numpy(edge_features), label])

    def __getitem__(self, idx):
        return self.subjects[idx]

    def __len__(self):
        return len(self.subjects)

class CustomModel(torch.nn.Module):
    def __init__(self, num_features=1, hidden_size=32, target_size=1, num_edge_features=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.num_edge_features = num_edge_features
        self.convs = [GATConv(self.num_features, self.hidden_size, edge_dim = self.num_edge_features)] + [GATConv(self.hidden_size, self.hidden_size, edge_dim=self.num_edge_features) for x in range(int(sys.argv[4])-1)]
        self.linear = nn.Linear(self.hidden_size, self.target_size)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    # x          = num nodes x num_node_features (i.e. list of all node features)
    # edge_index = 2 x num_edges (i.e. a list of all edges in the graph; if undirected, put both e.g. [0,1] and [1,0])
    # edge_attr  = num_edges x num_edge_features
    def forward(self, data):
        node_features, edge_index, edge_features = data.x, data.edge_index, data.edge_attr
        x = self.convs[0](node_features, edge_index, edge_attr=edge_features) # adding edge features here!
        for conv in self.convs[1:-1]:
            x = conv(x, edge_index, edge_attr=edge_features) # adding edge features here!
            x = self.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index, edge_attr=edge_features)

        batch = [[x for i in range(111)] for x in range(hyperparams['batch_size'])]
        batch = [j for i in batch for j in i]
        x = global_add_pool(x, batch=torch.tensor(batch))
        x = self.linear(x)

        return self.relu(x) 

''' 
Train model with given hyperparams dict.

Saves the following CSVs over the course of training:
1. the loss trajectory: the val and train loss every save_loss_interval epochs at
   filename 'results/{name_prefix}_{learning_rate}_train.csv' e.g. 'results/baseline_0.05_train.csv'
2. every save_model_interval save both the model at e.g. 'models/baseline_0.05_0_out_of_1000.pt`
   and the predicted values vs actual values in `results/baseline_0.05_0_out_of_1000_prediction.csv' on the test data.
'''

#model = GNNModel()
model = CustomModel(hidden_size=int(sys.argv[3]))

#uuunode_features = torch.ones(111,1).float()
#edge_index = torch.ones(2,111*111).int()
#edge_features = torch.ones(111*111, 2).float()
#result = model.forward(node_features, edge_index, edge_features)
#print(result)

hyperparams = {
    'batch_size' : int(sys.argv[1]),
    'save_loss_interval' : 10,
    'print_interval' : 50,
    'save_model_interval' : 250,
    'n_epochs' : 10,
    'learning_rate' : float(sys.argv[2]),
    'train_split': 0.8,
}

learning_rate = hyperparams['learning_rate']
batch_size = hyperparams['batch_size']
n_epochs = hyperparams['n_epochs']
save_loss_interval = hyperparams['save_loss_interval']
print_interval = hyperparams['print_interval']
save_model_interval = hyperparams['save_model_interval']

NUM_TRAIN = 745 # number of training items
NUM_VAL = 160 # number of valid items
NUM_TEST = 160 # number of valid items

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

whole_dataset = CustomDataset('./dataset')
train_size = int(hyperparams['train_split'] * len(whole_dataset))
valid_size = len(whole_dataset) - train_size
train_dataset, valid_dataset = random_split(whole_dataset, [train_size, valid_size])

trainloader = DataLoader(train_dataset, shuffle=True, batch_size=hyperparams['batch_size'], drop_last=True)
validloader = DataLoader(valid_dataset, shuffle=True, batch_size=hyperparams['batch_size'], drop_last=True)

losses = []
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    batch_count = 0
    train_outs = []
    for data in trainloader:
        optimizer.zero_grad()
        out = model(data)
        train_outs += list(out.detach().numpy().flatten())
        loss = CustomLoss(out, torch.unsqueeze(data.y.float(),1))
        epoch_loss += loss.item() 
        loss.backward()
        optimizer.step()
        losses.append(loss)
        batch_count += 1


    model.eval()
    valid_loss = 0
    valid_count = 0
    outs = []
    with torch.no_grad():
        for data in validloader:
            out = model(data)
            outs += list(out.numpy().flatten())
            loss = CustomLoss(out, torch.unsqueeze(data.y.float(),1))
            valid_loss += loss.item()
            valid_count += 1
    
    print("TRAIN Min: %.2f, Max: %.2f, Mean: %.2f" % (min(train_outs), max(train_outs), sum(train_outs)/len(train_outs)))
    print("VALID Min: %.2f, Max: %.2f, Mean: %.2f" % (min(outs), max(outs), sum(outs)/len(outs)))
    print("Train: %.3f\tValid: %.3f" % (epoch_loss/batch_count, valid_loss/valid_count))

result = np.array([min(train_outs), max(train_outs), sum(train_outs)/len(train_outs), min(outs), max(outs), sum(outs)/len(outs), epoch_loss/batch_count, valid_loss/valid_count])
save_string = '_'.join(sys.argv[1:]) + '.npy'
np.save('./logs/' + save_string, result)
