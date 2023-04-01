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

import visdom
from visdom_scripts.vis import VisdomLinePlotter

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
            #node_features = np.load(dataset_dir + '/node_features/' + subject + '.npy')
            node_features = np.reshape(np.array([i for i in range(84)]),(84,1))
            #print(node_features.shape)

            edge_index = np.load(dataset_dir + '/edge_index/' + subject + '.npy')

            # normalise into range [0,1]
            edge_features = (edge_features - np.min(edge_features)) / (np.max(edge_features) - np.min(edge_features))

            # add random noise
            #edge_features += np.random.rand(edge_features.shape) * 0.001

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
        self.convs = nn.ModuleList([GATConv(self.num_features, self.hidden_size, edge_dim = self.num_edge_features)] + [GATConv(self.hidden_size, self.hidden_size, edge_dim=self.num_edge_features) for x in range(int(sys.argv[4])-1)])
        self.linear = nn.Linear(self.hidden_size, self.target_size)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    # x          = num nodes x num_node_features (i.e. list of all node features)
    # edge_index = 2 x num_edges (i.e. a list of all edges in the graph; if undirected, put both e.g. [0,1] and [1,0])
    # edge_attr  = num_edges x num_edge_features
    def forward(self, data):
        node_features, edge_index, edge_features = data.x, data.edge_index, data.edge_attr
        x = self.convs[0](node_features, edge_index, edge_attr=edge_features) # adding edge features here!
        x = self.relu(x)
        x = self.dropout(x)
        for conv in self.convs[1:-1]:
            x = conv(x, edge_index, edge_attr=edge_features) # adding edge features here!
            x = self.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index, edge_attr=edge_features)

        batch = torch.repeat_interleave(torch.arange(hyperparams['batch_size'], device=device), 84)
        #batch = [[x for i in range(84)] for x in range(hyperparams['batch_size'])]
        #batch = [j for i in batch for j in i]
        #x = global_add_pool(x, batch=torch.tensor(batch))
        x = global_add_pool(x, batch=batch)
        x = self.linear(x)

        return self.relu(x) 
        #return x

''' 
Train model with given hyperparams dict.

Saves the following CSVs over the course of training:
1. the loss trajectory: the val and train loss every save_loss_interval epochs at
   filename 'results/{name_prefix}_{learning_rate}_train.csv' e.g. 'results/baseline_0.05_train.csv'
2. every save_model_interval save both the model at e.g. 'models/baseline_0.05_0_out_of_1000.pt`
   and the predicted values vs actual values in `results/baseline_0.05_0_out_of_1000_prediction.csv' on the test data.
'''

loss_plotter = VisdomLinePlotter(env_name='Age Prediction')
score_plotter = VisdomLinePlotter(env_name='Age Prediction')
vis = visdom.Visdom()
train_opts = dict(title='Train Histogram', xtickmin=20, xtickmax=40)
valid_opts = dict(title='Valid Histogram', xtickmin=20, xtickmax=40)
train_win = None
valid_win = None
#hist = vis.histogram(X=torch.ones(1000), env='Age Prediction', opts=dict(title='Output Histogram'))

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#model = GNNModel()
model = CustomModel(num_features=1, hidden_size=int(sys.argv[3]))
model.to(device)


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
    'n_epochs' : 20,
    'learning_rate' : float(sys.argv[2]),
    'train_split': 0.9,
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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
#loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

whole_dataset = CustomDataset('./84x84_dataset')
train_size = int(hyperparams['train_split'] * len(whole_dataset))
valid_size = len(whole_dataset) - train_size
train_dataset, valid_dataset = random_split(whole_dataset, [train_size, valid_size])

trainloader = DataLoader(train_dataset, shuffle=True, batch_size=hyperparams['batch_size'], drop_last=True)
validloader = DataLoader(valid_dataset, shuffle=True, batch_size=hyperparams['batch_size'], drop_last=True)

valid_losses = []
train_losses = []
valid_stats = []
train_stats = []
for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    train_count = 0
    train_outs = []
    for data in trainloader:
        optimizer.zero_grad()

        data = data.to(device)

        out = model(data)
        train_outs += list(out.cpu().detach().numpy().flatten())
        loss = CustomLoss(out, torch.unsqueeze(data.y.float(),1))
        train_loss += loss.item() 
        loss.backward()
        optimizer.step()
        train_count += 1


    model.eval()
    valid_loss = 0
    valid_count = 0
    outs = []
    with torch.no_grad():
        for data in validloader:
            data = data.to(device)

            out = model(data)
            outs += list(out.cpu().numpy().flatten())
            loss = CustomLoss(out, torch.unsqueeze(data.y.float(),1))
            valid_loss += loss.item()
            valid_count += 1

    scheduler.step(valid_loss)
    print('Current learning rate: %f' % (optimizer.param_groups[0]['lr']))
    
    loss_plotter.plot('score', 'valid loss', 'Metric Curves', epoch, valid_loss/valid_count)
    loss_plotter.plot('score', 'train loss', 'Metric Curves', epoch, train_loss/train_count)

    score_plotter.plot('score', 'train min', 'Metric Curves', epoch, min(train_outs))
    score_plotter.plot('score', 'train max', 'Metric Curves', epoch, max(train_outs))
    score_plotter.plot('score', 'train mean', 'Metric Curves', epoch, sum(train_outs)/len(train_outs))

    score_plotter.plot('score', 'valid min', 'Metric Curves', epoch, min(outs))
    score_plotter.plot('score', 'valid max', 'Metric Curves', epoch, max(outs))
    score_plotter.plot('score', 'valid mean', 'Metric Curves', epoch, sum(outs)/len(outs))

    train_win = vis.histogram(train_outs, win=train_win, opts=train_opts, env='Age Prediction')
    valid_win = vis.histogram(outs, win=valid_win, opts=valid_opts, env='Age Prediction')

    valid_losses.append(valid_loss/valid_count)
    train_losses.append(train_loss/train_count)
    valid_stats.append([min(outs), max(outs), sum(outs)/len(outs)])
    train_stats.append([min(train_outs), max(train_outs), sum(train_outs)/len(train_outs)])

    print(epoch)

#result = np.array([min(train_outs), max(train_outs), sum(train_outs)/len(train_outs), min(outs), max(outs), sum(outs)/len(outs), train_loss/train_count, valid_loss/valid_count])
save_string = '_'.join(sys.argv[1:]) + '.npy'

np.save('./logs/valid_loss_' + save_string, np.array(valid_loss))
np.save('./logs/train_loss_' + save_string, np.array(train_loss))
np.save('./logs/valid_stats_' + save_string, np.array(valid_stats))
np.save('./logs/train_stats_' + save_string, np.array(train_stats))
