import torch
import os
import csv
import numpy as np
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from glob import glob
import visdom
from visdom_scripts.vis import VisdomLinePlotter
from argparse import ArgumentParser
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import io

# From GPT-4:
def visualize_graph(node_connections, edge_values):
    # Check if input shapes are correct
    assert len(node_connections) == len(edge_values), "Input lengths mismatch"
    assert len(node_connections[0]) == 2, "Node connections shape mismatch"

    # Create an empty graph
    G = nx.Graph()

    # Add edges with their corresponding values
    for i, (node1, node2) in enumerate(node_connections):
        G.add_edge(node1, node2, value=edge_values[i])

    # Draw the graph
    pos = nx.spring_layout(G, seed=42, k=0.5)  # Increase 'k' for more separation between nodes
    plt.figure(figsize=(15, 15))  # Adjust the figure size
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10, linewidths=1.0, edgecolors='black')
    labels = nx.get_edge_attributes(G, 'value')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

    # Save the figure in a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)  # Increase dpi for higher resolution images
    buf.seek(0)

    # Convert the BytesIO object to a NumPy array
    image = Image.open(buf)
    image_np = np.array(image)

    # Close the buffer and clear the figure
    buf.close()
    plt.clf()

    return np.swapaxes(np.swapaxes(image_np[:,:,:3], 0,2), 1,2)

def MSE(output, target):
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

"""
def aug(x):
    x.x += np.float32(np.random.rand(x.x.shape[0], x.x.shape[1]) * 100)
    x.edge_attr += np.float32(np.random.rand(x.edge_attr.shape[0], x.edge_attr.shape[1]) * 100)
    return x
"""

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='/Users/aritchetchenian/Desktop/graph_cnn'):
        self.subjects = [] # store all data here

        subject_ids = [os.path.split(x)[-1].split('.')[0] for x in sorted(glob(dataset_dir + '/edge_features/*.npy'))]
        self.labels = get_labels_mapping(dataset_dir + '/ages.csv')

        for subject in subject_ids:
            edge_features = np.load(dataset_dir + '/edge_features/' + subject + '.npy')
            node_features = np.load(dataset_dir + '/node_features/' + subject + '.npy')
            edge_index = np.load(dataset_dir + '/edge_index/' + subject + '.npy')
            label = self.labels[subject]

            # normalise each edge feature into range [0,1]
            for i in range(edge_features.shape[-1]):
                edge_features[:,i] = (edge_features[:,i] - np.min(edge_features[:,i])) / (np.max(edge_features[:,i]) - np.min(edge_features[:,i]))

            # normalise node features into range [0,1]
            node_features = (node_features - np.min(node_features)) / (np.max(node_features) - np.min(node_features))

            self.subjects.append(Data(x=torch.from_numpy(node_features).float(), edge_index=torch.from_numpy(edge_index).long(), edge_attr=torch.from_numpy(edge_features).float(), y=label))

    def __getitem__(self, idx):
        #return aug(self.subjects[idx])
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
        self.convs = nn.ModuleList([GATConv(self.num_features, self.hidden_size, edge_dim = self.num_edge_features)] + [GATConv(self.hidden_size, self.hidden_size, edge_dim=self.num_edge_features) for x in range(args.depth-1)])
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

        batch = torch.repeat_interleave(torch.arange(args.batch_size, device=device), 84)
        #batch = [[x for i in range(84)] for x in range(hyperparams['batch_size'])]
        #batch = [j for i in batch for j in i]
        #x = global_add_pool(x, batch=torch.tensor(batch))
        x = global_add_pool(x, batch=batch)
        x = self.linear(x)

        return self.relu(x) 
        #return x

# Define the arguments
parser = ArgumentParser(description="Arguments for model training.")
parser.add_argument("-b", "--batch_size", help="Batch size.", default=1, type=int)
parser.add_argument("-e", "--epochs", help="Number of epochs.", default=100, type=int)
parser.add_argument("-lr", "--learning_rate", help="Learning rate.", default=0.001, type=float)
parser.add_argument("-u", "--hidden_units", help="Hidden units.", default=32, type=int)
parser.add_argument("-d", "--depth", help="Num conv. layers in the network.", default=2, type=int)
parser.add_argument("-t", "--train_split", help="What fraction of data to use as training set? e.g. 0.9.", default=0.9, type=float)
parser.add_argument("-vis", "--vis_mode", help="Presence of this flag enables plotting/visualisation of results.", action='store_true')
args = parser.parse_args()

# Print all specified arguments
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
args_string = '_'.join([str(getattr(args, arg)) for arg in vars(args)]) # create string for a unique ID

# Visdom plotting initialisation
if args.vis_mode:
    loss_plotter = VisdomLinePlotter(env_name='Age Prediction')
    score_plotter = VisdomLinePlotter(env_name='Age Prediction')
    vis = visdom.Visdom()
    train_opts = dict(title='Train Histogram', xtickmin=20, xtickmax=40)
    valid_opts = dict(title='Valid Histogram', xtickmin=20, xtickmax=40)
    truth_opts = dict(title='Truth Histogram', xtickmin=20, xtickmax=40)
    train_win = None
    valid_win = None
    truth_win = None
    train_scatter_win = None
    valid_scatter_win = None
    train_image = None
    valid_image = None
    train_graph_image = None
    valid_graph_image = None

# Choosing a device (CPU vs. GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Training on: ", device)

# Initialising model
model = CustomModel(num_features=84, hidden_size=args.hidden_units)
model.to(device)

# Initialising the optimiser/scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=15)

# Create the dataset
whole_dataset = CustomDataset('./84x84_dataset')
train_size = int(args.train_split * len(whole_dataset))
valid_size = len(whole_dataset) - train_size
train_dataset, valid_dataset = random_split(whole_dataset, [train_size, valid_size])
trainloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)
validloader = DataLoader(valid_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)

# Main training/validation loop
valid_losses, train_losses = [], []
valid_stats, train_stats = [], []
for epoch in range(args.epochs):
    # Training
    model.train()
    train_loss = 0
    train_count = 0
    train_outs = []
    train_truths = []
    for data in trainloader:
        # Send data to device
        data = data.to(device)

        # Reset optimizer
        optimizer.zero_grad()

        # Feed into model
        out = model(data)

        # Compute and backprop the loss
        loss = MSE(out, torch.unsqueeze(data.y.float(),1))
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Store output/metrics
        train_outs += list(out.cpu().detach().numpy().flatten())
        train_truths += list(data.y.cpu().detach().numpy())
        train_loss += loss.item() 
        train_count += 1
    train_node_features = data.x[:84].cpu().detach().numpy()
    #train_graph_im = visualize_graph(np.reshape(data.edge_index.cpu().detach().numpy(), (-1,2)), data.edge_attr.cpu().detach().numpy()[:,0])

    # Validation
    model.eval()
    valid_loss = 0
    valid_count = 0
    valid_outs = []
    valid_truths = []
    with torch.no_grad():
        for data in validloader:
            # Send data to device
            data = data.to(device)

            # Feed into model
            out = model(data)

            # Calculate loss
            loss = MSE(out, torch.unsqueeze(data.y.float(),1))

            # Store output/metrics
            valid_outs += list(out.cpu().numpy().flatten())
            valid_truths += list(data.y.cpu().detach().numpy())
            valid_loss += loss.item()
            valid_count += 1
    valid_node_features = data.x[:84].cpu().detach().numpy()
    #valid_graph_im = visualize_graph(np.reshape(data.edge_index.cpu().detach().numpy(), (-1,2)), data.edge_attr.cpu().detach().numpy()[:,0])

    # Step the scheduler and print the current LR
    scheduler.step(valid_loss)
    print('Current learning rate: %f' % (optimizer.param_groups[0]['lr']))

    train_r, _ = pearsonr(train_outs, train_truths)
    valid_r, _ = pearsonr(valid_outs, valid_truths)
    
    # Plotting
    if args.vis_mode:
        # Plot the losses
        loss_plotter.plot('score', 'valid loss', 'Metric Curves', epoch, valid_loss/valid_count, yaxis_type='log')
        loss_plotter.plot('score', 'train loss', 'Metric Curves', epoch, train_loss/train_count, yaxis_type='log')

        # Plot the metrics
        score_plotter.plot('score', 'train min', 'Metric Curves', epoch, min(train_outs), yaxis_type='linear')
        score_plotter.plot('score', 'train max', 'Metric Curves', epoch, max(train_outs), yaxis_type='linear')
        score_plotter.plot('score', 'train mean', 'Metric Curves', epoch, sum(train_outs)/len(train_outs), yaxis_type='linear')
        score_plotter.plot('score', 'valid min', 'Metric Curves', epoch, min(valid_outs), yaxis_type='linear')
        score_plotter.plot('score', 'valid max', 'Metric Curves', epoch, max(valid_outs), yaxis_type='linear')
        score_plotter.plot('score', 'valid mean', 'Metric Curves', epoch, sum(valid_outs)/len(valid_outs), yaxis_type='linear')

        # Plot the histograms
        train_win = vis.histogram(train_outs, win=train_win, opts=train_opts, env='Age Prediction')
        valid_win = vis.histogram(valid_outs, win=valid_win, opts=valid_opts, env='Age Prediction')
        truth_win = vis.histogram(train_truths + valid_truths, win=truth_win, opts=truth_opts, env='Age Prediction')

        # Plot the correlation coefficient
        train_scatter_win = vis.scatter(X=np.stack([train_outs, train_truths],axis=1), win=train_scatter_win, opts=dict(markersize=5, title=f"Train Corr: {train_r:.2f}"), env='Age Prediction')
        valid_scatter_win = vis.scatter(X=np.stack([valid_outs, valid_truths],axis=1), win=valid_scatter_win, opts=dict(markersize=5, title=f"Valid Corr: {valid_r:.2f}"), env='Age Prediction')

        # Show an example
        print(train_node_features.shape)
        train_image = vis.image(train_node_features, opts=dict(title="Train Node Features"), env='Age Prediction', win=train_image)
        valid_image = vis.image(valid_node_features, opts=dict(title="Valid Node Features"), env='Age Prediction', win=valid_image)
        #train_graph_image = vis.image(train_graph_im, opts=dict(title="Train Graph", width=400, height=400), env='Age Prediction', win=train_graph_image)
        #valid_graph_image = vis.image(valid_graph_im, opts=dict(title="Valid Graph", width=400, height=400), env='Age Prediction', win=valid_graph_image)

    # Update metrics
    valid_losses.append(valid_loss/valid_count)
    train_losses.append(train_loss/train_count)
    valid_stats.append([min(valid_outs), max(valid_outs), sum(valid_outs)/len(valid_outs), valid_r])
    train_stats.append([min(train_outs), max(train_outs), sum(train_outs)/len(train_outs), train_r])

    # Print the current epoch
    print(epoch)

# Save results
save_string = args_string + '.npy'
np.save('./logs/valid_loss_' + save_string, np.array(valid_losses))
np.save('./logs/train_loss_' + save_string, np.array(train_losses))
np.save('./logs/valid_stats_' + save_string, np.array(valid_stats))
np.save('./logs/train_stats_' + save_string, np.array(train_stats))
