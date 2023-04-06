import torch
import os
import csv
import numpy as np
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
from glob import glob
import visdom
from visdom_scripts.vis import VisdomLinePlotter
from argparse import ArgumentParser
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# MSE Loss function
def MSE(output, target):
    criterion = nn.MSELoss()
    return criterion(output,target)

# A loss function that combines MSE and Pearson correlation coefficient (R)
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def pearson_corr_coef(self, x, y, eps=1e-8):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        num = torch.sum(vx * vy)
        denom = torch.sqrt(torch.sum(vx ** 2) * torch.sum(vy ** 2)) + eps
        corr = num / denom
        return 1 - corr

    def forward(self, y_pred, y_true):
        mse_loss = self.mse_loss(y_pred, y_true)
        pearson_loss = self.pearson_corr_coef(y_pred, y_true)
        combined_loss = self.alpha * mse_loss + (1 - self.alpha) * pearson_loss
        return combined_loss

def MSE_corr_loss(output, target):
    criterion = CombinedLoss()
    return criterion(output, target)
 
# Return a subject->label mapping for float labels
def get_labels_mapping(f):
    result = {}
    with open(f, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            result[row[0]] = float(row[1])
    return result

# Return a subject->label mapping for string labels 
def get_gender_mapping(f):
    result = {}
    with open(f, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if row[1].lower() == 'm':
                result[row[0]] = 0
            else:
                result[row[0]] = 1
    return result

# Basic data augmentation that adds uniform noise to node and edge features
def aug(d):
    # Add noise
    c = d.clone()
    c.x += torch.rand(c.x.shape[0], c.x.shape[1]) * 0.2
    c.edge_attr += torch.rand(c.edge_attr.shape[0], c.edge_attr.shape[1]) * 0.2

    # Re-normalise
    c.x = (c.x - torch.min(c.x)) / (torch.max(c.x) - torch.min(c.x))
    c.edge_attr = (c.edge_attr - torch.min(c.edge_attr)) / (torch.max(c.edge_attr) - torch.min(c.edge_attr))

    return c

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='/Users/aritchetchenian/Desktop/graph_cnn'):
        self.subjects = [] # store all data here

        subject_ids = [os.path.split(x)[-1].split('.')[0] for x in sorted(glob(dataset_dir + '/edge_features/*.npy'))]
        self.labels = get_labels_mapping(dataset_dir + '/pic_vocab_unadj.csv')

        for subject in subject_ids:
            edge_features = np.load(dataset_dir + '/edge_features/' + subject + '.npy')
            node_features = np.load(dataset_dir + '/node_features/' + subject + '.npy')
            edge_index = np.load(dataset_dir + '/edge_index/' + subject + '.npy')
            label = self.labels[subject]

            # Normalise edge features into range [0,1]
            for i in range(edge_features.shape[-1]):
                edge_features[:,i] = (edge_features[:,i] - np.min(edge_features[:,i])) / (np.max(edge_features[:,i]) - np.min(edge_features[:,i]))

            # Extract a single node feature
            #node_features = np.reshape(np.count_nonzero(node_features,axis=1),(-1,1))

            # Normalise node features into range [0,1]
            node_features = (node_features - np.min(node_features)) / (np.max(node_features) - np.min(node_features))

            # Apply Thresholding
            #edge_features[edge_features<0.1] = 0
            #node_features[node_features<0.1] = 0
            #node_features = (node_features - np.min(node_features)) / (np.max(node_features) - np.min(node_features))
            #for i in range(edge_features.shape[-1]):
            #    edge_features[:,i] = (edge_features[:,i] - np.min(edge_features[:,i])) / (np.max(edge_features[:,i]) - np.min(edge_features[:,i]))

            self.subjects.append(Data(x=torch.from_numpy(node_features).float(), edge_index=torch.from_numpy(edge_index).long(), edge_attr=torch.from_numpy(edge_features).float(), y=label))

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
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        # Dynamic number of GATConv layers
        self.convs = nn.ModuleList([GATConv(self.num_features, self.hidden_size, edge_dim = self.num_edge_features)] 
            + [GATConv(self.hidden_size, self.hidden_size, edge_dim=self.num_edge_features) for x in range(args.depth-1)])

        self.linear = nn.Linear(self.hidden_size, 1)

        """
        # Dynamic number of GATConv layers where num features doubles every layer
        self.convs = nn.ModuleList([GATConv(self.num_features, self.hidden_size, edge_dim = self.num_edge_features)] 
            + [GATConv(self.hidden_size*(2**x), self.hidden_size*(2**(x+1)), edge_dim=self.num_edge_features) for x in range(args.depth-1)])

        self.linear = nn.Linear(self.hidden_size * (2**(args.depth-1)), 1)
        """

    """
    - node_features = num nodes x num_node_features (i.e. list of all node features)
    - edge_index    = 2 x num_edges (i.e. a list of all edges in the graph; if undirected, 
                      put both e.g. [0,1] and [1,0])
    - edge_attr     = num_edges x num_edge_features
    """
    def forward(self, data):
        # Get the data from the Data object
        node_features, edge_index, edge_features = data.x, data.edge_index, data.edge_attr

        # Feed through first GATConv layer
        x = self.convs[0](node_features, edge_index, edge_attr=edge_features)
        x = self.relu(x)
        x = self.dropout(x)

        # Feed through any additional GATConv layers
        for conv in self.convs[1:-1]:
            x = conv(x, edge_index, edge_attr=edge_features) # adding edge features here!
            x = self.relu(x)
            x = self.dropout(x)

        # Feed through the final layer separately from the loop
        # so that we can exclude ReLU and dropout
        x = self.convs[-1](x, edge_index, edge_attr=edge_features)


        """
        Compile all the features into a single feature for classification via 
        Sum all node hidden features to produce a single hidden feature for the entire graph
          - Before 'global_add_pool': num_nodes * batch_size x hidden_features
          - After 'global_add_pool':  batch_size x hidden_features

        The 'batch' variable is just a list of batch ids for every node. It is necessary because
        torch geometric concats all node features from all graphs in the current batch into a long
        feature vector, so we need to tell the global_add_pool function which nodes belong to which
        batches.
          - e.g. batch = [0,0,0,...,0,1,1,...,1] with 84 0s and 84 1s would assign the first 84
            hidden features to the 1st batch, then the second 84 hidden features to the 2nd batch
        """
        batch = torch.repeat_interleave(torch.arange(args.batch_size, device=device), 84)
        x = global_add_pool(x, batch=batch)
            
        # Feed into a linear layer
        x = self.linear(x)

        # We are predicting age, so output should be positive. Hence, we can just apply a relu to
        #  make it easier for the network
        return self.relu(x) 

# Define the arguments
parser = ArgumentParser(description="Arguments for model training.")
parser.add_argument("-b", "--batch_size", help="Batch size.", default=1, type=int)
parser.add_argument("-e", "--epochs", help="Number of epochs.", default=100, type=int)
parser.add_argument("-lr", "--learning_rate", help="Learning rate.", default=0.001, type=float)
parser.add_argument("-u", "--hidden_units", help="Hidden units.", default=32, type=int)
parser.add_argument("-d", "--depth", help="Num conv. layers in the network.", default=2, type=int)
parser.add_argument("-vis", "--vis_mode", help="Presence of this flag enables plotting/visualisation of results.", action='store_true')
parser.add_argument("-s", "--save_name", help="Folder in which to save all results to.", type=str)
args = parser.parse_args()

# Create the results directory
while os.path.exists('./results/' + args.save_name):
    args.save_name = input("Already exists. Enter new save name:")
os.mkdir('./results/' + args.save_name)

# Print all specified arguments
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
args_string = '_'.join([str(getattr(args, arg)) for arg in vars(args)]) # create string for a unique ID

# Choosing a device (CPU vs. GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Training on: ", device)

# Create the dataset
whole_dataset = CustomDataset('./84x84_dataset')
#whole_dataset = CustomDataset('./84x84_dataset_0.05_thresh')
#whole_dataset = CustomDataset('./84x84_dataset_0.9_threshold')

"""
shuffle=True will shuffle the items in the dataset before splitting them into 5 folds
e.g. if you set it to false, and you have 10 items, you will always get:
            train subjects      valid subjects
    fold 1: [2,3,4,5,6,7,8,9]   [0,1]
    fold 2: [0,1,4,5,6,7,8,9]   [2,3]
    fold 3: [0,1,2,3,6,7,8,9]   [4,5]
    ...
    ...
if you set it to true, the valid subjects will not be in that order.

Since we've set a random_state (seed), we'll get the same order every time we run,
but it's still good in principle to use the shuffle flag
"""
kf = KFold(n_splits=5, shuffle=True, random_state=666)

# fold      = integer fold number (starting at 0)
# train_idx = list of indexes for items in the training fold
# val_idx   = list of indexes for items in the validation fold
for fold, (train_idx, val_idx) in enumerate(kf.split(whole_dataset)):
    train_dataset = Subset(whole_dataset, train_idx)
    valid_dataset = Subset(whole_dataset, val_idx)

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Visdom plotting initialisation
    if args.vis_mode:
        loss_plotter = VisdomLinePlotter(env_name='Age Prediction')
        score_plotter = VisdomLinePlotter(env_name='Age Prediction')
        vis = visdom.Visdom()
        train_opts = dict(title='Train Histogram', xtickmin=90, xtickmax=160)
        valid_opts = dict(title='Valid Histogram', xtickmin=90, xtickmax=160)
        truth_opts = dict(title='Truth Histogram', xtickmin=90, xtickmax=160)
        train_win = None
        valid_win = None
        truth_win = None
        train_scatter_win = None
        valid_scatter_win = None
        train_image = None
        valid_image = None

    # Initialising model
    model = CustomModel(num_features=84, hidden_size=args.hidden_units, num_edge_features=2)
    model.to(device)
    print(model)

    # Initialising the optimiser/scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=15)

    # Main training/validation loop for the current fold of cross-validation
    valid_losses, train_losses = [], []
    valid_stats, train_stats = [], []
    best_valid_loss = None
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
            #loss = CustomLoss(out, torch.unsqueeze(data.y.float(),1))
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Store output/metrics
            train_outs += list(out.cpu().detach().numpy().flatten())
            train_truths += list(data.y.cpu().detach().numpy())
            train_loss += loss.item() 
            train_count += 1
        train_node_features = data.x[:84].cpu().detach().numpy()

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
                #loss = CustomLoss(out, torch.unsqueeze(data.y.float(),1))

                # Store output/metrics
                valid_outs += list(out.cpu().numpy().flatten())
                valid_truths += list(data.y.cpu().detach().numpy())
                valid_loss += loss.item()
                valid_count += 1
        valid_node_features = data.x[:84].cpu().detach().numpy()

        # Step the scheduler and print the current LR
        scheduler.step(valid_loss)
        print('Current learning rate: %f' % (optimizer.param_groups[0]['lr']))

        train_r, _ = pearsonr(train_outs, train_truths)
        valid_r, _ = pearsonr(valid_outs, valid_truths)
        
        train_mae = np.mean(np.abs(np.array(train_outs) - np.array(train_truths)))
        valid_mae = np.mean(np.abs(np.array(valid_outs) - np.array(valid_truths)))
        
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
            score_plotter.plot('score', 'valid R', 'Metric Curves', epoch, valid_r, yaxis_type='linear')
            score_plotter.plot('score', 'train R', 'Metric Curves', epoch, train_r, yaxis_type='linear')
            score_plotter.plot('score', 'train MAE', 'Metric Curves', epoch, train_mae, yaxis_type='linear')
            score_plotter.plot('score', 'valid MAE', 'Metric Curves', epoch, valid_mae, yaxis_type='linear')

            # Plot the histograms
            train_win = vis.histogram(train_outs, win=train_win, opts=train_opts, env='Age Prediction')
            valid_win = vis.histogram(valid_outs, win=valid_win, opts=valid_opts, env='Age Prediction')
            truth_win = vis.histogram(train_truths + valid_truths, win=truth_win, opts=truth_opts, env='Age Prediction')

            # Plot the correlation coefficient
            train_scatter_win = vis.scatter(X=np.stack([train_outs, train_truths],axis=1), win=train_scatter_win, opts=dict(markersize=5, title=f"Train Corr: {train_r:.2f}"), env='Age Prediction')
            valid_scatter_win = vis.scatter(X=np.stack([valid_outs, valid_truths],axis=1), win=valid_scatter_win, opts=dict(markersize=5, title=f"Valid Corr: {valid_r:.2f}"), env='Age Prediction')

            # Show an example
            train_image = vis.image(train_node_features, opts=dict(title="Train Node Features", width=300,height=300), env='Age Prediction', win=train_image)
            valid_image = vis.image(valid_node_features, opts=dict(title="Valid Node Features", width=300, height=300), env='Age Prediction', win=valid_image)

        # Update metrics
        valid_losses.append(valid_loss/valid_count)
        train_losses.append(train_loss/train_count)
        valid_stats.append([min(valid_outs), max(valid_outs), sum(valid_outs)/len(valid_outs), valid_r])
        train_stats.append([min(train_outs), max(train_outs), sum(train_outs)/len(train_outs), train_r])

        # Print the current epoch
        print(epoch)

        # Save the current model if it has the lowest validation loss
        update_loss = False
        if best_valid_loss is None:
            update_loss = True
        elif valid_loss/valid_count < best_valid_loss:
            update_loss = True
        if update_loss:
            best_valid_loss = valid_loss / valid_count

            # Save the model
            save_object = {
                # state dicts
                'model_state_dict': model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),

                # useful info
                'training_epoch': epoch,
                'total_epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.learning_rate,
                'hidden_units': args.hidden_units,
                'depth': args.depth,
                'vis_mode': args.vis_mode,
                'save_name': args.save_name,

                # cross-val info
                'fold': fold,
                'train_idx': train_idx,
                'val_idx': val_idx,

                # metric info
                'train_outs': train_outs,
                'valid_outs': valid_outs,
                'train_r': train_r,
                'valid_r': valid_r,
                'train_truths': train_truths,
                'valid_truths': valid_truths,
                'valid_losses': valid_losses,
                'train_losses': train_losses,
            }
            torch.save(save_object, './results/' + args.save_name + '/fold_' + str(fold) + '_best_model_checkpoint.pth')
