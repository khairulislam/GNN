import os, torch
import torch_geometric
import torch.nn.functional as F
import torch_geometric.utils as utils
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, GATConv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from early_stopping import EarlyStopping

def visualize(h, color):
    z = TSNE(
        n_components=2, init='pca', 
        learning_rate='auto', random_state=1234
    ).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()
    
# Load the Cora dataset:
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}') # 1
print(f'Number of features: {dataset.num_features}') # 1433
print(f'Number of classes: {dataset.num_classes}') # 7

data = dataset[0]  # Get the first graph object.

print()
# data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], 
# train_mask=[2708], val_mask=[2708], test_mask=[2708])
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}') # 2708
print(f'Number of edges: {data.num_edges}') # 10556
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}') # 3.90
print(f'Number of training nodes: {data.train_mask.sum()}') # 140
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}') # 0.05
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x):
        x = self.lin1(x).relu()
        x = torch.nn.functional.dropout(x, 0.5, training=self.training)
        x = self.lin2(x).relu()
        return x
    
model = MLP(
    in_channels=dataset.num_features, 
    hidden_channels=64, 
    out_channels=dataset.num_classes
)

# In theory, we should be able to infer the category of a 
# document solely based on its content, *i.e.* its bag-of-words 
# feature representation, without taking any relational 
# information into account.

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def test():
      model.eval()
      out = model(data.x)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc
  
for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
    
# The MLP performs rather bad because it suffers from overfitting.
# only having access to the small amount oftraining nodes
# it also fails to imcorporate an important bias that the cited
# papers are very likely related to the category of a document.
# that is where the GNN comes into play
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
model = GCN(
    in_channels=dataset.num_features, 
    hidden_channels=8, 
    out_channels=dataset.num_classes
)

# lets visualize the untrained embedding
# TSNE will embed the 7-dimensional node embeddings into 2-dimensions
model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    
    test_correct = pred[dataset.test_mask] == data.y[dataset.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(dataset.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc

for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
# this performs much better than just MLP

# lets visualize the trained embedding
out = model(data.x, data.edge_index)
visualize(out, color=data.y)

# Can we do even better?
# How about attention?
class GAT(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, 
        out_channels
    ):
        super().__init__()
        
        self.in_head = 8
        self.out_head = 1
        torch.manual_seed(1234567)
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=self.in_head)
        self.conv2 = GATConv(
            hidden_channels*self.in_head, out_channels, 
            heads=self.out_head, concat=False
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
model = GAT(
    in_channels=dataset.num_features, 
    hidden_channels=32, 
    out_channels=dataset.num_classes
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
# Reduce the learning rate on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.1, 
    patience=5, 
    min_lr=1e-6
)

early_stopping = EarlyStopping(patience=5, verbose=True, path=None)

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss
  
def val(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask], data.y[mask])
    return loss

def test(mask):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
      acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
      return acc


for epoch in range(1, 201):
    loss = train()
    val_loss = val(data.val_mask)
    
    if epoch % 10 == 0:
        test_loss = val(data.test_mask)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f}')
    
    scheduler.step(val_loss)
    early_stopping(val_loss=val_loss, model=model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    
test_acc = test(data.test_mask)
print(f'Test accuracy: {test_acc:.4f}')